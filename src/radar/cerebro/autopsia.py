"""
src/radar/cerebro/autopsia.py — onde os trades ganharam e perderam.

A autópsia inverte a fila do projeto. Antes: `Hipótese → Walk Forward →
Evidence Map → Validada?`. Agora: `Hipótese → Trades → AUTÓPSIA → Mapa de
Contextos → Nova hipótese → Walk Forward`. O objetivo deixa de ser **validar
estratégia** e passa a ser **descobrir contexto**.

Os eixos, os cortes e os pisos estão congelados em
`docs/estudos/PRE_REGISTRO_AUTOPSIA.md` — este módulo os implementa e não os
escolhe. Ler aquele documento antes de mexer aqui.

Três coisas que este módulo faz de propósito, e que são a diferença entre
autópsia e p-hacking:

1. **Deriva os eixos com janela ESTRITAMENTE passada** (`rolling(...).shift(1)`).
   A liquidez e a volatilidade de um trade são as que se conheciam na véspera,
   nunca as do dia — senão a partição usa informação que a decisão não tinha
   (Regra 5, sem look-ahead).

2. **A fonte de contexto é escolhida pelo MERCADO, nunca herdada.**
   `RegimeTimeline` é macro brasileira; cripto usa `detectar_regime_cripto`.
   Cruzar as duas reproduz o `CEREBRO_STORE_CRIPTO_CALENDARIO_B301`, onde
   `dia_util_do_mes` — calendário da B3 — saiu como 6º sinal mais forte do
   cripto. Onde o modelo achou sinal ali, achou artefato.

3. **Célula abaixo do piso sai `NAO_ESTIMAVEL`, nunca "sem efeito".** É o 4º
   veredito do Ledger, e ele existe porque 240 das 1.037 vermelhas do projeto
   eram "não consegui medir" lidas como "medi e reprovei".

O módulo **descreve**; ele não classifica hipótese nem promove nada. Quem
classifica é `radar.cerebro.mapa_validade`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd

from radar.cerebro.mapa_validade import MIN_DIAS_INDEPENDENTES, MIN_TRADES
from radar.evidence.ledger import t_newey_west

__all__ = [
    "BLOCOS_HORARIO",
    "EIXOS_DERIVADOS",
    "EIXOS_NATIVOS",
    "FRACAO_MINIMA_DE_DISPERSAO",
    "JANELA_ADTV",
    "JANELA_ATR",
    "MINIMO_VALORES_DIARIOS_DISTINTOS",
    "LeituraDeCelula",
    "Omnibus",
    "bloco_horario",
    "derivar_liquidez_e_volatilidade",
    "em_tercos",
    "ler_eixo",
    "omnibus_permutacao",
]

# ── Config CONGELADA pelo pré-registro §1 e §2 ─────────────────────────────

JANELA_ADTV: Final[int] = 21
"""Barras da média de volume financeiro. 21 = um mês de pregão."""

JANELA_ATR: Final[int] = 14
"""Barras do ATR normalizado. 14 é o valor clássico, e o ponto aqui é NÃO
escolher: qualquer ajuste fino da janela seria grau de liberdade sobre o
resultado."""

ROTULOS_TERCOS: Final[tuple[str, str, str]] = ("baixa", "media", "alta")

BLOCOS_HORARIO: Final[tuple[str, ...]] = ("00-05", "06-11", "12-17", "18-23")
"""Quatro blocos de 6h em UTC. Blocos, e não 24 horas soltas, porque 24 células
por eixo é multiplicidade de graça — e o pré-registro §2 fixou isso antes de
qualquer número."""

EIXOS_NATIVOS: Final[tuple[str, ...]] = (
    "ticker", "horizonte", "holding_barras", "mfe", "mae", "ano", "mercado",
)
"""Colunas que `oos_trades_*.parquet` já carrega — custo zero, sem join."""

EIXOS_DERIVADOS: Final[tuple[str, ...]] = ("liquidez", "volatilidade", "regime", "horario")
"""Precisam de join ou de fonte externa. `horario` só existe onde a barra é
intradiária: na B3 a entrada é sempre 00:00 e o eixo sai NÃO ESTIMÁVEL."""

LAG_NEWEY_WEST_PADRAO: Final[int] = 21

MINIMO_VALORES_DIARIOS_DISTINTOS: Final[int] = 10
"""Piso grosso: série com menos valores distintos que isto não tem o que medir."""

FRACAO_MINIMA_DE_DISPERSAO: Final[float] = 0.01
"""
A trava que importa, e ela é **relativa ao mercado** para não arbitrar magnitude.

Uma célula só é mensurável se o desvio diário dela for ao menos 1% do desvio
diário TÍPICO (mediana) das células daquele recorte. Achada medindo, em duas
formas da mesma doença:

* `swing_cripto_h10_B#topo || BTCUSDT` — desvio `4,3e-19`, `t` de `2,9e+16`. O
  braço B mede excesso sobre o benchmark e o benchmark do cripto **é o BTC**: o
  excesso do BTC sobre ele mesmo é exatamente zero, então o "retorno" era o
  custo, idêntico todo dia.
* **Stablecoins e tokens pareados** — `BFUSDUSDT` (`t` 137,8), `XUSDUSDT`
  (104,8), `USD1USDT`, `USDEUSDT`, `WBTCUSDT`. Desvio diário de `3e-4` a
  `1,6e-3` contra **`1,67e-1`** típico do mercado, e média exatamente igual ao
  custo de round-trip. Não medem alpha: medem *"o ativo não anda, logo perde-se
  a taxa, com altíssima consistência"*.

Sem esta trava o `max |t|` do omnibus é sempre a célula mais degenerada — e a
comparação contra o nulo vira ficção, porque a permutação embaralha os rótulos e
desfaz exatamente a degenerescência que produziu o número.

**Esta é uma trava de leitura, não a correção de fundo.** O certo é o universo de
cripto não oferecer instrumento pareado como candidato — ver o card
`CEREBRO_UNIVERSO_CRIPTO_PAREADOS01`. Mesma família do `beta_63 = 33.256` (LUNA
no colapso) que contaminou o v1 (ADR 0032 §6).
"""


@dataclass(frozen=True)
class LeituraDeCelula:
    """Uma célula da autópsia: um valor de um eixo, com o que se pôde medir."""

    eixo: str
    valor: str
    n_trades: int
    dias_independentes: int
    excesso_medio: float
    t_nw: float
    estimavel: bool
    motivo: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "eixo": self.eixo,
            "valor": self.valor,
            "n_trades": self.n_trades,
            "dias_independentes": self.dias_independentes,
            "excesso_medio": None if np.isnan(self.excesso_medio) else self.excesso_medio,
            "t_nw": None if np.isnan(self.t_nw) else self.t_nw,
            "estimavel": self.estimavel,
            "motivo": self.motivo,
        }


@dataclass(frozen=True)
class Omnibus:
    """
    O teste que pergunta *"este eixo carrega alguma informação?"* — uma vez, sem
    escolher célula nenhuma.

    Existe porque a alternativa (olhar as células e reportar a melhor) é o
    p-hacking na sua forma mais pura: com 2.410 células vazias, a maior |t| por
    acaso passa de 4. O omnibus gasta **1 teste** no denominador do FDR em vez
    de um por célula, e responde antes de qualquer seleção.

    É o Portão 3 do ADR 0032 §8 (rótulo permutado) aplicado a um eixo em vez de
    a um modelo.
    """

    eixo: str
    n_celulas: int
    max_t_real: float
    max_t_nulo_media: float
    max_t_nulo_mediana: float
    max_t_nulo_p95: float
    n_permutacoes: int
    p_valor: float
    """Fração das permutações cujo `max |t|` foi >= o real. Alto = o eixo não se
    distingue de ruído."""

    @property
    def distingue_de_ruido(self) -> bool:
        """`True` só quando o real supera o nulo com folga convencional."""
        return bool(self.p_valor < 0.05)

    def to_dict(self) -> dict[str, Any]:
        return {
            "eixo": self.eixo,
            "n_celulas": self.n_celulas,
            "max_t_real": self.max_t_real,
            "nulo": {
                "media": self.max_t_nulo_media,
                "mediana": self.max_t_nulo_mediana,
                "p95": self.max_t_nulo_p95,
                "n_permutacoes": self.n_permutacoes,
            },
            "p_valor": self.p_valor,
            "distingue_de_ruido": self.distingue_de_ruido,
        }


def derivar_liquidez_e_volatilidade(
    ohlcv: pd.DataFrame,
    *,
    col_high: str = "high",
    col_low: str = "low",
    col_close: str = "close",
    col_volume: str = "volume",
) -> pd.DataFrame:
    """
    `(ticker, date) -> adtv21, atr14`, com janela **estritamente passada**.

    O `.shift(1)` depois do `rolling` é o que separa autópsia de look-ahead: a
    liquidez que classifica um trade é a que se conhecia na véspera da entrada.
    Sem ele, a partição usaria a barra do próprio dia do trade — e a leitura
    mediria o futuro, não o contexto.

    `min_periods` é metade da janela para não descartar o começo da série de um
    ativo novo; abaixo disso sai `NaN`, que a leitura reporta como não medida.
    """
    faltando = [c for c in (col_high, col_low, col_close, col_volume) if c not in ohlcv.columns]
    if faltando:
        raise ValueError(f"derivar_liquidez_e_volatilidade: coluna(s) ausente(s): {faltando}")

    df = ohlcv.sort_values(["ticker", "date"]).copy()
    por_ticker = df.groupby("ticker", sort=False)

    df["adtv21"] = por_ticker[col_volume].transform(
        lambda s: s.rolling(JANELA_ADTV, min_periods=JANELA_ADTV // 2).mean().shift(1)
    )
    amplitude = (df[col_high] - df[col_low]) / df[col_close].replace(0.0, np.nan)
    df["atr14"] = amplitude.groupby(df["ticker"], sort=False).transform(
        lambda s: s.rolling(JANELA_ATR, min_periods=JANELA_ATR // 2).mean().shift(1)
    )
    return df[["ticker", "date", "adtv21", "atr14"]]


def em_tercos(serie: pd.Series, rotulos: tuple[str, str, str] = ROTULOS_TERCOS) -> pd.Series:
    """
    Terços do valor observado — o corte de menor resolução que ainda separa.

    Terço e não decil de propósito: quanto mais fino o corte, mais células e
    mais multiplicidade, e o pré-registro §2 fixou a resolução ANTES de olhar
    resultado. Série sem variação suficiente para 3 faixas devolve tudo `NaN`
    (vira NÃO ESTIMÁVEL na leitura), nunca uma faixa artificial.
    """
    limpa = pd.to_numeric(serie, errors="coerce")
    try:
        return pd.qcut(limpa, 3, labels=list(rotulos), duplicates="drop")
    except ValueError:
        return pd.Series(pd.NA, index=serie.index, dtype="object")


def bloco_horario(datas: pd.Series) -> pd.Series:
    """
    Hora de entrada em blocos de 6h UTC.

    Onde a barra é diária a hora é sempre `00:00`, e o resultado é um bloco
    único — **constante**. Constante não é eixo (G-P1), e é por isso que
    `ler_eixo` marca o caso como NÃO ESTIMÁVEL em vez de reportar "sem efeito":
    na B3 o fenômeno não foi medido, não foi refutado.
    """
    horas = pd.to_datetime(datas).dt.hour
    return pd.cut(
        horas, bins=[-1, 5, 11, 17, 23], labels=list(BLOCOS_HORARIO)
    ).astype("object")


def _t_e_dias(grupo: pd.DataFrame, *, lag: int, col_valor: str) -> tuple[float, int, float]:
    """`(t_NW, dias independentes, excesso médio)` de um recorte de trades.

    A série vai para média POR DIA antes do `t` — o clustering de trades do
    mesmo dia já inflou `t` de 5,08 para 0,30 neste projeto uma vez.

    Série degenerada (menos de `MINIMO_VALORES_DIARIOS_DISTINTOS` valores
    distintos) devolve `t = NaN`: é o caso do benchmark medido contra ele mesmo,
    e um `t` de `2,9e+16` ali não é edge, é divisão por zero."""
    por_dia = grupo.groupby("date")[col_valor].mean()
    if por_dia.nunique() < MINIMO_VALORES_DIARIOS_DISTINTOS:
        return float("nan"), int(por_dia.size), float(grupo[col_valor].mean())
    return (
        float(t_newey_west(por_dia, lag=lag)),
        int(por_dia.size),
        float(grupo[col_valor].mean()),
    )


def ler_eixo(
    trades: pd.DataFrame,
    eixo: str,
    *,
    col_valor: str = "liquido",
    lag: int = LAG_NEWEY_WEST_PADRAO,
    min_trades: int = MIN_TRADES,
    min_dias: int = MIN_DIAS_INDEPENDENTES,
) -> list[LeituraDeCelula]:
    """
    Uma leitura por valor do eixo, com o piso de potência aplicado ANTES do
    número.

    Célula abaixo de `min_trades` ou `min_dias` sai `estimavel=False` **e o `t`
    dela não é reportado como resultado** — é a diferença entre "medi e não
    achei" e "não consegui medir", que é a razão de o 4º veredito existir.

    Eixo constante (um valor só) também sai não estimável: partir por coluna
    constante é o G-P1 do Contrato de Pensamento, e a leitura seria sobre nada.
    """
    if eixo not in trades.columns:
        return [LeituraDeCelula(eixo, "—", 0, 0, float("nan"), float("nan"),
                                False, f"eixo ausente nos trades: {eixo}")]

    presentes = trades[trades[eixo].notna()]
    valores = presentes[eixo].astype(str).unique()
    if len(valores) <= 1:
        unico = str(valores[0]) if len(valores) else "—"
        return [LeituraDeCelula(
            eixo, unico, len(presentes), 0, float("nan"), float("nan"), False,
            "eixo CONSTANTE neste recorte — partir por ele violaria G-P1; "
            "o fenômeno não foi medido, não foi refutado",
        )]

    leituras: list[LeituraDeCelula] = []
    for valor, grupo in presentes.groupby(presentes[eixo].astype(str), sort=True):
        n = len(grupo)
        t, dias, media = _t_e_dias(grupo, lag=lag, col_valor=col_valor)
        if n < min_trades or dias < min_dias:
            leituras.append(LeituraDeCelula(
                eixo, str(valor), n, dias, media, float("nan"), False,
                f"abaixo do piso (n={n}<{min_trades} ou dias={dias}<{min_dias})",
            ))
            continue
        if np.isnan(t):
            leituras.append(LeituraDeCelula(
                eixo, str(valor), n, dias, media, float("nan"), False,
                "série diária DEGENERADA (quase sem dispersão) — o `t` aqui seria "
                "artefato de divisão por zero, não efeito",
            ))
            continue
        leituras.append(LeituraDeCelula(eixo, str(valor), n, dias, media, t, True))
    return leituras


def omnibus_permutacao(
    trades: pd.DataFrame,
    eixo: str,
    *,
    col_valor: str = "liquido",
    col_estrategia: str = "evidencia",
    lag: int = LAG_NEWEY_WEST_PADRAO,
    n_permutacoes: int = 50,
    seed: int = 20260805,
    min_trades: int = MIN_TRADES,
    min_dias: int = MIN_DIAS_INDEPENDENTES,
) -> Omnibus:
    """
    O eixo carrega informação, ou o `max |t|` dele é o que o acaso produz?

    O nulo embaralha o rótulo do eixo **dentro de cada estratégia**, o que
    destrói a associação eixo↔resultado preservando o número de células, o
    tamanho de cada uma e a distribuição de retorno da estratégia. Só uma coisa
    muda: se o rótulo significa algo.

    O `p_valor` é a fração das permutações cujo `max |t|` alcançou o real. Alto
    = **o real é indistinguível do ruído** — e isso é um resultado publicável,
    não uma falha da medição.

    Custa **1** no denominador do FDR, não uma entrada por célula: nenhuma
    célula é selecionada aqui.
    """
    if eixo not in trades.columns:
        raise ValueError(f"omnibus_permutacao: eixo ausente nos trades: {eixo}")

    base = trades[trades[eixo].notna()].copy()
    base["_celula"] = base[col_estrategia].astype(str) + "||" + base[eixo].astype(str)

    # Piso de dispersão calibrado NO PRÓPRIO recorte: 1% do desvio diário
    # mediano. Fixo antes de qualquer permutação, e o mesmo para real e nulo —
    # calibrá-lo dentro do laço deixaria o nulo com régua diferente da do real.
    dispersoes = [
        float(g.groupby("date")[col_valor].mean().std())
        for _, g in base.groupby("_celula", sort=False)
        if len(g) >= min_trades
    ]
    validas = [d for d in dispersoes if d == d and d > 0]
    piso_dispersao = (
        FRACAO_MINIMA_DE_DISPERSAO * float(np.median(validas)) if validas else 0.0
    )

    def max_t(df: pd.DataFrame) -> tuple[float, int]:
        ts: list[float] = []
        for _, grupo in df.groupby("_celula", sort=False):
            if len(grupo) < min_trades:
                continue
            por_dia = grupo.groupby("date")[col_valor].mean()
            if float(por_dia.std()) < piso_dispersao:
                continue  # pareado/degenerado — mede a taxa, não o mercado
            t, dias, _ = _t_e_dias(grupo, lag=lag, col_valor=col_valor)
            if dias >= min_dias and not np.isnan(t):
                ts.append(abs(t))
        return (max(ts) if ts else float("nan")), len(ts)

    real, n_celulas = max_t(base)

    rng = np.random.default_rng(seed)
    nulos: list[float] = []
    for _ in range(n_permutacoes):
        embaralhado = base.copy()
        # permuta DENTRO da estratégia: o que morre é só a identidade do eixo
        embaralhado[eixo] = embaralhado.groupby(col_estrategia, sort=False)[eixo].transform(
            lambda s: pd.Series(rng.permutation(s.to_numpy()), index=s.index)
        )
        embaralhado["_celula"] = (
            embaralhado[col_estrategia].astype(str) + "||" + embaralhado[eixo].astype(str)
        )
        t_nulo, _ = max_t(embaralhado)
        if not np.isnan(t_nulo):
            nulos.append(t_nulo)

    arr = np.asarray(nulos, dtype=float)
    p = float(np.mean(arr >= real)) if arr.size and not np.isnan(real) else float("nan")
    return Omnibus(
        eixo=eixo,
        n_celulas=n_celulas,
        max_t_real=real,
        max_t_nulo_media=float(arr.mean()) if arr.size else float("nan"),
        max_t_nulo_mediana=float(np.median(arr)) if arr.size else float("nan"),
        max_t_nulo_p95=float(np.percentile(arr, 95)) if arr.size else float("nan"),
        n_permutacoes=int(arr.size),
        p_valor=p,
    )

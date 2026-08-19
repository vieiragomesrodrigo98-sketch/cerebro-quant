"""
Ledger de Evidências — a tabela que decide os PESOS do motor. Medida, não escrita.

O problema que isto resolve
--------------------------
Hoje o `entry_engine` usa pesos FIXOS (`0.35×score_historico + 0.20×score_tecnico`...).
Esses números não foram medidos: foram **decididos**. E a S143 provou o custo disso —
o maior peso do motor (`score_historico`, 0,35) é **ruído**: o `t=5,08` que o
autorizava estava inflado por contagem de amostra errada (FIN-006); o honesto é ~0,7.

    O software sempre foi sólido. Os botões é que foram girados no chute.

A tese do ADR 0020 é que **o peso da evidência é função do CONTEXTO**, não uma
constante. Este módulo é a Trava 1 desse ADR: nenhum peso entra em produção sem uma
linha 🟢 aqui — e a linha só fica verde depois de sobreviver a todas as facas.

As facas (uma célula só fica 🟢 se passar em TODAS)
---------------------------------------------------
1. **`t` Newey-West** (>= 2,0). Corrige o cluster do mesmo dia (retornos do mesmo
   pregão não são independentes) E a janela sobreposta (dois eventos próximos com
   H=10 compartilham 7 dias de retorno). O `t` ingênuo já mentiu uma vez.
2. **Excesso sobre o mercado.** Bruto mede beta, e beta se compra com BOVA11.
3. **Paga o custo.** Slippage de equilíbrio > 0 depois da corretagem.
4. **Persistente** (>= 70% dos anos positivos). Matou o "fade" (14/21) e o drift de
   10 dias (9/18).
5. **Operável.** O efeito tem que existir onde dá para OPERAR. Uma célula que só
   aparece na ponta ilíquida não vira peso: o spread come o resultado antes de você.
   Foi assim que o dividendo/JCP morreu (+0,28% no ilíquido, **−0,03%** no líquido).
   Quem monta a grade declara o que é operável (`operavel=False`) — a definição de
   liquidez muda de mercado para mercado, e este módulo é neutro de mercado.
6. **Amostra real.** >= 15 dias distintos, senão é ficção.
7. **Sobrevive à própria grade** (`aplicar_fdr`, e é a mais importante). Uma grade de
   220 células testadas a `t≥2` produz **~11 células significativas SÓ POR ACASO** —
   se você olhar 220 moedas, algumas vão dar 10 caras seguidas. Selecionar as verdes
   e chamar de descoberta é data mining com passos extras. A faca 7 controla o FDR
   (Benjamini-Hochberg) sobre a **grade inteira**, e por isso não pode morar dentro de
   `medir()`: é uma propriedade do conjunto, não da célula.

    E mesmo a faca 7 não basta — ela só limpa o teste IN-SAMPLE. A prova de vida é
    walk-forward: escolher a célula com dado ANTERIOR ao ano e medir o que ela rendeu
    NAQUELE ano. É o que `scripts/build_evidence_ledger.py` faz na seção OOS.

Neutro em mercado
-----------------
O schema carrega `mercado` ("b3", "cripto", ...) desde o começo. Cripto muda a fonte
do evento, o custo e o calendário — **não muda o método**. Ficar pronto para cripto
é ter esta medição sem premissa de B3 embutida, não é ter um `if mercado == "cripto"`
espalhado depois.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

# --- as facas, como constantes auditáveis ------------------------------------
T_MINIMO = 2.0            # Newey-West, não o t ingênuo
PERSISTENCIA_MINIMA = 0.70
MIN_DIAS_DISTINTOS = 15
MIN_EVENTOS = 50
FDR_Q = 0.10              # taxa de falsa descoberta tolerada na grade

NAO_ESTIMAVEL = "nao_estimavel"
"""
O QUARTO estado, e a razão de ele existir
------------------------------------------
Até aqui o Ledger tinha três vereditos, e **"não consegui medir" caía em
`vermelho`**. São afirmações diferentes e a diferença é grande:

* `vermelho` = *medi e o efeito não passou nas facas*. É **refutação** — a
  hipótese foi testada e perdeu.
* `nao_estimavel` = *a estatística não existe para esta amostra*. Nenhum teste
  aconteceu. Não é evidência contra nada.

Colapsar os dois faz o mapa de validade mentir de duas formas, ambas na mesma
direção — pessimista e falsamente conclusiva: (a) uma hipótese morre por falta
de amostra parecendo ter sido derrubada pelo mercado; (b) o relatório anuncia
"N células reprovaram" quando parte delas nunca foi testada. Medido no Ledger
Técnico em 2026-08-04: **240 das 1.037 vermelhas (23%) eram não-estimáveis** —
quase um quarto do "fracasso" era ausência de medição.

Regra mecânica, e ela coincide exatamente com o que o FDR já ignorava:

    nao_estimavel  ⟺  p_valor é NaN
                   ⟺  N < MIN_EVENTOS  ou  dias < MIN_DIAS_DISTINTOS  ou  t indefinido

`aplicar_fdr` **já** excluía essas células do denominador (filtra `p_valor`
NaN), então o controle de multiplicidade nunca esteve contaminado — o defeito
era de rótulo e de leitura, não de aritmética. O invariante acima é testado
(`test_nao_estimavel_equivale_a_p_nan`) justamente para que as duas definições
não possam divergir no futuro.
"""

VEREDITOS = ("verde", "amarelo", "vermelho", NAO_ESTIMAVEL)
"""Os quatro estados possíveis. Só `verde` gera peso (ver `Celula.peso`)."""


@dataclass(frozen=True)
class Celula:
    """Uma linha do Ledger: evidência × horizonte × contexto, com o veredito."""
    mercado: str
    evidencia: str
    horizonte: int
    contexto: str
    n: int
    dias: int
    excesso: float           # retorno médio em excesso sobre o mercado (fração)
    t_nw: float              # t Newey-West (lag = horizonte)
    p_valor: float           # bilateral, a partir do t_nw
    persistencia: float      # fração de anos com média positiva
    equilibrio: float        # slippage/lado que zera o resultado
    veredito: str            # "verde" | "amarelo" | "vermelho"
    motivo: str

    @property
    def peso(self) -> float:
        """
        O peso que o motor deve dar a esta evidência NESTE contexto.

        Só evidência 🟢 pesa. E o peso é proporcional ao `t` honesto — não a uma
        opinião de design. Célula amarela/vermelha pesa ZERO: **dar zero é uma
        decisão, não uma omissão.**
        """
        if self.veredito != "verde":
            return 0.0
        return float(max(0.0, self.t_nw))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["peso"] = self.peso
        return d


def t_newey_west(serie_diaria: pd.Series, lag: int) -> float:
    """
    `t` da média diária contra zero, robusto a autocorrelação até `lag` dias.

    Corrige as DUAS inflações: cluster do mesmo dia (a série já é média por dia) e
    janela sobreposta (o kernel de Bartlett desconta a autocorrelação induzida por
    eventos próximos compartilhando retorno).
    """
    x = serie_diaria.dropna().to_numpy()
    n = len(x)
    if n < MIN_DIAS_DISTINTOS:
        return float("nan")
    u = x - x.mean()
    var = float(u @ u) / n
    for k in range(1, min(lag, n - 1) + 1):
        gk = float(u[k:] @ u[:-k]) / n
        var += 2.0 * (1.0 - k / (lag + 1.0)) * gk
    if var <= 0:
        return float("nan")
    return float(x.mean() / np.sqrt(var / n))


def p_bilateral(t: float) -> float:
    """p bilateral do `t` sob aproximação normal. Sem dependência de scipy."""
    if t is None or math.isnan(t):
        return float("nan")
    return float(math.erfc(abs(t) / math.sqrt(2.0)))


def persistencia_anual(df: pd.DataFrame, col: str, min_por_ano: int = 8) -> float:
    """Fração de anos com média positiva. Um efeito real não vive de dois anos bons."""
    d = df[df[col].notna()].copy()
    if d.empty:
        return 0.0
    d["_ano"] = pd.to_datetime(d["date"]).dt.year
    anos = [g[col].mean() for _, g in d.groupby("_ano") if len(g) >= min_por_ano]
    if not anos:
        return 0.0
    return float(sum(1 for m in anos if m > 0) / len(anos))


def medir(
    df: pd.DataFrame,
    col_retorno: str,
    *,
    mercado: str,
    evidencia: str,
    horizonte: int,
    contexto: str,
    custo_lado: float,
    operavel: bool = True,
) -> Celula:
    """
    Mede UMA célula e aplica as facas 1-6. A faca 7 (FDR) é da grade — ver
    `aplicar_fdr`, que precisa ver todas as células juntas.

    `df` precisa ter `date` e a coluna de retorno em EXCESSO já calculada — a
    neutralização de mercado é responsabilidade de quem monta o dataset, porque
    depende do universo (na B3 é a média transversal do dia; em cripto pode ser
    o BTC).

    `operavel=False` marca um contexto onde o resultado não é capturável (a ponta
    ilíquida da B3, por exemplo). A célula ainda é MEDIDA e publicada — saber que o
    efeito só existe onde não dá para operar é o que desmascara o artefato — mas
    nunca vira peso.
    """
    d = df[df[col_retorno].notna()]
    x = d[col_retorno].to_numpy()
    n = len(x)

    if n < MIN_EVENTOS:
        return Celula(mercado, evidencia, horizonte, contexto, n, 0, float("nan"),
                      float("nan"), float("nan"), 0.0, float("nan"), NAO_ESTIMAVEL,
                      f"amostra insuficiente (N={n} < {MIN_EVENTOS}) — nenhum teste "
                      "aconteceu; isto NÃO é evidência contra a hipótese")

    por_dia = pd.Series(x).groupby(d["date"].to_numpy()).mean()
    dias = len(por_dia)
    excesso = float(x.mean())
    t = t_newey_west(por_dia, lag=max(1, horizonte))
    p = p_bilateral(t)
    persist = persistencia_anual(d, col_retorno)
    equilibrio = (excesso - 2 * custo_lado) / 2.0

    # NÃO ESTIMÁVEL antes de qualquer faca: sem `t`, nenhuma faca foi aplicada,
    # e chamar isso de reprovação seria inventar um teste que não aconteceu.
    # Cobre os dois modos: poucos dias distintos (o `t` de amostra com
    # clustering é ficção — S142) e variância nula (`t_newey_west` → NaN).
    if dias < MIN_DIAS_DISTINTOS:
        return Celula(mercado, evidencia, horizonte, contexto, n, dias, excesso, t, p,
                      persist, equilibrio, NAO_ESTIMAVEL,
                      f"só {dias} dias distintos (< {MIN_DIAS_DISTINTOS}) — o `t` de "
                      "uma amostra assim é ficção; nada foi testado")
    if np.isnan(t):
        return Celula(mercado, evidencia, horizonte, contexto, n, dias, excesso, t, p,
                      persist, equilibrio, NAO_ESTIMAVEL,
                      "`t` indefinido (variância nula) — a estatística não existe "
                      "para esta amostra")

    falhas = []
    if t < T_MINIMO:
        falhas.append(f"t_NW={t:.2f} < {T_MINIMO}")
    if equilibrio <= 0:
        falhas.append("não paga o custo")
    if persist < PERSISTENCIA_MINIMA:
        falhas.append(f"persistência {persist:.0%} < {PERSISTENCIA_MINIMA:.0%}")
    if not operavel:
        falhas.append("só existe onde não dá para operar")

    if not falhas:
        veredito, motivo = "verde", "passou nas facas 1-6"
    elif len(falhas) == 1 and equilibrio > 0:
        veredito, motivo = "amarelo", "; ".join(falhas)
    else:
        veredito, motivo = "vermelho", "; ".join(falhas)

    return Celula(mercado, evidencia, horizonte, contexto, n, dias, excesso, t, p,
                  persist, equilibrio, veredito, motivo)


def aplicar_fdr(celulas: list[Celula], q: float = FDR_Q) -> list[Celula]:
    """
    Faca 7 — a que separa descoberta de garimpo.

    Uma grade grande produz "significância" de graça: a 220 células, o acaso entrega
    ~11 com `|t|>2`. Benjamini-Hochberg controla a fração ESPERADA de falsas entre as
    aprovadas: ordena os p, e rejeita o nulo até o maior `i` com `p_(i) <= q·i/K`.

    A família de testes é a grade INTEIRA (todas as células medidas nesta rodada) —
    não só as candidatas a verde. Restringir a família às que já passaram nas outras
    facas seria escolher o denominador depois de ver o resultado: o mesmo truque, uma
    camada acima.

    Célula que era verde e não sobrevive vira **amarela** — o efeito pode ser real,
    mas a grade não consegue distingui-lo do acaso, e peso a gente só dá para o que
    foi provado.

    **Célula `nao_estimavel` fica fora do denominador**, e isso já era verdade antes
    de o quarto veredito existir: o filtro `not math.isnan(c.p_valor)` abaixo remove
    exatamente o mesmo conjunto (ver o invariante documentado em `NAO_ESTIMAVEL`).
    É o comportamento correto — uma célula que não produziu estatística não é um
    teste, e inflar `k` com não-testes tornaria o corte de BH artificialmente
    severo, matando descobertas reais para "pagar" por medições que nunca houve.
    """
    ps = sorted(
        [(c.p_valor, i) for i, c in enumerate(celulas) if not math.isnan(c.p_valor)]
    )
    k = len(ps)
    if k == 0:
        return list(celulas)

    corte = 0.0
    for posicao, (p, _) in enumerate(ps, start=1):
        if p <= q * posicao / k:
            corte = p

    saida = []
    for c in celulas:
        if c.veredito == "verde" and not (c.p_valor <= corte):
            saida.append(replace(
                c, veredito="amarelo",
                motivo=(f"não sobrevive à grade: p={c.p_valor:.4f} > corte FDR "
                        f"{corte:.4f} (q={q:.0%}, {k} células testadas)")))
        else:
            saida.append(c)
    return saida

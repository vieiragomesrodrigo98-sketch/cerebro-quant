"""
src/radar/mie/dataset.py — MIE-3 (ADR 0028/0030): montagem de X/y por célula
e gerador de splits walk-forward, AO PÉ DA LETRA do pré-registro CONGELADO
`docs/estudos/PRE_REGISTRO_MIE3_B3_R1.md` (commit `b99e76da`). Núcleo 100%
PURO — nenhum `pd.read_parquet`, nenhum `from config import settings`,
nenhuma chamada de rede: todo I/O (ler o feature store, ler o parquet de
preços para `detectar_regime`) é responsabilidade de `scripts/mie_training/
train.py` (mesmo par núcleo/script já usado por `radar.evidence.ledger` +
`scripts/build_evidence_ledger.py`).

As 3 células da família `mie_b3_v1` (pré-registro §2)
------------------------------------------------------
`y = 1[exc_fwd_h > limiar_h]` — `CELULAS_MIE_B3_V1` é a única fonte da
verdade dos 3 pares (horizonte, limiar); mudar um valor aqui é família NOVA
(v2), nunca emenda silenciosa (cabeçalho do pré-registro).

`X` — 43 features + 2 categóricas (pré-registro §1)
-----------------------------------------------------
`COLUNAS_X` = `radar.features.colunas.COLUNAS_FEATURES` (as 43 colunas
causais do MIE-1, reusadas — NUNCA reimplementadas) + `regime`
(broadcast do feature store, janela de calendário fixa) + `regime_mie`
(MIE-2, `radar.mie.regime.detectar_regime`, rule-based diário — calculado
pelo CHAMADOR sobre `data/historical/phase0/prices_full.parquet` e juntado
aqui por `juntar_regime_mie`, uma função pura de broadcast por DATA, mesmo
padrão de `radar.learning.feature_store._aplicar_regime`). Nenhuma coluna
`ret_fwd_*`/`exc_fwd_*` (o TARGET) entra em `COLUNAS_X` — travado por
`TestAntiLookahead` sobre a lista REAL (não uma cópia hardcoded).

Walk-forward com embargo + holdout de calibração (pré-registro §4)
----------------------------------------------------------------------
Expanding window: o 1º treino usa TODO o histórico disponível até
`ano_primeiro_teste - 1` (2005 → 2012-12-31 no dado real de 2005-2026); cada
ano `Y >= ano_primeiro_teste` é testado com o "treino bruto" = todo o
histórico anterior a `Y`, MENOS os últimos `embargo_pregoes` pregões
(default 21 — o maior horizonte da família) imediatamente antes de `Y`
(embargo no FIM da janela de treino, nunca um deslocamento do início do
teste — mesma convenção de `scripts/build_technical_ledger.py::
treino_com_embargo`, aqui aplicada com o calendário REAL de pregões em vez
da aproximação dias-úteis->dias-corridos, porque o dado aqui já É diário
com datas de calendário reais).

Dentro do treino bruto, o holdout interno de calibração é os últimos
`meses_holdout` meses (default 12) — excluído do FIT do GBM, protegido por
um SEGUNDO embargo de `embargo_pregoes` pregões contra o resto do treino
(o "fit"). Os `embargo_pregoes` pregões entre o fim do fit e o início do
holdout, e entre o fim do treino bruto e o início do teste, são
DESCARTADOS por inteiro (nem fit, nem holdout, nem teste) — são a folga de
segurança, não pertencem a nenhum dos três.

`gerar_split_final` é a MESMA máquina de embargo, sem teste OOS: o modelo de
PRODUÇÃO (pré-registro §7) usa TODO o histórico como treino bruto (o "hoje"
é a última data disponível), com fit/holdout tirados dele exatamente como em
cada ano do walk-forward.

Extensão CRIPTO (`docs/estudos/PRE_REGISTRO_MIE_CRIPTO_1D_R1.md`) — parametrização MÍNIMA, comportamento B3 intocado
----------------------------------------------------------------------------------------------------------------------
Este módulo ganhou 2 pontos de parametrização para servir a família
`mie_cripto_1d_v1` (`CELULAS_MIE_CRIPTO_1D_V1`) SEM tocar uma linha do
comportamento B3 (todo default continua bit a bit igual ao de antes):

  1. `montar_dataset_celula(..., colunas_categoricas=...)` — o store cripto
     (`radar.learning.feature_store_cripto`) já chega com `regime` PRONTO
     (rule-based `radar.historical.regime_cripto`, calculado por quem monta o
     store) — não há um segundo "regime_mie" MIE-2 para juntar (o cripto não
     tem `juntar_regime_mie`; `scripts/mie_training/train.py --mercado
     cripto` NUNCA chama essa função). O pré-registro cripto usa `regime` +
     `faixa_liquidez` (`COLUNAS_CATEGORICAS_CRIPTO`) como as 2 categóricas,
     no lugar de `regime`/`regime_mie` do B3.
  2. `unidade_embargo="dias_corridos"` em `gerar_splits_walk_forward`/
     `gerar_split_final` — o calendário cripto é 24/7 (sem "pregão"), então
     o embargo (pré-registro §4: 21 dias) é medido em DIAS DE RELÓGIO
     (`_data_maxima_com_embargo_dias_corridos`), não em posições de datas
     distintas do calendário (`_data_maxima_com_embargo`, a versão B3/
     pregões, default inalterado). As duas coincidem quando o calendário não
     tem gap; divergem sob um gap real — ver docstring de
     `_data_maxima_com_embargo_dias_corridos`.

`ano_primeiro_teste=2021`/`embargo_pregoes=21` (via `ANO_PRIMEIRO_TESTE_
CRIPTO_PADRAO`/`EMBARGO_DIAS_CORRIDOS_CRIPTO_PADRAO`) e `meses_holdout=12`
(reusa `MESES_HOLDOUT_PADRAO`, o pré-registro cripto pede o MESMO valor do
B3) são só ARGUMENTOS passados pelo chamador (`scripts/mie_training/
train.py`) — nenhuma outra mudança de mecânica foi necessária.

Extensão CRIPTO 1h (`docs/estudos/PRE_REGISTRO_MIE_CRIPTO_1H_R1.md`) — embargo/holdout em BARRAS, não em calendário
----------------------------------------------------------------------------------------------------------------------
A família `mie_cripto_1h_v1` (`CELULAS_MIE_CRIPTO_1H_V1`) reusa TUDO da
extensão cripto acima (`COLUNAS_CATEGORICAS_CRIPTO_1H`/`COLUNAS_X_CRIPTO_1H`
são o MESMO objeto de `COLUNAS_CATEGORICAS_CRIPTO`/`COLUNAS_X_CRIPTO` — o
store 1h também chega com `regime`/`faixa_liquidez` prontos, sem
`regime_mie`) e precisa de 3 pontos de parametrização NOVOS, nenhum deles
tocando o comportamento B3/cripto-1d (todo default continua idêntico):

  1. `unidade_embargo="barras"` — o pré-registro 1h mede o embargo em 168
     BARRAS-HORA (a grade horária, não dias corridos nem pregões). Como
     `_data_maxima_com_embargo` (a variante "pregoes") já conta POSIÇÕES de
     valores distintos no calendário — nunca dias de calendário — ela É,
     por construção, uma contagem de BARRAS sempre que o `calendario` for
     feito de timestamps de barra (em vez de datas normalizadas para o
     dia). `"barras"` é um ALIAS dela mesma (mesma função, nome exposto
     documentando a intenção) — nenhuma lógica nova.
  2. `holdout_barras` (novo parâmetro de `gerar_splits_walk_forward`/
     `gerar_split_final`, default `None`) — o holdout do pré-registro 1h é
     "últimas ~2.160 barras (~90 dias)", uma contagem de BARRAS, não um
     número exato de meses (`pd.DateOffset(months=...)` não representa
     2.160 horas). Quando informado, a fronteira do holdout é calculada por
     POSIÇÃO no calendário (mesmo raciocínio do embargo por posição) em vez
     de por `DateOffset` — ver `_fronteira_holdout`. `None` (default)
     preserva EXATAMENTE o caminho `meses_holdout`/`DateOffset` de antes
     desta parametrização, byte a byte.
  3. `montar_dataset_celula(..., normalizar_data=False)` — o B3/cripto-1d
     sempre normalizam `meta["date"]` para o DIA (`.dt.normalize()`), pois
     uma linha por (ticker, dia) já é a granularidade nativa. O store 1h
     tem uma linha por (symbol, BARRA-HORA) — normalizar colapsaria todas
     as barras do mesmo dia no mesmo timestamp, quebrando o contrato "1
     trade por (ticker, barra)" de `radar.mie.metrics.construir_trades` e a
     contagem por posição do embargo/holdout em barras. `normalizar_data`
     default `True` preserva o comportamento de antes bit a bit; o
     chamador cripto 1h passa `False`.

`permutar_y_por_dia` ganhou um 4º parâmetro opcional, `chave_dia` — a prova
de fogo (a) do pré-registro (qualquer timeframe) é sempre "permutar POR
DIA-CALENDÁRIO", nunca por barra. Com `normalizar_data=False` (1h),
`meta["date"]` fica em resolução de hora — sem `chave_dia`, a permutação
aconteceria por BARRA (grupo errado). O chamador 1h passa
`chave_dia=meta["date"].dt.normalize()`; default `None` usa `meta["date"]`
como sempre (comportamento B3/cripto-1d inalterado, já que ali `date` já é
o próprio dia).

`amostrar_uma_barra_por_ticker_dia` (função NOVA, achado MIE_CRIPTO_1H01 —
prova de fogo R1) — B3/cripto-1d têm 1 linha por (ticker, dia); o cripto 1h
tem até 24 (uma por barra-hora). Para `h>=24` barras (toda a família
`mie_cripto_1h_v1`), as até-24 linhas do MESMO (ticker, dia) têm alvo quase
idêntico (janelas de excesso 97%+ sobrepostas) — não são observações
independentes. `permutar_y_por_dia` preserva a prevalência EXATA de cada
dia; combinada com colunas BROADCAST (`regime`/`breadth_sma50`/
`ibov_dist_ema50`, iguais para todo ticker numa mesma hora, quase
constantes dentro do dia), a prova de fogo (a) do cripto 1h conseguia achar
uma correlação dia/regime→prevalência real e causal, mas NÃO discriminante
por instrumento/barra — reforçada por até 24 cópias do mesmo evento,
aprendida com baixa variância pelo walk-forward re-treinado mesmo com `y`
lixo (`scripts/mie_training/evaluate.py::provas_de_fogo_celula` chama esta
função, só quando `reduzir_multiplicidade_intradia=True`, ANTES de
`permutar_y_por_dia`/`rodar_walk_forward_celula` — reduz `X`/`y`/`meta` a 1
linha por (ticker, dia), a MESMA granularidade em que a prova já era válida
no B3/cripto-1d; o resto da avaliação real — `medir_celula_em_gate`, prova
(b) — usa a resolução cheia, sem mudança). Ver docstring da função para o
mecanismo completo e `tests/unit/test_mie_dataset.py::
TestAmostrarUmaBarraPorTickerDia`/`TestEmbargoProtegeAlvoGappy` para a
fixture que reproduz o achado.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from radar.features.colunas import (  # noqa: F401  (reexport — ver comentário abaixo)
    COLUNAS_CALENDARIO_B3,
    COLUNAS_CATEGORICAS,
    COLUNAS_CATEGORICAS_CRIPTO,
    COLUNAS_CATEGORICAS_CRIPTO_1H,
    COLUNAS_CONTEXTO_DIA,
    COLUNAS_FEATURES,
    COLUNAS_FLUXO_CRIPTO,
    COLUNAS_SELECAO_B3,
    COLUNAS_SELECAO_CRIPTO,
    COLUNAS_SELECAO_CRIPTO_NATIVO,
    COLUNAS_X,
    COLUNAS_X_CRIPTO,
    COLUNAS_X_CRIPTO_1H,
)

# ── As 3 células da família `mie_b3_v1` (pré-registro §2) ────────────────────


@dataclass(frozen=True)
class CelulaMIE:
    """Uma célula do MIE-3: nome canônico + (horizonte, limiar) do alvo binário."""

    nome: str
    horizonte: int
    limiar: float


CELULAS_MIE_B3_V1: tuple[CelulaMIE, ...] = (
    CelulaMIE("mie_b3_h5", 5, 0.02),
    CelulaMIE("mie_b3_h10", 10, 0.03),
    CelulaMIE("mie_b3_h21", 21, 0.04),
)
"""Família FDR `mie_b3_v1` (pré-registro §2/§6) — EXATAMENTE estes 3 pares
(horizonte, limiar), nesta ordem. Mudar qualquer valor é família v2."""

CELULAS_POR_HORIZONTE: dict[int, CelulaMIE] = {c.horizonte: c for c in CELULAS_MIE_B3_V1}

# ── As 3 células da família `mie_cripto_1d_v1` (pré-registro CRIPTO §2) ──────

CELULAS_MIE_CRIPTO_1D_V1: tuple[CelulaMIE, ...] = (
    CelulaMIE("mie_cripto_h5", 5, 0.04),
    CelulaMIE("mie_cripto_h10", 10, 0.06),
    CelulaMIE("mie_cripto_h21", 21, 0.08),
)
"""Família FDR `mie_cripto_1d_v1` (`docs/estudos/PRE_REGISTRO_MIE_CRIPTO_1D_
R1.md` §2) — EXATAMENTE estes 3 pares (horizonte em DIAS CORRIDOS, limiar),
nesta ordem. Mesmos 3 horizontes do B3 (5/10/21), limiares escalados (4/6/8%
vs. 2/3/4% do B3 — vol diária cripto ≈2-3× a da B3, racional registrado ANTES
de olhar o dado, pré-registro §2). Mudar qualquer valor é família v2."""

CELULAS_CRIPTO_POR_HORIZONTE: dict[int, CelulaMIE] = {
    c.horizonte: c for c in CELULAS_MIE_CRIPTO_1D_V1
}

# ── As 3 células da família `mie_cripto_1h_v1` (pré-registro CRIPTO 1h §2) ───

CELULAS_MIE_CRIPTO_1H_V1: tuple[CelulaMIE, ...] = (
    CelulaMIE("mie_cripto_1h_h24", 24, 0.02),
    CelulaMIE("mie_cripto_1h_h72", 72, 0.03),
    CelulaMIE("mie_cripto_1h_h168", 168, 0.05),
)
"""Família FDR `mie_cripto_1h_v1` (`docs/estudos/PRE_REGISTRO_MIE_CRIPTO_1H_
R1.md` §2) — EXATAMENTE estes 3 pares (horizonte em BARRAS-HORA, limiar),
nesta ordem. Racional pré-declarado no pré-registro: mesma regra ~0,5σ da
família 1d, escalada por √t a partir dos limiares 1d (5d→4%: 1d≈1,8%→2,0%;
3d≈3,1%→3,0%; 7d≈4,7%→5,0%). Mudar qualquer valor é família v2."""

CELULAS_CRIPTO_1H_POR_HORIZONTE: dict[int, CelulaMIE] = {
    c.horizonte: c for c in CELULAS_MIE_CRIPTO_1H_V1
}

PREFIXO_TARGET_EXC = "exc_fwd_"
"""Mesmo prefixo de `radar.learning.feature_store.PREFIXO_TARGET_EXC` —
reimplementado como constante de UMA linha para este módulo não precisar
importar o prefixo de RET (`PREFIXO_TARGET_RET`), que não é usado aqui (o
alvo do MIE-3 é sempre o EXCESSO, nunca o retorno bruto)."""

# ── X e espaços de seleção: mudaram para `radar.features.colunas` (S169) ────
#
# Estavam aqui, e por isso `radar.cerebro.catalogo` importava `radar.mie` para
# declarar uma família — invertendo a dependência que o ADR 0032 fixa. Agora
# vivem numa folha sem dependências (importada no topo deste módulo); os nomes
# seguem reexportados daqui, então nenhum dos chamadores mudou.


def permutar_y_por_dia(
    y: pd.Series, meta: pd.DataFrame, *, seed: int, chave_dia: pd.Series | None = None
) -> pd.Series:
    """
    Prova de fogo (a) do pré-registro §6 ("labels permutados por dia"):
    embaralha `y` DENTRO de cada dia com semente fixa — quebra a relação
    genuína entre feature e rótulo em CADA linha (o modelo treinado sobre
    este `y` permutado não pode aprender nada real), preservando a
    prevalência de cada dia (quantos "1" o dia tinha continua o mesmo, só
    troca de QUAL ticker). Usada por `scripts/mie_training/evaluate.py`
    para re-rodar o walk-forward inteiro (`radar.mie.pipeline.
    rodar_walk_forward_celula`) com rótulo embaralhado, medindo o excesso
    REAL (`exc_fwd`, nunca permutado) dos trades que esse treino
    "contaminado" selecionaria — um pipeline sem vazamento produz `|t|<2`
    aqui (pré-registro §6).

    `chave_dia` (default `None` usa `meta["date"]`, comportamento IDÊNTICO
    ao de antes deste parâmetro): a coluna de agrupamento "dia" propriamente
    dita — B3/cripto-1d já têm `meta["date"]` na granularidade de dia, então
    o default sempre foi correto ali. O cripto 1h chama `montar_dataset_
    celula(..., normalizar_data=False)` para preservar a hora em `meta
    ["date"]` (necessário para `construir_trades`/embargo em barras) — sem
    um `chave_dia` explícito (`meta["date"].dt.normalize()`), este grupo-por
    aconteceria por BARRA, não por DIA-CALENDÁRIO, reabrindo exatamente a
    lição do "t inflado" (S142-143) dentro da própria prova de fogo.

    Devolve uma NOVA `pd.Series` (mesmo índice de `y`) — nunca muta `y`.
    """
    rng = np.random.default_rng(seed)
    chave = chave_dia if chave_dia is not None else meta["date"]
    df = pd.DataFrame({"y": y.to_numpy(), "_dia": pd.to_datetime(chave).to_numpy()})
    permutado = df.groupby("_dia")["y"].transform(lambda s: rng.permutation(s.to_numpy()))
    return pd.Series(permutado.to_numpy(), index=y.index, name=y.name)


def amostrar_uma_barra_por_ticker_dia(
    meta: pd.DataFrame, *, chave_dia: pd.Series, seed: int
) -> np.ndarray:
    """
    Achado MIE_CRIPTO_1H01 (prova de fogo R1: t_NW=+2,43 sob `y` permutado
    por dia, 11.109 trades, vs t=-1,07 no caminho real) — máscara booleana
    POSICIONAL (mesmo índice `0..N-1` de `meta`) que seleciona, para cada
    par (`ticker`, `chave_dia`), UMA única barra (sorteada com semente
    fixa). Usada por `scripts/mie_training/evaluate.py::provas_de_fogo_
    celula` para SUBAMOSTRAR `X`/`y`/`meta` antes de permutar+re-treinar na
    prova de fogo (a) do cripto 1h — nunca no B3/cripto-1d (lá 1 linha já É
    1 (ticker, dia), esta função não muda nada).

    Por quê (causa raiz, não sintoma): `permutar_y_por_dia` embaralha `y`
    DENTRO de cada dia preservando a PREVALÊNCIA do dia — correto e
    suficiente no B3/cripto-1d, onde cada ticker contribui EXATAMENTE 1
    linha por dia, então "quantos tickers hoje são 1" é uma medida
    genuinamente cross-sectional. No grid de 1h, com horizonte `h>=24`
    barras amostrado a cada 1 barra, um ÚNICO (ticker, dia) contribui até
    24 linhas cujo alvo (`exc_fwd_h`) é quase o MESMO evento (janelas
    97%+ sobrepostas para h=24) — não são 24 observações independentes.
    Colunas BROADCAST (`regime`, `breadth_sma50`, `ibov_dist_ema50` —
    calculadas do benchmark/universo, iguais para TODOS os tickers numa
    mesma hora, quase constantes dentro de um dia) são causais e legítimas,
    mas correlacionam com "quantas linhas do dia são 1" — exatamente a
    quantidade que a permutação preserva. Com até 24 cópias reforçando o
    MESMO evento por (ticker, dia), o walk-forward re-treinado com `y`
    lixo consegue aprender essa correlação de dia/regime com baixa
    variância (repetida 24x) e generalizar de ano em ano — um artefato da
    PROVA, não uma habilidade preditiva por instrumento/barra (que é
    exatamente o que a prova de fogo deveria estar medindo). Reduzir para
    1 linha por (ticker, dia) ANTES de permutar+re-treinar devolve a prova
    à mesma granularidade — 1 evento independente por (ticker, dia) — em
    que ela já era válida para B3/cripto-1d; a prova (b) (seleção aleatória
    do pool real, sem re-treino) não sofre disso e continua sem mudança.

    `chave_dia`: a MESMA chave passada a `permutar_y_por_dia` (dia
    normalizado) — o grupo de amostragem é (`ticker`, `chave_dia`), nunca
    (`ticker`, `meta["date"]` crua, que já É a barra).
    """
    posicoes = pd.DataFrame(
        {
            "_pos": np.arange(len(meta)),
            "_ticker": meta["ticker"].to_numpy(),
            "_dia": pd.to_datetime(chave_dia).to_numpy(),
        }
    )
    escolhidas = posicoes.groupby(["_ticker", "_dia"], sort=False, observed=True).sample(
        n=1, random_state=seed
    )
    mascara = np.zeros(len(meta), dtype=bool)
    mascara[escolhidas["_pos"].to_numpy()] = True
    return mascara


def montar_alvo(store: pd.DataFrame, horizonte: int, limiar: float) -> pd.Series:
    """
    `y = 1[exc_fwd_h > limiar]` — `NaN` onde `exc_fwd_h` for `NaN` (linha sem
    alvo resolvido: índice indisponível naquela data, ou horizonte alcança
    além do fim do histórico). Nunca inventa rótulo para linha sem alvo.
    """
    coluna = f"{PREFIXO_TARGET_EXC}{horizonte}"
    if coluna not in store.columns:
        raise ValueError(f"montar_alvo: coluna de alvo ausente no store: {coluna!r}")

    excesso = store[coluna]
    y = pd.Series(np.nan, index=store.index, dtype=float)
    valido = excesso.notna()
    y.loc[valido] = (excesso.loc[valido] > limiar).astype(float)
    return y


def juntar_regime_mie(
    store: pd.DataFrame, regime_mie_por_data: Mapping[pd.Timestamp, str]
) -> pd.DataFrame:
    """
    Broadcast por DATA de `regime_mie` (MIE-2) sobre o feature store — mesmo
    padrão de `radar.learning.feature_store._aplicar_regime`: uma data
    ausente do mapa vira `NaN` (nunca um regime inventado). `regime_mie_por_
    data` normalmente vem de `dict(zip(detectar_regime(precos)["date"],
    detectar_regime(precos)["regime_mie"]))`, calculado pelo CHAMADOR
    (`scripts/mie_training/train.py`) — este módulo não lê parquet nem chama
    `detectar_regime` diretamente.
    """
    saida = store.copy()
    datas = pd.to_datetime(saida["date"]).dt.normalize()
    mapa = {pd.Timestamp(k).normalize(): v for k, v in regime_mie_por_data.items()}
    saida["regime_mie"] = datas.map(mapa)
    return saida


def montar_dataset_celula(
    store_com_regime_mie: pd.DataFrame,
    celula: CelulaMIE,
    *,
    colunas_categoricas: tuple[str, ...] = COLUNAS_CATEGORICAS,
    normalizar_data: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    `(X, y, meta)` de UMA célula — linhas com alvo `NaN` já CAÍRAM (pré-
    registro §4: "Linhas com target NA caem"). `X` tem `colunas_categoricas`
    como dtype `category` (LightGBM as reconhece via `categorical_feature=
    "auto"`, ver `radar.mie.model`) — as categorias são fixadas sobre o
    dataset INTEIRO da célula, antes de qualquer split temporal, para que o
    código de categoria permaneça idêntico em todo ano do walk-forward
    (nenhum fold vê um subconjunto de categorias diferente do resto).

    `colunas_categoricas` (default `COLUNAS_CATEGORICAS` = `regime`/
    `regime_mie`, o par do B3 — comportamento IDÊNTICO ao de antes desta
    parametrização, bit a bit) é a ÚNICA coisa B3-hardcoded aqui: o mercado
    cripto (`docs/estudos/PRE_REGISTRO_MIE_CRIPTO_1D_R1.md` §1) passa
    `COLUNAS_CATEGORICAS_CRIPTO` (`regime`/`faixa_liquidez` — o store cripto
    já chega com `regime` pronto, sem um segundo "regime_mie" para juntar;
    ver docstring daquela constante). `X` tem sempre 43 + `len(colunas_
    categoricas)` colunas.

    `normalizar_data` (default `True`, comportamento IDÊNTICO ao de antes
    deste parâmetro): quando `True`, `meta["date"]` é normalizada para o DIA
    (`.dt.normalize()`) — correto para B3/cripto-1d, onde a granularidade
    nativa já é uma linha por (ticker, dia). O cripto 1h (`docs/estudos/
    PRE_REGISTRO_MIE_CRIPTO_1H_R1.md`) passa `False`: o store tem uma linha
    por (symbol, BARRA-HORA), e normalizar colapsaria todas as barras do
    mesmo dia no mesmo timestamp — quebraria o contrato "1 trade por
    (ticker, barra)" de `radar.mie.metrics.construir_trades` e a contagem
    por posição do embargo/holdout em barras (ver docstring do módulo,
    seção "Extensão CRIPTO 1h").

    `meta` carrega `ticker`, `date`, `ano` (ano civil de `date`) e `exc_fwd`
    (o excesso REAL no horizonte da célula, renomeado de `exc_fwd_{h}` para
    um nome genérico — o que os construtores de trade de `radar.mie.metrics`
    consomem, célula-agnóstico). `X`/`y`/`meta` compartilham o MESMO índice
    posicional `0..N-1` (reset), condição para os splits de `gerar_splits_
    walk_forward`/`gerar_split_final` (que devolvem máscaras booleanas
    posicionais) indexarem os três em conjunto.
    """
    y_bruto = montar_alvo(store_com_regime_mie, celula.horizonte, celula.limiar)
    valido = y_bruto.notna()

    base = store_com_regime_mie.loc[valido].reset_index(drop=True)
    y = y_bruto.loc[valido].reset_index(drop=True).astype(float)

    colunas_x = (*COLUNAS_FEATURES, *colunas_categoricas)
    faltantes = [c for c in colunas_x if c not in base.columns]
    if faltantes:
        raise ValueError(f"montar_dataset_celula: coluna(s) de X ausente(s) no store: {faltantes}")

    X = base[list(colunas_x)].copy()
    for coluna in colunas_categoricas:
        X[coluna] = X[coluna].astype("category")

    col_exc = f"{PREFIXO_TARGET_EXC}{celula.horizonte}"
    meta = base[["ticker", "date"]].copy()
    meta["date"] = pd.to_datetime(meta["date"])
    if normalizar_data:
        meta["date"] = meta["date"].dt.normalize()
    meta["ano"] = meta["date"].dt.year
    meta["exc_fwd"] = base[col_exc].astype(float)

    return X, y, meta


# ── FASE 1 do Plano de Reconstrução do Cérebro — filtro de universo ─────────
#
# Diagnóstico (sessão desta Fase): o universo de treino tinha lixo entrando
# sem filtro nenhum — `AAPLBUSDT` com 1 (uma) barra, ações tokenizadas
# (`NVDABUSDT`/`AMZNBUSDT`/`AMDBUSDT`, ...) misturadas com cripto nativo,
# sem mínimo de liquidez. `filtrar_universo_treino` é o filtro DECLARADO e
# PARAMETRIZADO (nunca tuning exploratório) aplicado pelo CHAMADOR (`scripts/
# mie_training/train.py`) sobre o store INTEIRO, ANTES de `montar_dataset_
# celula` — os 3 critérios são independentes e cada um é reportado à parte
# (`FiltroUniversoResultado.to_manifest`) para o manifest registrar quantos
# símbolos saíram e por qual motivo (um símbolo pode acionar mais de um
# critério ao mesmo tempo; a UNIÃO é removida uma única vez).

MIN_BARRAS_UNIVERSO_PADRAO = 252
"""Histórico mínimo (em barras do próprio store — pregões B3, dias corridos
cripto 1d) para um ticker ENTRAR no treino. ~1 ano de calendário — um
número REDONDO, declarado ANTES de olhar o resultado do experimento (não é
tuning): o bastante para o modelo ver pelo menos um ciclo anual de regime,
sem impor um mínimo tão alto que já eliminasse por construção ativos jovens
mas genuinamente líquidos."""

MIN_BARRAS_UNIVERSO_CRIPTO_1H_PADRAO = MIN_BARRAS_UNIVERSO_PADRAO * 24
"""Análogo de `MIN_BARRAS_UNIVERSO_PADRAO` em BARRAS-HORA (24 barras/dia) —
mesmo ~1 ano de cobertura, unidade diferente (grade horária do cripto 1h)."""

FAIXAS_LIQUIDEZ_EXCLUIDAS_PADRAO: tuple[str, ...] = ("iliquida",)
"""Tercil de liquidez excluído por default — mesma doutrina de `scripts/
build_evidence_ledger.py`/`radar.evidence.categorias.CONTEXTO_NAO_OPERAVEL`
("a ponta ilíquida não vira peso, o spread come o resultado antes de você"),
aqui aplicada no UNIVERSO DE TREINO em vez de post-hoc na medição: um ticker
cuja `faixa_liquidez` (tercil de volume mediano, já calculado por `radar.
learning.feature_store*`) é `"iliquida"` nunca entra em `X`/`y`."""


_TICKERS_TOKENIZADOS_CONHECIDOS: frozenset[str] = frozenset(
    {
        # Ações
        "AAOI",
        "AAPL",
        "AMAT",
        "AMD",
        "AMZN",
        "ARM",
        "AVGO",
        "AXTI",
        "BABA",
        "CBRS",
        "COIN",
        "CRCL",
        "CRWV",
        "DELL",
        "FLNC",
        "GLW",
        "GOOGL",
        "GS",
        "HOOD",
        "IBM",
        "INTC",
        "LITE",
        "META",
        "MRVL",
        "MSFT",
        "MSTR",
        "NBIS",
        "NOK",
        "NVDA",
        "ORCL",
        "PLTR",
        "PYPL",
        "QCOM",
        "RKLB",
        "SNDK",
        "SPCX",
        "TSLA",
        "TSM",
        "WDC",
        # ETFs/ETPs
        "EWY",
        "KORU",
        "QQQ",
        "SMH",
        "SOXL",
        "SOXS",
        "SPY",
        "TQQQ",
    }
)
"""
Base (ticker de bolsa, SEM o `B` de "Backed" nem o `USDT`) dos instrumentos
"Backed"/xStocks CONHECIDOS no universo cripto (`data/crypto_universe.json`,
auditado nesta Fase) — ações/ETFs REAIS replicados 1:1, nunca cripto nativo.
Lista CURADA (não um regex de sufixo — ver o incidente abaixo) — conservadora
de propósito: um falso NEGATIVO (uma ação tokenizada que escapa do filtro)
é muito mais barato que um falso POSITIVO (excluir um cripto nativo de
verdade). Símbolos com sufixo `"BUSDT"` que não pude confirmar como
ação/ETF real (`BEBUSDT`, `DRAMBUSDT`, `INTWBUSDT`, `MUBUSDT`, `MUUBUSDT`,
`MVLLBUSDT`, `SKHYBUSDT`, `SNXXBUSDT`, `YBUSDT`) foram DEIXADOS DE FORA —
permanecem no universo, dívida aceitável.

Por que NÃO é um regex de sufixo (`"BUSDT"`)
------------------------------------------------
A primeira versão desta função usava `ticker.endswith("BUSDT")` — parecia
seguro ("nenhum cripto nativo termina em B antes do USDT"), mas a rodada
real da FASE 1 provou o contrário: `ARBUSDT` (Arbitrum), `BNBUSDT` (Binance
Coin — um dos maiores do universo), `SHIBUSDT` (Shiba Inu), `CKBUSDT`
(Nervos Network), `DGBUSDT` (DigiByte) e `TRBUSDT` (Tellor) são cripto
NATIVO cujo TICKER PRÓPRIO termina em "B" (`ARB`, `BNB`, `SHIB`, `CKB`,
`DGB`, `TRB`) — o sufixo `"BUSDT"` aparece por ACASO, não por serem
"<ativo>" + "B" de Backed + "USDT". Um regex de sufixo não consegue
distinguir os dois casos; só uma lista CURADA do ativo de bolsa real
resolve sem excluir cripto de verdade.
"""


def eh_ticker_tokenizado(ticker: str) -> bool:
    """
    `True` só se `ticker` terminar em `"BUSDT"` E a parte ANTES desse
    sufixo (o ativo de bolsa, sem o `B` de "Backed" nem o `USDT`) estiver
    em `_TICKERS_TOKENIZADOS_CONHECIDOS` — ver a docstring da constante
    para o incidente que motivou trocar um regex de sufixo por uma lista
    curada. Símbolos B3 (`PETR4`, ...) nunca terminam em `"BUSDT"`, então
    esta função é um no-op seguro fora do universo cripto.
    """
    t = ticker.upper()
    if not t.endswith("BUSDT"):
        return False
    base = t[: -len("BUSDT")]
    return base in _TICKERS_TOKENIZADOS_CONHECIDOS


@dataclass(frozen=True)
class FiltroUniversoResultado:
    """Resultado de `filtrar_universo_treino`: o store JÁ filtrado
    (`store`) + as estatísticas por critério, prontas para `scripts/
    mie_training/train.py` registrar no `manifest.json` (`to_manifest`)."""

    store: pd.DataFrame
    n_tickers_antes: int
    n_tickers_depois: int
    removidos_historico_curto: tuple[str, ...]
    removidos_iliquidos: tuple[str, ...]
    removidos_tokenizados: tuple[str, ...]
    min_barras: int
    faixas_liquidez_excluidas: tuple[str, ...]

    @property
    def n_removidos_total(self) -> int:
        """Contagem de tickers DISTINTOS removidos (a UNIÃO dos 3
        critérios — um ticker acionando 2 critérios conta uma vez só),
        nunca a soma das 3 listas (que dupla-contaria)."""
        return self.n_tickers_antes - self.n_tickers_depois

    def to_manifest(self) -> dict[str, Any]:
        """Dict serializável em JSON — `scripts/mie_training/train.py`
        grava isto em `manifest.json["filtro_universo"]`."""
        return {
            "min_barras": self.min_barras,
            "faixas_liquidez_excluidas": list(self.faixas_liquidez_excluidas),
            "n_tickers_antes": self.n_tickers_antes,
            "n_tickers_depois": self.n_tickers_depois,
            "n_removidos_total": self.n_removidos_total,
            "removidos_historico_curto": {
                "n": len(self.removidos_historico_curto),
                "tickers": list(self.removidos_historico_curto),
            },
            "removidos_iliquidos": {
                "n": len(self.removidos_iliquidos),
                "tickers": list(self.removidos_iliquidos),
            },
            "removidos_tokenizados": {
                "n": len(self.removidos_tokenizados),
                "tickers": list(self.removidos_tokenizados),
            },
        }


def filtrar_universo_treino(
    store: pd.DataFrame,
    *,
    coluna_ticker: str = "ticker",
    min_barras: int = MIN_BARRAS_UNIVERSO_PADRAO,
    faixas_liquidez_excluidas: tuple[str, ...] = FAIXAS_LIQUIDEZ_EXCLUIDAS_PADRAO,
    excluir_tokenizados: bool = True,
) -> FiltroUniversoResultado:
    """
    Filtro de universo DECLARADO (FASE 1), aplicado sobre o store INTEIRO
    ANTES de `montar_dataset_celula` — 3 critérios independentes, cada
    ticker removido por PELO MENOS um deles nunca mais aparece em nenhuma
    célula desta rodada:

      1. `min_barras`: histórico TOTAL do ticker no store (todas as linhas,
         não só as com alvo resolvido) `< min_barras` — descarta ruído
         extremo tipo `AAPLBUSDT` com 1 barra.
      2. `faixas_liquidez_excluidas`: `faixa_liquidez` (já calculada por
         `radar.learning.feature_store*`, broadcast constante por ticker)
         dentro do conjunto excluído (default só `"iliquida"`). Ticker sem
         `faixa_liquidez` resolvida (`NaN` — universo pequeno demais para
         tercil, ou ticker fora do mapa) NÃO é removido por este critério
         — a régua nunca inventa uma faixa para decidir, e a ausência de
         dado não pode ser tratada como "ilíquido" por omissão.
      3. `excluir_tokenizados`: `eh_ticker_tokenizado(ticker)` — ver
         docstring da função.

    Levanta `ValueError` se `coluna_ticker` não existir em `store`, ou se
    `faixas_liquidez_excluidas` não for vazio e `store` não tiver a coluna
    `faixa_liquidez` (o critério 2 não pode ser SILENCIOSAMENTE ignorado
    por falta de coluna — isso esconderia um filtro que o chamador pediu
    explicitamente).
    """
    if coluna_ticker not in store.columns:
        raise ValueError(f"filtrar_universo_treino: coluna de ticker ausente: {coluna_ticker!r}")
    if faixas_liquidez_excluidas and "faixa_liquidez" not in store.columns:
        raise ValueError(
            "filtrar_universo_treino: faixas_liquidez_excluidas pedido, mas o store não tem "
            "a coluna 'faixa_liquidez' (esperada de radar.learning.feature_store*)"
        )

    tickers = store[coluna_ticker]
    n_tickers_antes = int(tickers.nunique())

    contagem_por_ticker = store.groupby(coluna_ticker).size()
    removidos_historico_curto = tuple(
        sorted(str(t) for t in contagem_por_ticker[contagem_por_ticker < min_barras].index)
    )

    removidos_iliquidos: tuple[str, ...] = ()
    if faixas_liquidez_excluidas:
        faixa_por_ticker = store.groupby(coluna_ticker)["faixa_liquidez"].first()
        removidos_iliquidos = tuple(
            sorted(
                str(t)
                for t in faixa_por_ticker[faixa_por_ticker.isin(faixas_liquidez_excluidas)].index
            )
        )

    removidos_tokenizados: tuple[str, ...] = ()
    if excluir_tokenizados:
        removidos_tokenizados = tuple(
            sorted(str(t) for t in tickers.unique() if eh_ticker_tokenizado(str(t)))
        )

    todos_removidos = (
        set(removidos_historico_curto) | set(removidos_iliquidos) | set(removidos_tokenizados)
    )
    store_filtrado = store.loc[~tickers.isin(todos_removidos)].reset_index(drop=True)
    n_tickers_depois = (
        int(store_filtrado[coluna_ticker].nunique()) if not store_filtrado.empty else 0
    )

    return FiltroUniversoResultado(
        store=store_filtrado,
        n_tickers_antes=n_tickers_antes,
        n_tickers_depois=n_tickers_depois,
        removidos_historico_curto=removidos_historico_curto,
        removidos_iliquidos=removidos_iliquidos,
        removidos_tokenizados=removidos_tokenizados,
        min_barras=min_barras,
        faixas_liquidez_excluidas=faixas_liquidez_excluidas,
    )


# ── Walk-forward com embargo + holdout de calibração (pré-registro §4) ──────


def _posicao_no_calendario(calendario: np.ndarray, data: pd.Timestamp) -> int:
    """Posição de inserção (lado esquerdo) de `data` em `calendario` (array
    `datetime64` ordenado, sem duplicatas) — `calendario[pos]` é a primeira
    data `>= data` (ou `len(calendario)` se `data` é posterior a tudo)."""
    return int(np.searchsorted(calendario, np.datetime64(data), side="left"))


def _data_maxima_com_embargo(
    calendario: np.ndarray, corte: pd.Timestamp, pregoes: int
) -> pd.Timestamp | None:
    """
    Última data do calendário estritamente anterior a `corte` que fica a
    PELO MENOS `pregoes` pregões de distância de `corte` — a fronteira do
    embargo, sempre no FIM da janela que precede `corte` (nunca um
    deslocamento de `corte` para frente). `None` quando o histórico
    disponível é insuficiente (`pos - pregoes - 1 < 0`).
    """
    pos = _posicao_no_calendario(calendario, corte)
    idx = pos - pregoes - 1
    if idx < 0:
        return None
    return pd.Timestamp(calendario[idx])


def _data_maxima_com_embargo_dias_corridos(
    calendario: np.ndarray, corte: pd.Timestamp, dias: int
) -> pd.Timestamp | None:
    """
    Variante em DIAS CORRIDOS de `_data_maxima_com_embargo` — parametrização
    do MIE-CRIPTO (`docs/estudos/PRE_REGISTRO_MIE_CRIPTO_1D_R1.md` §4:
    "embargo 21 dias corridos" — o calendário cripto é 24/7, então o embargo
    conta DIAS de relógio, não POSIÇÕES no array de datas distintas do
    calendário).

    Última data do calendário `<= corte - Timedelta(days=dias + 1)` — o
    `+1` replica a MESMA folga estrita de `_data_maxima_com_embargo`
    (naquela versão, o boundary fica `pregoes + 1` POSIÇÕES antes de
    `corte`, nunca `pregoes`: a fronteira em si não conta como parte do
    embargo, só o que fica ESTRITAMENTE entre ela e `corte`) — sem o `+1`,
    esta versão em dias corridos coincidiria com a por pregões só por
    acaso, nunca por construção; com ele, as duas convenções produzem a
    MESMA data em QUALQUER calendário denso sem gap (`dias` dias corridos
    seguidos == `dias` posições distintas seguidas), provado em `tests/
    unit/test_mie_dataset.py::TestEmbargoDiasCorridos::
    test_dias_corridos_coincide_com_pregoes_sem_gap`.

    A diferença para `_data_maxima_com_embargo` só aparece quando o
    calendário tem um GAP (um trecho sem NENHUM símbolo negociado, ex.:
    falha de coleta): a versão por PREGÕES conta POSIÇÕES presentes e, ao
    atravessar o buraco, recua no tempo mais do que os `dias` pedidos (ela
    nunca "vê" o gap, só conta quantas entradas existem); esta versão
    calendário nunca é afetada pelo TAMANHO do gap — usa SEMPRE a distância
    real em dias corridos entre `corte` e a fronteira do embargo, gap ou
    não. No dado real cripto (471 pares, calendário efetivamente sem furos
    agregados) as duas coincidem quase sempre; a diferença some à tona só
    sob um gap real — ver `tests/unit/test_mie_dataset.py::
    TestEmbargoDiasCorridos::test_dias_corridos_diverge_de_pregoes_sob_gap_
    no_calendario` para a prova com um gap sintético. `None` quando o
    histórico disponível não alcança `dias + 1` dias de distância (nenhuma
    data `<= corte - (dias + 1)`).
    """
    limite = pd.Timestamp(corte) - pd.Timedelta(days=dias + 1)
    pos = int(np.searchsorted(calendario, np.datetime64(limite), side="right"))
    idx = pos - 1
    if idx < 0:
        return None
    return pd.Timestamp(calendario[idx])


_UNIDADES_EMBARGO_VALIDAS = ("pregoes", "dias_corridos", "barras")


def _data_maxima_com_embargo_unidade(
    calendario: np.ndarray, corte: pd.Timestamp, embargo: int, unidade_embargo: str
) -> pd.Timestamp | None:
    """Despacha para `_data_maxima_com_embargo` (B3, `unidade_embargo=
    "pregoes"`, default — comportamento IDÊNTICO ao de antes desta
    parametrização), `_data_maxima_com_embargo_dias_corridos` (CRIPTO 1d,
    `unidade_embargo="dias_corridos"`), ou de volta a `_data_maxima_com_
    embargo` para `unidade_embargo="barras"` (CRIPTO 1h) — `"barras"` é um
    ALIAS deliberado de `"pregoes"`: a função já conta POSIÇÕES de valores
    distintos no `calendario`, nunca dias de calendário, então ela É uma
    contagem de barras sempre que `calendario` vier de timestamps de barra
    em vez de datas normalizadas para o dia (ver docstring do módulo, seção
    "Extensão CRIPTO 1h") — dois nomes para a MESMA mecânica, documentando a
    intenção de cada chamador."""
    if unidade_embargo in ("pregoes", "barras"):
        return _data_maxima_com_embargo(calendario, corte, embargo)
    if unidade_embargo == "dias_corridos":
        return _data_maxima_com_embargo_dias_corridos(calendario, corte, embargo)
    raise ValueError(
        f"unidade_embargo desconhecida: {unidade_embargo!r} — use um de {_UNIDADES_EMBARGO_VALIDAS}"
    )


EMBARGO_PREGOES_PADRAO = 21
ANO_PRIMEIRO_TESTE_PADRAO = 2013
MESES_HOLDOUT_PADRAO = 12

EMBARGO_DIAS_CORRIDOS_CRIPTO_PADRAO = 21
"""Pré-registro CRIPTO §4: "embargo 21 dias corridos" — MESMO número que
`EMBARGO_PREGOES_PADRAO`, unidade DIFERENTE (ver `_data_maxima_com_embargo_
dias_corridos`/`unidade_embargo="dias_corridos"`)."""

ANO_PRIMEIRO_TESTE_CRIPTO_PADRAO = 2021
"""Pré-registro CRIPTO §4: "1º treino 2017 → 2020-12-31; teste anual 2021 →
último ano com target não-nulo"."""

EMBARGO_BARRAS_CRIPTO_1H_PADRAO = 168
"""Pré-registro CRIPTO 1h §4: "embargo de 168 barras-hora (o maior horizonte
da família)" — usado com `unidade_embargo="barras"`."""

HOLDOUT_BARRAS_CRIPTO_1H_PADRAO = 2160
"""Pré-registro CRIPTO 1h §4: "holdout interno... últimas ~2.160 barras
(~90 dias)" — usado como `holdout_barras` de `gerar_splits_walk_forward`/
`gerar_split_final`, no lugar de `meses_holdout` (2.160 horas não é um
número exato de meses)."""

ANO_PRIMEIRO_TESTE_CRIPTO_1H_PADRAO = ANO_PRIMEIRO_TESTE_CRIPTO_PADRAO
"""Reexport DIRETO (MESMO valor, 2021) de `ANO_PRIMEIRO_TESTE_CRIPTO_PADRAO`
— o pré-registro CRIPTO 1h §4 pede o MESMO ano de início ("1º treino 2017 →
2020-12-31; teste anual 2021...") do cripto 1d. Constante PRÓPRIA (não um
uso direto da 1d) só para quem lê `scripts/mie_training/train.py --mercado
cripto_1h` não precisar saber que o valor "1d" é o mesmo — nunca diverge em
silêncio porque é o MESMO objeto/valor."""


@dataclass(frozen=True)
class SplitWF:
    """Um ano de teste do walk-forward — três máscaras booleanas POSICIONAIS
    (alinhadas com `X`/`y`/`meta` de `montar_dataset_celula`) mutuamente
    exclusivas (as folgas de embargo não pertencem a nenhuma das três)."""

    ano_teste: int
    idx_fit: np.ndarray
    idx_holdout: np.ndarray
    idx_teste: np.ndarray
    data_max_treino_bruto: pd.Timestamp
    data_max_fit: pd.Timestamp
    data_min_holdout: pd.Timestamp
    data_max_holdout: pd.Timestamp
    data_min_teste: pd.Timestamp
    data_max_teste: pd.Timestamp


def _fronteira_holdout(
    calendario: np.ndarray,
    datas: pd.Series,
    idx_treino_bruto: pd.Series,
    data_max_treino_bruto: pd.Timestamp,
    *,
    meses_holdout: int,
    holdout_barras: int | None,
) -> tuple[pd.Series, pd.Timestamp] | None:
    """
    Fronteira do holdout de calibração — `(idx_holdout, holdout_start_real)`
    ou `None` se o histórico dentro de `idx_treino_bruto` for curto demais.
    Duas convenções, despachadas por `holdout_barras`:

      - `holdout_barras=None` (default — comportamento IDÊNTICO ao de antes
        deste helper existir, B3/cripto-1d): últimos `meses_holdout` MESES
        de calendário do treino bruto (`pd.DateOffset`).
      - `holdout_barras=N` (cripto 1h, pré-registro §4: "~2.160 barras"):
        últimas `N` entradas distintas do `calendario` dentro do treino
        bruto, por POSIÇÃO — mesmo raciocínio (nunca calendário) de
        `_data_maxima_com_embargo`, necessário porque 2.160 barras-hora não
        é um número exato de meses. `data_max_treino_bruto` é sempre um
        elemento do `calendario` (devolvido por `_data_maxima_com_embargo*`
        ou, em `gerar_split_final`, a última data de `datas`), então sua
        posição é exata.
    """
    if holdout_barras is not None:
        pos_fim = _posicao_no_calendario(calendario, data_max_treino_bruto)
        pos_inicio = pos_fim - holdout_barras + 1
        if pos_inicio < 0:
            return None
        holdout_start_real = pd.Timestamp(calendario[pos_inicio])
        idx_holdout = idx_treino_bruto & (datas >= holdout_start_real)
        if not idx_holdout.any():
            return None
        return idx_holdout, holdout_start_real

    holdout_cutoff_raw = data_max_treino_bruto - pd.DateOffset(months=meses_holdout)
    idx_holdout = idx_treino_bruto & (datas > holdout_cutoff_raw)
    if not idx_holdout.any():
        return None
    pos_holdout_start = _posicao_no_calendario(
        calendario, holdout_cutoff_raw + pd.Timedelta(days=1)
    )
    if pos_holdout_start >= len(calendario):
        return None
    holdout_start_real = pd.Timestamp(calendario[pos_holdout_start])
    return idx_holdout, holdout_start_real


def gerar_splits_walk_forward(
    meta: pd.DataFrame,
    *,
    embargo_pregoes: int = EMBARGO_PREGOES_PADRAO,
    ano_primeiro_teste: int = ANO_PRIMEIRO_TESTE_PADRAO,
    meses_holdout: int = MESES_HOLDOUT_PADRAO,
    unidade_embargo: str = "pregoes",
    holdout_barras: int | None = None,
    normalizar_data: bool = True,
) -> Iterator[SplitWF]:
    """
    Um `SplitWF` por ano testável, em ORDEM crescente. `meta` precisa ter a
    coluna `date` (de `montar_dataset_celula`, já sem linhas de alvo `NaN`).
    Um ano sem nenhuma linha de teste (ex.: fora do range do dado) ou sem
    histórico suficiente para o embargo/holdout é SILENCIOSAMENTE pulado —
    o chamador (`radar.mie.pipeline.rodar_walk_forward_celula`) decide o que
    fazer com uma família de splits mais curta que o range nominal de anos.

    `unidade_embargo` (default `"pregoes"` — comportamento IDÊNTICO ao de
    antes desta parametrização, bit a bit): `embargo_pregoes` conta POSIÇÕES
    de datas distintas no calendário (B3, pregões — e CRIPTO 1h, barras-hora,
    ver `"barras"` em `_data_maxima_com_embargo_unidade`). `"dias_corridos"`
    (MIE-CRIPTO 1d, calendário 24/7) faz `embargo_pregoes` contar DIAS de
    relógio em vez de posições — ver `_data_maxima_com_embargo_dias_corridos`
    para a diferença exata (só aparece sob um gap real no calendário). O NOME
    do parâmetro (`embargo_pregoes`) fica como está por compatibilidade — o
    valor é sempre "a quantidade do embargo, na unidade de `unidade_
    embargo`", nunca necessariamente pregões.

    `holdout_barras`/`normalizar_data` (defaults `None`/`True` — IDÊNTICOS
    ao comportamento de antes destes 2 parâmetros existirem): ver docstring
    do módulo, seção "Extensão CRIPTO 1h", e `_fronteira_holdout`.
    """
    datas = pd.to_datetime(meta["date"])
    if normalizar_data:
        datas = datas.dt.normalize()
    calendario = np.array(sorted(datas.unique()))
    if calendario.size == 0:
        return

    ano_max = int(datas.dt.year.max())

    for ano_teste in range(ano_primeiro_teste, ano_max + 1):
        cutoff_teste = pd.Timestamp(ano_teste, 1, 1)
        # Limite superior EXCLUSIVO (início do ano seguinte) em vez de
        # `<= 31/12` — equivalente byte a byte quando `datas` está
        # normalizada para o dia (todo timestamp cai à meia-noite), e
        # correto também para `normalizar_data=False` (grade horária: inclui
        # TODAS as barras de 31/12, não só a de 00:00).
        limite_superior_teste = pd.Timestamp(ano_teste + 1, 1, 1)

        idx_teste = (datas >= cutoff_teste) & (datas < limite_superior_teste)
        if not idx_teste.any():
            continue

        data_max_treino_bruto = _data_maxima_com_embargo_unidade(
            calendario, cutoff_teste, embargo_pregoes, unidade_embargo
        )
        if data_max_treino_bruto is None:
            continue

        idx_treino_bruto = datas <= data_max_treino_bruto
        if not idx_treino_bruto.any():
            continue

        fronteira = _fronteira_holdout(
            calendario,
            datas,
            idx_treino_bruto,
            data_max_treino_bruto,
            meses_holdout=meses_holdout,
            holdout_barras=holdout_barras,
        )
        if fronteira is None:
            continue
        idx_holdout, holdout_start_real = fronteira

        data_max_fit = _data_maxima_com_embargo_unidade(
            calendario, holdout_start_real, embargo_pregoes, unidade_embargo
        )
        if data_max_fit is None:
            continue
        idx_fit = idx_treino_bruto & (datas <= data_max_fit)
        if not idx_fit.any():
            continue

        yield SplitWF(
            ano_teste=ano_teste,
            idx_fit=idx_fit.to_numpy(),
            idx_holdout=idx_holdout.to_numpy(),
            idx_teste=idx_teste.to_numpy(),
            data_max_treino_bruto=data_max_treino_bruto,
            data_max_fit=datas[idx_fit].max(),
            data_min_holdout=datas[idx_holdout].min(),
            data_max_holdout=datas[idx_holdout].max(),
            data_min_teste=datas[idx_teste].min(),
            data_max_teste=datas[idx_teste].max(),
        )


@dataclass(frozen=True)
class SplitFinal:
    """Split do modelo de PRODUÇÃO (pré-registro §7) — sem teste OOS, só
    fit + holdout de calibração sobre TODO o histórico disponível."""

    idx_fit: np.ndarray
    idx_holdout: np.ndarray
    data_max_fit: pd.Timestamp
    data_min_holdout: pd.Timestamp
    data_max_holdout: pd.Timestamp
    data_corte: pd.Timestamp


def gerar_split_final(
    meta: pd.DataFrame,
    *,
    embargo_pregoes: int = EMBARGO_PREGOES_PADRAO,
    meses_holdout: int = MESES_HOLDOUT_PADRAO,
    unidade_embargo: str = "pregoes",
    holdout_barras: int | None = None,
    normalizar_data: bool = True,
) -> SplitFinal:
    """
    Split do modelo de PRODUÇÃO (pré-registro §7: "re-treinado com TODO o
    histórico até a data de corte, mesma config"): `data_corte` é a última
    data presente em `meta` (o "hoje" deste treino); fit/holdout tirados
    dela com a MESMA máquina de embargo do walk-forward, sem teste OOS.
    Levanta `ValueError` se o histórico for curto demais para o embargo ou
    o holdout (degenerado — não esperado no dado real de 2005-2026).

    `unidade_embargo`/`holdout_barras`/`normalizar_data` — mesmos
    parâmetros/mesmos defaults de `gerar_splits_walk_forward`, ver
    docstring lá e `_fronteira_holdout`.
    """
    datas = pd.to_datetime(meta["date"])
    if normalizar_data:
        datas = datas.dt.normalize()
    calendario = np.array(sorted(datas.unique()))
    if calendario.size == 0:
        raise ValueError("gerar_split_final: meta vazio — nenhuma data disponível")

    data_corte = datas.max()
    idx_treino_bruto = pd.Series(
        True, index=datas.index
    )  # tudo é "treino bruto" (sem teste OOS aqui)

    fronteira = _fronteira_holdout(
        calendario,
        datas,
        idx_treino_bruto,
        data_corte,
        meses_holdout=meses_holdout,
        holdout_barras=holdout_barras,
    )
    if fronteira is None:
        raise ValueError("gerar_split_final: histórico insuficiente para o holdout de calibração")
    idx_holdout, holdout_start_real = fronteira

    data_max_fit = _data_maxima_com_embargo_unidade(
        calendario, holdout_start_real, embargo_pregoes, unidade_embargo
    )
    if data_max_fit is None:
        raise ValueError("gerar_split_final: histórico insuficiente para o embargo fit/holdout")
    idx_fit = datas <= data_max_fit
    if not idx_fit.any():
        raise ValueError("gerar_split_final: nenhuma linha de fit sobrou após o embargo")

    return SplitFinal(
        idx_fit=idx_fit.to_numpy(),
        idx_holdout=idx_holdout.to_numpy(),
        data_max_fit=datas[idx_fit].max(),
        data_min_holdout=datas[idx_holdout].min(),
        data_max_holdout=datas[idx_holdout].max(),
        data_corte=data_corte,
    )

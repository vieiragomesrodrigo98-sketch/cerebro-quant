"""
src/radar/features/colunas.py — o vocabulário de colunas, num lugar só.

**Este módulo não importa nada.** É a folha da árvore de dependências, e é
isso que o torna útil: tanto a camada de dados (`radar.learning.feature_store*`)
quanto os motores (`radar.mie`) quanto o contrato (`radar.cerebro`) podem
depender dele sem que nenhum precise depender dos outros.

Por que existe (S169)
---------------------
O espaço de SELEÇÃO nascia em `radar.mie.dataset`, e `radar.cerebro.catalogo`
precisava dele para declarar famílias — violando a invariante que o próprio
`radar.cerebro.__init__` declara ("zero import de `radar.mie`; a dependência é
motor → contrato, nunca o inverso").

A saída óbvia — mover para `radar.learning` — **não resolveria**: o
`feature_store` importa `radar.mie.regime` no topo do módulo, então o Cérebro
passaria a arrastar `radar.mie` transitivamente. Um teste que lê imports diretos
ficaria verde e a arquitetura não teria mudado. Era relabeling, não conserto.

A única costura honesta é uma **folha sem dependência alguma**, e é esta. O
ciclo de pacote `radar.learning ⇄ radar.mie` continua existindo por causa de
`_adx`/`_breadth_sma50` (card `GOV_MIE_PACOTE_NAO_CONGELADO01`) — mas o
vocabulário de colunas saiu dele.

Quem é o dono de quê
--------------------
Aqui vivem apenas **nomes de coluna e suas derivações** — nunca cálculo. Quem
*materializa* cada coluna continua em `radar.learning.feature_store*`; quem
monta X/y por célula continua em `radar.mie.dataset`. Os dois módulos
reexportam o que sempre exportaram, então nenhum chamador precisou mudar.

Regra ao acrescentar coluna: acrescente na tupla de origem
(`_COLUNAS_TECNICAS`, `_COLUNAS_MIE_*`, …). Os espaços de seleção são
**derivados por compreensão** e absorvem a mudança sozinhos — e o G-P1 do
Contrato de Pensamento (`radar.cerebro.contrato`) é quem decide se ela podia
entrar no espaço de seleção.
"""

from __future__ import annotations

# ── Chave e features materializadas (origem: radar.learning.feature_store) ───

COLUNAS_CHAVE: tuple[str, ...] = ("ticker", "date")
"""Chave única da tabela — uma linha por (ticker, date)."""

_COLUNAS_TECNICAS: tuple[str, ...] = (
    "rsi_14",
    "atr_14_pct",
    "razao_atr_5_20",
    "dist_sma_20",
    "dist_sma_50",
    "dist_sma_200",
    "retorno_5",
    "retorno_20",
    "retorno_60",
    "retorno_120",
    "retorno_250",
    "razao_volume_20",
    "largura_banda_20_2",
    "distancia_maxima_20",
    "topo_idade_1_5",
    "fundo_idade_1_5",
)
"""As 16 colunas técnicas causais (derivadas incluídas: `atr_14_pct` =
`atr_14/Close`; `dist_sma_{20,50,200}` = `Close/sma_n - 1` — as colunas cruas
`atr_14`/`sma_20`/`sma_50`/`sma_200` NUNCA entram na saída, só existem de
passagem em `_materializar_tecnicas`)."""

_COLUNAS_TEMPORAIS: tuple[str, ...] = ("dia_da_semana", "dia_util_do_mes", "dias_uteis_ate_fim_mes")
"""As 3 colunas temporais/calendário — junto de `_COLUNAS_TECNICAS`, fecham
as "19 colunas causais pré-declaradas" do plano original (pré-MIE-1)."""

# ── MIE-1 (ADR 0028 §5/§6) — features novas, por família/camada ─────────────
# Todas POR TICKER (materializadas dentro de `_features_mie_ticker`), exceto
# `_COLUNAS_MIE_CONTEXTO_BROADCAST` (por DATA, ver `_aplicar_contexto_mie`) e
# `cs_rank_ret20`/`cs_rank_ret60` dentro de `_COLUNAS_MIE_ENGINEERING` (rank
# CROSS-SECTIONAL por data, ver `_aplicar_cross_sectional`). Ver docstring do
# módulo para a fórmula/decisão de cada uma.
_COLUNAS_MIE_PRICE_ACTION: tuple[str, ...] = (
    "gap_abertura",
    "body_ratio",
    "wick_superior_pct",
    "wick_inferior_pct",
    "dist_max_252",
)
_COLUNAS_MIE_TENDENCIA: tuple[str, ...] = (
    "ema_21",
    "ema_50",
    "ema21_slope_5",
    "adx_14",
)
_COLUNAS_MIE_MOMENTUM: tuple[str, ...] = (
    "macd_hist",
    "estocastico_k_14",
)
_COLUNAS_MIE_VOLUME: tuple[str, ...] = (
    "obv_slope_20",
    "updown_vol_ratio_21",
)
_COLUNAS_MIE_CONTEXTO_TICKER: tuple[str, ...] = ("beta_63",)
_COLUNAS_MIE_CONTEXTO_BROADCAST: tuple[str, ...] = ("ibov_dist_ema50", "breadth_sma50")
_COLUNAS_MIE_ENGINEERING: tuple[str, ...] = (
    "z_rsi_252",
    "z_rvol_252",
    "cs_rank_ret20",
    "cs_rank_ret60",
    "delta_adx_5",
    "macd_hist_sign_change",
)

_COLUNAS_MIE_TODAS: tuple[str, ...] = (
    *_COLUNAS_MIE_PRICE_ACTION,
    *_COLUNAS_MIE_TENDENCIA,
    *_COLUNAS_MIE_MOMENTUM,
    *_COLUNAS_MIE_VOLUME,
    *_COLUNAS_MIE_CONTEXTO_TICKER,
    *_COLUNAS_MIE_CONTEXTO_BROADCAST,
    *_COLUNAS_MIE_ENGINEERING,
)
"""As 22 colunas novas do MIE-1 — usada só para montar `COLUNAS_FEATURES` e o
denominador ampliado de `resumir_feature_store` (chave `"tecnicas"`)."""

COLUNAS_FEATURES: tuple[str, ...] = (
    "close",
    "volume",
    *_COLUNAS_TECNICAS,
    *_COLUNAS_TEMPORAIS,
    *_COLUNAS_MIE_TODAS,
)
"""Tupla completa das features: base (`close` ajustado, `volume` bruto) + as
19 colunas causais originais (16 técnicas + 3 temporais) + as 22 colunas
novas do MIE-1 (ADR 0028 — price action/tendência/momentum/volume/contexto +
feature engineering §5; ver docstring do módulo). Nunca inclui target nem
coluna de contexto (regime/liquidez/evento) — essas têm suas próprias
constantes (`COLUNAS_REGIME`/`COLUNAS_LIQUIDEZ`/`COLUNAS_EVENTO`) porque não
são "causais" no mesmo sentido determinístico (dependem de fontes externas
que podem ficar indisponíveis, daí o `pd.NA` em vez de feature ausente)."""

# ── Espaços de TREINO e de SELEÇÃO (origem: radar.mie.dataset) ───────────────
#
# `COLUNAS_X*` é o espaço de TREINO; `COLUNAS_SELECAO_*` é o de SELEÇÃO, e a
# diferença entre os dois é o G-P1 do Contrato de Pensamento: contexto entra
# como PARTIÇÃO, nunca como feature que compete dentro de um alvo agrupado.

# ── X: 43 features (reusadas) + 2 categóricas ────────────────────────────────

COLUNAS_CATEGORICAS: tuple[str, ...] = ("regime", "regime_mie")
"""`regime`: broadcast do feature store (`radar.historical.regime_timeline`,
janela de calendário fixa). `regime_mie`: MIE-2 (`radar.mie.regime.
detectar_regime`, rule-based diário) — juntado por `juntar_regime_mie`, NUNCA
vem pronto no parquet do feature store (calculado à parte pelo chamador)."""

COLUNAS_X: tuple[str, ...] = (*COLUNAS_FEATURES, *COLUNAS_CATEGORICAS)
"""As 43 + 2 = 45 colunas de entrada do GBM. `COLUNAS_FEATURES` é IMPORTADO
(nunca copiado à mão) de `radar.learning.feature_store` — é a mesma lista que
já garante, por construção, nunca conter prefixo de target (ver docstring
daquele módulo, doutrina anti-mineração); `TestAntiLookahead` (testes deste
módulo) reafirma essa garantia na fronteira específica do MIE-3."""

# ── X do CRIPTO: mesmas 43 features + 2 categóricas PRÓPRIAS (pré-reg §1) ────

COLUNAS_CATEGORICAS_CRIPTO: tuple[str, ...] = ("regime", "faixa_liquidez")
"""Parametrização MÍNIMA do B3-hardcoded (pré-registro CRIPTO §1: "categóricas
`regime` (`regime_cripto` rule-based) e `faixa_liquidez`"). Diferente do B3
(`COLUNAS_CATEGORICAS` = `regime`/`regime_mie`, duas fontes SEPARADAS — a
`RegimeTimeline` macro do feature store e o MIE-2 recalculado à parte), o
store cripto (`radar.learning.feature_store_cripto`) já chega com UMA única
coluna `regime` (broadcast do rule-based `radar.historical.regime_cripto.
detectar_regime_cripto`, calculada por quem monta o store — NUNCA recalculada
aqui, ver docstring de `scripts/mie_training/train.py`) — não há um segundo
regime "MIE" separado para juntar. O pré-registro cripto usa a vaga da
segunda categórica para `faixa_liquidez` (tercis de quote-volume, régua
própria cripto) em vez de repetir `regime` duas vezes."""

COLUNAS_X_CRIPTO: tuple[str, ...] = (*COLUNAS_FEATURES, *COLUNAS_CATEGORICAS_CRIPTO)
"""As mesmas 43 `COLUNAS_FEATURES` (objeto IDÊNTICO ao B3, ver `COLUNAS_X`) +
2 categóricas cripto = 45 colunas — mesma contagem do B3, categóricas
diferentes."""

# ── X do CRIPTO 1h: MESMO objeto do cripto 1d (schema idêntico) ─────────────

COLUNAS_CATEGORICAS_CRIPTO_1H: tuple[str, ...] = COLUNAS_CATEGORICAS_CRIPTO
"""Reexport DIRETO (MESMO objeto, não uma cópia) de `COLUNAS_CATEGORICAS_
CRIPTO` — o store `feature_store_cripto_1h` chega com `regime`/
`faixa_liquidez` PRONTOS, exatamente como o store 1d (mesma régua rule-based
`radar.historical.regime_cripto`, só alimentada com barras de 1h em vez de 1d).
Nenhum "regime_mie" separado a juntar em nenhum dos dois."""

COLUNAS_CONTEXTO_DIA: tuple[str, ...] = (
    "ibov_dist_ema50",
    "breadth_sma50",
    "dia_da_semana",
    "dia_util_do_mes",
    "dias_uteis_ate_fim_mes",
)
"""As features NUMÉRICAS de `COLUNAS_FEATURES` que são constantes DENTRO do
dia — medidas, não presumidas (2026-08-04, sobre `data/learning/feature_store_
diario_mie.parquet`, 747 dias e 162 tickers): a fração da variância que é
ENTRE dias é 1,0000 nas quatro primeiras e 0,9997 em `dia_util_do_mes`, e a
mediana de valores distintos dentro de um dia é 1 nas cinco. As duas de
broadcast (`ibov_dist_ema50`/`breadth_sma50`) descrevem o MERCADO e as três de
calendário descrevem a DATA — nenhuma delas pode, por construção, separar um
ativo de outro no mesmo dia.

Somadas às categóricas de regime (`COLUNAS_CATEGORICAS` no B3,
`regime` no cripto), são as 7 colunas de 45 que fizeram o MIE v1 responder o
mesmo número para todos os ativos do dia — a de maior *gain* no modelo em
produção era `breadth_sma50`. Elas NÃO são lixo: são CONTEXTO, e contexto tem
outro papel (partição pré-registrada, `radar.cerebro.contrato.
DIMENSOES_CONTEXTO`), nunca o de feature competindo com features de ativo
dentro de um alvo agrupado. Ver G-P1 do Contrato de Pensamento (ADR 0032)."""

COLUNAS_SELECAO_B3: tuple[str, ...] = tuple(
    c for c in COLUNAS_X if c not in COLUNAS_CONTEXTO_DIA and c not in COLUNAS_CATEGORICAS
)
"""Espaço de SELEÇÃO do B3 (v2): `COLUNAS_X` menos contexto-do-dia e menos as
categóricas de regime. Derivado por compreensão, nunca redigitado — se
`COLUNAS_FEATURES` ganhar uma coluna, ela entra aqui automaticamente e o
G-P1 é quem decide se ela podia entrar."""

COLUNAS_SELECAO_CRIPTO: tuple[str, ...] = tuple(
    c
    for c in COLUNAS_X_CRIPTO
    if c not in COLUNAS_CONTEXTO_DIA and c not in COLUNAS_CATEGORICAS_CRIPTO
)
"""Espaço de SELEÇÃO do cripto (v2). Note que `faixa_liquidez` sai daqui por
ser categórica de contexto — mas, ao contrário de `regime`, ela NÃO é
constante no dia (é um atributo do par, varia entre pares e quase não varia no
tempo). Sai por decisão de VOCABULÁRIO (liquidez é uma `DIMENSOES_CONTEXTO`,
logo é eixo de partição), não por falha no G-P1 — e a distinção importa: o
G-P1 é uma checagem mecânica, o vocabulário é uma decisão declarada."""

# ── Espaço de seleção NATIVO de cripto (ADR0031_F2_DADOS_NATIVOS01) ─────────

COLUNAS_CALENDARIO_B3: frozenset[str] = frozenset({
    "dia_util_do_mes", "dias_uteis_ate_fim_mes", "dia_da_semana", "ibov_dist_ema50",
})
"""Colunas que descrevem o calendário e o índice da B3.

Elas não têm significado num mercado 24/7 sem pregão, sem feriado e sem
Ibovespa — e não são inofensivas: `dia_util_do_mes` saiu como **6º sinal mais
forte do cripto** (IC +0,049, `t` +5,07) no v1. Onde o modelo achou sinal ali,
achou artefato (`CEREBRO_STORE_CRIPTO_CALENDARIO_B301`).

**Medido ao aplicar este filtro: ele é no-op sobre o espaço de SELEÇÃO.** As
quatro já saíam por `COLUNAS_CONTEXTO_DIA` — são constantes no instante e o
G-P1 as barra sozinho. O artefato do v1 veio de `COLUNAS_X` (o espaço de
TREINO, 45 colunas), não daqui. A exclusão fica explícita mesmo assim: é
declaração de vocabulário, e um filtro que hoje não remove nada continua
correto amanhã, quando alguém acrescentar uma coluna de calendário nova."""

COLUNAS_FLUXO_CRIPTO: tuple[str, ...] = (
    "taker_buy_ratio", "trade_size_medio", "intensidade_trades", "delta_taker_ratio",
)
"""As features que **só existem em cripto**: fluxo agressor e microestrutura,
derivadas de `taker_buy_base`/`taker_buy_quote`/`trades` da Binance
(`radar.lab.fluxo`, materializadas por `feature_store_cripto_1m`).

Nenhuma tem equivalente no dado da B3 — é literalmente a informação que o
mercado de cripto oferece a mais, e que nunca entrou num espaço de seleção."""

COLUNAS_SELECAO_CRIPTO_NATIVO: tuple[str, ...] = tuple(
    c for c in COLUNAS_SELECAO_CRIPTO if c not in COLUNAS_CALENDARIO_B3
) + COLUNAS_FLUXO_CRIPTO
"""Espaço de seleção que cripto nunca teve: o técnico **menos** o calendário da
B3, **mais** o fluxo agressor.

Existe porque as 38 colunas de `COLUNAS_SELECAO_CRIPTO` são exatamente as 38 da
B3 — interseção 38, exclusivas de cripto ZERO. Enquanto isso valia, a frase
*"cripto reprova 27 de 27 em evidência"* não era evidência sobre cripto: era
evidência sobre a régua do outro mercado.

**Trocar o espaço de seleção cria FAMÍLIA NOVA** (ADR 0032): esta tupla não
substitui `COLUNAS_SELECAO_CRIPTO` em nenhuma leitura já publicada, e a família
que a usar precisa de pré-registro próprio e entra no denominador do FDR como
tentativa nova."""

COLUNAS_SELECAO_REVERSAO_RELATIVA: tuple[str, ...] = ("posto_residuo",)
"""O espaço de seleção da família `reversao_relativa_cripto_v1` — **uma coluna**.

`posto_residuo` é o posto percentual, dentro do instante, do resíduo do retorno
do ativo contra a **mediana da seção transversal do dia**. Não existe no store:
é derivado no runner, do próprio `ret_fwd_<h>`.

**Nenhuma feature técnica entra**, e isso é o desenho. Se entrassem, um resultado
positivo não distinguiria *"reversão relativa funciona"* de *"o técnico já dizia
isso"* — o mesmo argumento que mantém o candle fora do espaço do scalp.

⚠️ Uma coluna só significa que esta família **não é um modelo, é uma REGRA**.
Não há LightGBM, não há calibração, não há risco de `num_trees=1`. O
`G-P0` (o instrumento aprendeu?) perde o sentido; os outros quatro portões
continuam valendo integralmente. Isso a torna mais barata de medir e mais difícil
de sobreajustar — há um único grau de liberdade, e ele está congelado no
pré-registro.

Base medida (`scripts/medir_ativo_vs_indice_cripto.py`, 2.409 dias, 524.037 obs):
**63,5% da variância é do ativo**, não do mercado, e o posto do resíduo reverte
(`rho` de −0,039 a −0,025, janelas não sobrepostas)."""

COLUNAS_SELECAO_MICROESTRUTURA_1S: tuple[str, ...] = (
    "ofi",
    "desequilibrio_medio",
    "desvio_micropreco_bps_fim",
    "spread_bps_medio",
    "spread_bps_max",
    "atualizacoes_livro",
    "negocios",
    "desequilibrio_de_fluxo",
    "volume_comprador",
    "volume_vendedor",
    "notional_comprador",
    "notional_vendedor",
    "retorno_bps",
)
"""O espaço de seleção da família de **Scalp**, transcrito do §3 do pré-registro
congelado `PRE_REGISTRO_SCALP_CRIPTO_V3_MICROESTRUTURA.md`.

São os campos de `radar.ingestion.microestrutura_1s.Barra1s` — e **só eles**.

**Nenhuma feature de candle entra, e isso é o desenho.** Se as de candle
entrassem, um resultado positivo não distinguiria *"microestrutura funciona"* de
*"o candle já dizia isso"*, e a família perderia a razão de existir. É a única
das quatro famílias de cripto que **não** carrega a lacuna de medir com a régua
da B3 — as outras três usam as 38 técnicas, que são exatamente as 38 da B3.

⚠️ Note o que está FORA por decisão, não por esquecimento: `regime`,
`faixa_liquidez` e sessão entram como **partição** (G-P1 do Contrato de
Pensamento), nunca como coluna de seleção. Contexto que vira feature foi
justamente o defeito que fez o motor v1 aprender o *dia* em vez do ativo."""

COLUNAS_X_CRIPTO_1H: tuple[str, ...] = COLUNAS_X_CRIPTO
"""Reexport DIRETO (MESMO objeto) de `COLUNAS_X_CRIPTO` — as mesmas 43
`COLUNAS_FEATURES` + as mesmas 2 categóricas cripto = 45 colunas. O timeframe
(1d vs 1h) muda a JANELA de cada feature (em nº de barras — 1h vira 1h em
vez de 1d, automaticamente, porque as funções de indicador são agnósticas a
calendário), nunca o CONJUNTO de colunas."""

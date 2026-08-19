"""
MIE-CRIPTO — detector de regime rule-based v1 para o universo cripto (ADR
0028 §6.2 + ADR 0030, fundação MIE-CRIPTO).

PARAMETRIZAÇÃO de `radar.mie.regime` (MIE-2, B3) para o mercado cripto — as
MESMAS 4 regras, na MESMA ordem de precedência, REUSANDO os 3 helpers "core"
já provados por `tests/unit/test_mie_regime.py` (`_adx`, `_percentil_
rolling`, `_breadth_sma50`, importados diretos — mesmo padrão de reuso de
nome "privado" entre módulos-irmãos que `radar.learning.feature_store` já
usa para os mesmos 2 primeiros). O que muda, e é a razão deste módulo
existir em vez de só chamar `radar.mie.regime.detectar_regime` com outro
DataFrame:

  1. **Benchmark**: `BTCUSDT` no lugar de `^BVSP` — cripto não tem um índice
     sintético separado das "ações" (o próprio `^BVSP` do B3 não é uma ação
     negociável; `BTCUSDT` É um par negociável como qualquer outro do
     universo). Por isso o parâmetro chama-se `benchmark`, não "índice", e é
     configurável (default `TICKER_BENCHMARK_CRIPTO_PADRAO`).
  2. **Universo de breadth**: aqui é o universo cripto CARREGADO INTEIRO,
     INCLUINDO o próprio benchmark — ao contrário de `radar.mie.regime`, que
     exclui `^BVSP` do breadth via `_PADRAO_NAO_ACAO` (porque `^BVSP` não é
     um instrumento operável, não porque seja "o benchmark"). Não há
     analógo de `_PADRAO_NAO_ACAO` aqui: todo símbolo do universo cripto é
     um par `*USDT` operável, nenhum precisa ser excluído por não ser um
     ativo.
  3. **Nome de saída**: a coluna de posição do benchmark chama-se
     `btc_vs_ema50` (não `ibov_vs_ema50`) — mais claro para quem lê o
     relatório cripto, já que o nome do benchmark é parte do contrato aqui
     (poderia ser outro par no futuro).

Camada 7 (notícias) do MIE permanece DELIBERADAMENTE VAZIA nesta rodada (ADR
0029: Grupo F/notícias só entra quando a rodada R5 autorizar) — este módulo,
como `radar.mie.regime`, não tem e não pretende ter qualquer feature de
notícia; é puro preço/volume.

Núcleo 100% PURO — SEM I/O (`detectar_regime_cripto`): recebe um
`pd.DataFrame` já carregado (formato longo `COLUNAS_PADRAO` de
`radar.lab.data.carregar_cripto` — `ticker, date, Open, High, Low, Close,
Volume`) e devolve um regime por dia corrido (cripto roda 7/7 — não há
"pregão", todo dia com barra é um dia observado). Quem lê parquet é o
chamador (`radar.learning.feature_store_cripto.montar_feature_store_cripto`).

Causalidade — herdada, não reprovada aqui
------------------------------------------
Todo indicador usado (`ema` de `radar.lab.indicators`; `_adx`/`_percentil_
rolling`/`_breadth_sma50` de `radar.mie.regime`) já é causal por construção
e já provado por `tests/unit/test_mie_regime.py::TestAntiLookahead` — este
módulo não reimplementa nem reabre essa prova, só troca a fonte de dado
(cripto em vez de B3) e o benchmark (BTC em vez de IBOV). `tests/unit/
test_regime_cripto.py::TestAntiLookahead` repete o MESMO método (mutar
futuro / truncar série) especificamente sobre `detectar_regime_cripto`,
porque o benchmark e o universo de breadth mudaram — não é redundante, é a
mesma garantia sobre uma composição de dados diferente.

Insumos por dia D e as 4 regras, NESTA ordem (a primeira que casar vence) —
IDÊNTICAS a `radar.mie.regime.detectar_regime`, só com `ibov`->`btc`:
  1. `panico`        — percentil de vol >= 0,90 E breadth < 0,30
  2. `bear`           — BTC < EMA_50 E ADX_14 >= 20
  3. `bull_trending`  — BTC > EMA_50 E ADX_14 >= 20 E breadth > 0,60
  4. `bull_lateral`   — default (nenhuma das anteriores — inclui os dias em
                         que algum insumo ainda é `NaN` por falta de janela)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from radar.lab.indicators import adx, breadth_sma, ema, percentil_rolling

# ── Constantes da regra (mesmas de `radar.mie.regime`, ADR 0028 §6.2) ───────
REGIMES_MIE_CRIPTO: tuple[str, str, str, str] = ("panico", "bear", "bull_trending", "bull_lateral")

TICKER_BENCHMARK_CRIPTO_PADRAO: str = "BTCUSDT"

_JANELA_EMA_LONGA = 50
_JANELA_ADX = 14
_JANELA_BREADTH_SMA = 50
_JANELA_VOL_REALIZADA = 21
_JANELA_PERCENTIL_VOL = 252

_LIMIAR_ADX_TENDENCIA = 20.0
_LIMIAR_BREADTH_BULL = 0.60
_LIMIAR_BREADTH_PANICO = 0.30
_LIMIAR_VOL_PCT_PANICO = 0.90

COLUNAS_PRECOS_OBRIGATORIAS: tuple[str, ...] = (
    "ticker",
    "date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
)

COLUNAS_SAIDA: tuple[str, ...] = (
    "date",
    "regime_mie",
    "btc_vs_ema50",
    "adx_14",
    "breadth_sma50",
    "vol_pct_252",
)


def _exigir_precos_positivos(close: pd.Series, origem: str) -> None:
    """
    FIN-003: `(close - ema50) / ema50` e `close.pct_change()` dividem por preço.

    Preço zero ou negativo numa série de fechamento é **corrupção de dado**, não
    um caso de negócio — e o efeito dela seria `inf` viajando em silêncio até
    virar feature de regime, exatamente o padrão de falha muda que o Contrato de
    Pensamento existe para proibir. Falhar alto aqui custa uma exceção legível;
    não falhar custa um modelo treinado em `inf` sem ninguém saber.

    `NaN` NÃO é erro: as janelas móveis (`min_periods`) produzem `NaN` legítimo
    no início da série, e reprová-lo quebraria o caminho normal.
    """
    invalidos = close.dropna()
    if (invalidos <= 0).any():
        n = int((invalidos <= 0).sum())
        raise ValueError(
            f"{origem}: {n} fechamento(s) <= 0 na série de preço — divisão por "
            "preço produziria `inf` silencioso nas features de regime"
        )


def _validar_colunas(precos: pd.DataFrame) -> None:
    faltantes = [c for c in COLUNAS_PRECOS_OBRIGATORIAS if c not in precos.columns]
    if faltantes:
        raise ValueError(
            f"detectar_regime_cripto: coluna(s) obrigatória(s) ausente(s) em `precos`: {faltantes}"
        )


def detectar_regime_cripto(
    precos: pd.DataFrame, benchmark: str = TICKER_BENCHMARK_CRIPTO_PADRAO
) -> pd.DataFrame:
    """
    Detector de regime rule-based v1 do MIE-CRIPTO — SEM look-ahead, SEM
    aprendizado, as mesmas 4 regras fixas de `radar.mie.regime.
    detectar_regime`, parametrizadas para o mercado cripto (ver docstring do
    módulo).

    Parâmetros
    ----------
    `precos`: formato longo `COLUNAS_PADRAO` (`ticker, date, Open, High, Low,
    Close, Volume`) contendo OBRIGATORIAMENTE ao menos uma linha do símbolo
    `benchmark` (default `BTCUSDT`) — levanta `ValueError` claro se ausente
    (mesmo contrato de `radar.mie.regime.detectar_regime` para `^BVSP`).

    O universo de `breadth_sma50` é `precos` INTEIRO (incluindo o próprio
    `benchmark` — ver docstring do módulo, item 2, para o porquê de NÃO
    excluir o benchmark aqui, ao contrário do B3).

    Devolve uma linha por dia corrido do `benchmark`, ordenada por `date`,
    com as colunas `COLUNAS_SAIDA` — nunca uma coluna a mais nem a menos.
    """
    _validar_colunas(precos)

    bm = (
        precos.loc[precos["ticker"] == benchmark, ["date", "High", "Low", "Close"]]
        .sort_values("date")
        .drop_duplicates(subset="date", keep="last")
        .reset_index(drop=True)
    )
    if bm.empty:
        raise ValueError(f"detectar_regime_cripto: nenhuma linha de {benchmark!r} em `precos`")

    close = bm["Close"]
    ema50 = ema(close, _JANELA_EMA_LONGA)
    _exigir_precos_positivos(close, "detectar_regime_cripto")
    btc_vs_ema50 = (close - ema50) / ema50
    adx14 = adx(bm["High"], bm["Low"], close, _JANELA_ADX)

    vol21 = (
        close.pct_change()
        .rolling(window=_JANELA_VOL_REALIZADA, min_periods=_JANELA_VOL_REALIZADA)
        .std()
    )
    vol_pct = percentil_rolling(vol21, _JANELA_PERCENTIL_VOL)

    # Universo de breadth = `precos` inteiro (INCLUI o benchmark — decisão
    # documentada no docstring do módulo, item 2).
    universo = precos[["ticker", "date", "Close"]]
    breadth_por_data = breadth_sma(universo, _JANELA_BREADTH_SMA)
    breadth = bm["date"].map(breadth_por_data)

    saida = pd.DataFrame(
        {
            "date": bm["date"].to_numpy(),
            "btc_vs_ema50": btc_vs_ema50.to_numpy(),
            "adx_14": adx14.to_numpy(),
            "breadth_sma50": breadth.to_numpy(dtype=float),
            "vol_pct_252": vol_pct.to_numpy(),
        }
    )

    cond_panico = (saida["vol_pct_252"] >= _LIMIAR_VOL_PCT_PANICO) & (
        saida["breadth_sma50"] < _LIMIAR_BREADTH_PANICO
    )
    cond_bear = (saida["btc_vs_ema50"] < 0) & (saida["adx_14"] >= _LIMIAR_ADX_TENDENCIA)
    cond_bull_trending = (
        (saida["btc_vs_ema50"] > 0)
        & (saida["adx_14"] >= _LIMIAR_ADX_TENDENCIA)
        & (saida["breadth_sma50"] > _LIMIAR_BREADTH_BULL)
    )

    saida["regime_mie"] = np.select(
        [cond_panico, cond_bear, cond_bull_trending],
        ["panico", "bear", "bull_trending"],
        default="bull_lateral",
    )

    return saida[list(COLUNAS_SAIDA)]

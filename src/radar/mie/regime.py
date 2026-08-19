"""
MIE-2 — detector de regime rule-based v1 (ADR 0028, §6.2 do spec "MIE v1.0").

Função pura, SEM I/O (`detectar_regime`): recebe um `pd.DataFrame` de preços
já carregado (formato longo `COLUNAS_PADRAO` de `radar.lab.data` — `ticker,
date, Open, High, Low, Close, Volume`) e devolve um regime por pregão. Quem
lê parquet é o chamador (`scripts/mie_regime_report.py`) — mesma separação
núcleo/I/O que `radar.lab.harness`/`radar.lab.indicators` já seguem.

Por que este detector é INDEPENDENTE da `regime` do feature store
--------------------------------------------------------------------
A coluna `regime`/`regime_grupo` que `radar.learning.feature_store` faz
broadcast por data vem de `radar.historical.regime_timeline.RegimeTimeline`
— um mapa ESTÁTICO de ~14 janelas de calendário (2006-2025+), cada uma
etiquetada manualmente com um regime macro (ex. "bear_estrutural_br"
2014-01 a 2016-05). É retrospectivo e de baixa frequência: o regime de um
dia é decidido pelo MÊS em que ele cai, não pelo que o preço/breadth/vol
estavam fazendo NAQUELE dia. Este módulo é o oposto — 4 regimes RULE-BASED
recalculados a cada pregão, causais (só dado com índice `<= D`), sem
qualquer tabela de datas hardcoded. Os dois nunca são fundidos aqui: o CLI
de diagnóstico (`scripts/mie_regime_report.py`) só os COMPARA lado a lado
(concordância), nunca um sobrescreve o outro.

Causalidade (SEM look-ahead) — o mesmo padrão de `radar.lab.indicators`
--------------------------------------------------------------------------
Todo indicador usado (`ema`, `adx`, `breadth_sma`, `percentil_rolling` — os
quatro de `radar.lab.indicators`) é uma janela FECHANDO em `D`:
`.rolling`/`.ewm` com `min_periods` explícito, nunca preenchido/chutado.
`tests/unit/test_mie_regime.py::TestAntiLookahead` prova isto mutando barras
FUTURAS (depois de `D`) e conferindo que o regime em `D` não muda — mesmo
método de `tests/unit/test_lab_indicators.py`.

Insumos por dia D (ver docstring de `detectar_regime` para a fórmula exata
de cada um) e as 4 regras, nesta ordem (a primeira que casar vence):
  1. `panico`        — percentil de vol >= 0,90 E breadth < 0,30
  2. `bear`           — IBOV < EMA_50 E ADX_14 >= 20
  3. `bull_trending`  — IBOV > EMA_50 E ADX_14 >= 20 E breadth > 0,60
  4. `bull_lateral`   — default (nenhuma das anteriores — inclui os dias em
                         que algum insumo ainda é `NaN` por falta de janela,
                         já que toda comparação com `NaN` é `False`)

Decisão de implementação além do espec (documentada, não escondida): o
espec (§6.2, resumido no ADR 0028) lista "posição vs EMA_50 E EMA_21" como
insumo do IBOV, mas NENHUMA das 4 regras usa EMA_21, e a API pedida no
pacote (`date, regime_mie, ibov_vs_ema50, adx_14, breadth_sma50,
vol_pct_252`) não tem coluna para ela. EMA_21 não é computada nesta v1 —
recalculá-la sem uso nem coluna de saída seria trabalho morto; fica como
gancho óbvio para v2 caso uma regra futura precise dela.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# `adx`, `breadth_sma` e `percentil_rolling` mudaram de `radar.mie.regime` para
# `radar.lab.indicators` na S169 — são indicadores puros e estavam aqui só por
# terem nascido com o MIE-2. A consequência era `radar.learning.feature_store*`
# importar `radar.mie` para materializar coluna: a camada de DADOS dependendo do
# MOTOR, com o motor declarado congelado. Os nomes privados seguem válidos aqui
# porque `detectar_regime` os usa e `tests/unit/test_mie_regime.py` os importa
# deste módulo.
from radar.lab.indicators import adx as _adx
from radar.lab.indicators import breadth_sma as _breadth_sma50
from radar.lab.indicators import ema
from radar.lab.indicators import percentil_rolling as _percentil_rolling

# ── Constantes da regra (ADR 0028 §6.2) ────────────────────────────────────
REGIMES_MIE: tuple[str, str, str, str] = ("panico", "bear", "bull_trending", "bull_lateral")

TICKER_INDICE: str = "^BVSP"

_JANELA_EMA_LONGA = 50
_JANELA_ADX = 14
_JANELA_BREADTH_SMA = 50
_JANELA_VOL_REALIZADA = 21
_JANELA_PERCENTIL_VOL = 252

_LIMIAR_ADX_TENDENCIA = 20.0
_LIMIAR_BREADTH_BULL = 0.60
_LIMIAR_BREADTH_PANICO = 0.30
_LIMIAR_VOL_PCT_PANICO = 0.90

# Mesmo padrão de exclusão de não-ação de `radar.lab.data._PADRAO_NAO_ACAO`
# (símbolo de índice/câmbio contém "^" ou "=") — reimplementado aqui (não
# importado) porque é uma constante de UMA linha e este módulo não deve
# depender de `radar.lab.data` (que é I/O; este é núcleo puro).
_PADRAO_NAO_ACAO = r"[\^=]"

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
    "ibov_vs_ema50",
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
            f"detectar_regime: coluna(s) obrigatória(s) ausente(s) em `precos`: {faltantes}"
        )


def detectar_regime(precos: pd.DataFrame) -> pd.DataFrame:
    """
    Detector de regime rule-based v1 do MIE (ADR 0028 §6.2) — SEM look-ahead,
    SEM aprendizado, só as 4 regras fixas do cabeçalho do módulo.

    Parâmetros
    ----------
    `precos`: formato longo `COLUNAS_PADRAO` (`ticker, date, Open, High, Low,
    Close, Volume`) contendo OBRIGATORIAMENTE ao menos uma linha por pregão
    de `^BVSP` (`TICKER_INDICE`) — o índice é excluído por
    `radar.lab.data.carregar_diario_b3_fresco` de propósito (não é ação),
    então quem monta este DataFrame precisa incluí-lo separadamente (ver
    `scripts/mie_regime_report.py::_carregar_precos_com_indice`, que segue o
    mesmo padrão documentado de `scripts/tese_morte_report.py::carregar_ibov`).
    Os demais tickers (o "resto do parquet") formam o universo de breadth —
    qualquer linha cujo `ticker` contenha `^`/`=` (índice/câmbio) é excluída
    dessa conta, mesma convenção de `radar.lab.data._PADRAO_NAO_ACAO`.

    Insumos calculados por dia D (todos causais):
      - `ibov_vs_ema50 = (Close_ibov − EMA_50(Close_ibov)) / EMA_50(...)` em
        D — sinal decide as regras 2/3 (`< 0` = abaixo, `> 0` = acima;
        empate exato não ativa nem uma nem outra, cai no default).
      - `adx_14` do índice (`_adx`, ver docstring acima).
      - `breadth_sma50` = `_breadth_sma50` sobre o universo de ações.
      - `vol_pct_252` = percentil rolling 252d (`_percentil_rolling`) da vol
        realizada 21d do IBOV (`Close.pct_change().rolling(21).std()`).

    Regras, NESTA ordem — a primeira que casar vence (`np.select` preserva
    a ordem da lista de condições):
      1. `panico`: `vol_pct_252 >= 0,90` E `breadth_sma50 < 0,30`
      2. `bear`: `ibov_vs_ema50 < 0` E `adx_14 >= 20`
      3. `bull_trending`: `ibov_vs_ema50 > 0` E `adx_14 >= 20` E
         `breadth_sma50 > 0,60`
      4. `bull_lateral`: default — inclui os dias em que algum insumo ainda
         é `NaN` (warm-up: toda comparação com `NaN` é `False`, então nunca
         propaga `NaN` para `regime_mie`, só cai no default documentado).

    Devolve uma linha por pregão do ÍNDICE (`^BVSP`), ordenada por `date`,
    com as colunas `COLUNAS_SAIDA` — nunca uma coluna a mais nem a menos.
    """
    _validar_colunas(precos)

    ibov = (
        precos.loc[precos["ticker"] == TICKER_INDICE, ["date", "High", "Low", "Close"]]
        .sort_values("date")
        .drop_duplicates(subset="date", keep="last")
        .reset_index(drop=True)
    )
    if ibov.empty:
        raise ValueError(f"detectar_regime: nenhuma linha de {TICKER_INDICE!r} em `precos`")

    close = ibov["Close"]
    _exigir_precos_positivos(close, "detectar_regime")
    ema50 = ema(close, _JANELA_EMA_LONGA)
    ibov_vs_ema50 = (close - ema50) / ema50
    adx14 = _adx(ibov["High"], ibov["Low"], close, _JANELA_ADX)

    vol21 = (
        close.pct_change()
        .rolling(window=_JANELA_VOL_REALIZADA, min_periods=_JANELA_VOL_REALIZADA)
        .std()
    )
    vol_pct = _percentil_rolling(vol21, _JANELA_PERCENTIL_VOL)

    eh_nao_acao = precos["ticker"].astype(str).str.contains(_PADRAO_NAO_ACAO, regex=True)
    equities = precos.loc[~eh_nao_acao, ["ticker", "date", "Close"]]
    breadth_por_data = _breadth_sma50(equities, _JANELA_BREADTH_SMA)
    breadth = ibov["date"].map(breadth_por_data)

    saida = pd.DataFrame(
        {
            "date": ibov["date"].to_numpy(),
            "ibov_vs_ema50": ibov_vs_ema50.to_numpy(),
            "adx_14": adx14.to_numpy(),
            "breadth_sma50": breadth.to_numpy(dtype=float),
            "vol_pct_252": vol_pct.to_numpy(),
        }
    )

    cond_panico = (saida["vol_pct_252"] >= _LIMIAR_VOL_PCT_PANICO) & (
        saida["breadth_sma50"] < _LIMIAR_BREADTH_PANICO
    )
    cond_bear = (saida["ibov_vs_ema50"] < 0) & (saida["adx_14"] >= _LIMIAR_ADX_TENDENCIA)
    cond_bull_trending = (
        (saida["ibov_vs_ema50"] > 0)
        & (saida["adx_14"] >= _LIMIAR_ADX_TENDENCIA)
        & (saida["breadth_sma50"] > _LIMIAR_BREADTH_BULL)
    )

    saida["regime_mie"] = np.select(
        [cond_panico, cond_bear, cond_bull_trending],
        ["panico", "bear", "bull_trending"],
        default="bull_lateral",
    )

    return saida[list(COLUNAS_SAIDA)]

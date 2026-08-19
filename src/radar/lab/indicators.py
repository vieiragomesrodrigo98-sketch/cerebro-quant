"""
Indicadores técnicos puros do laboratório — CAUSAIS por construção.

Todas as funções recebem `pd.Series`/`pd.DataFrame` e devolvem uma `pd.Series`
alinhada ao índice de entrada. "Causal" aqui tem um significado preciso e não
negociável: **o valor em `t` usa somente dado com índice `<= t`**. Mudar uma
barra posterior a `t` nunca pode mudar o valor calculado em `t` — é exatamente
o que `tests/unit/test_lab_indicators.py` verifica truncando a série em vários
pontos e comparando contra o cálculo na série completa.

Onde a janela ainda não está completa (as primeiras `n-1` barras de uma média
de `n`, por exemplo), o valor é `NaN` — nunca preenchido, interpolado ou
"chutado" com o primeiro valor disponível. Preencher esconderia justamente a
falta de amostra que a Sessão 142 (ADR 0020) já mostrou ser fatal quando
ignorada silenciosamente.

Fórmulas mantidas consistentes com o motor de produção (mesmo sem reusar seu
código — o laboratório é independente, ver ADR 0021):
  - RSI de Wilder — mesma convenção de `radar.engines.technical_context._rsi`
    (média exponencial de ganhos/perdas, alpha=1/n).
  - ATR de Wilder — true range com fechamento anterior, suavizado com o mesmo
    alpha=1/n do RSI (Wilder usou a mesma constante de suavização nos dois).
  - VWAP por sessão — preço típico (H+L+C)/3, acumulado desde a abertura de
    CADA sessão (o acumulador reseta a cada troca de dia do índice).

Duas categorias merecem cuidado extra com look-ahead:
  - `maxima_n`/`minima_n` EXCLUEM a barra `t` (via `shift(1)`) — são usadas
    para nível de rompimento, e "romper a própria máxima" seria trapacear.
  - `razao_volume` também exclui `t` da média — o volume da barra atual não
    pode inflar a própria média de referência.

`INDICADORES` é o registro nome→construtor que o harness (Sessão A2) usa para
montar colunas nomeadas (ex. "rsi_14" = `INDICADORES["rsi"](close, 14)`) a
partir da spec declarativa (`radar.lab.spec.EstrategiaSpec`).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta

import numpy as np
import pandas as pd

from radar.utils.b3_calendar import is_b3_trading_day


def _validar_janela(n: int) -> None:
    if n <= 0:
        raise ValueError(f"janela precisa ser > 0, recebeu {n!r}")


def sma(close: pd.Series, n: int) -> pd.Series:
    """Média móvel simples de `n` barras. `NaN` enquanto a janela não fecha."""
    _validar_janela(n)
    return close.rolling(window=n, min_periods=n).mean()


def ema(close: pd.Series, n: int) -> pd.Series:
    """
    Média móvel exponencial de `n` barras (`adjust=False` — recursiva: o valor
    em `t` depende só de `close[t]` e do valor acumulado até `t-1`, o que já a
    torna causal por construção). `min_periods=n` para não devolver valor antes
    de `n` observações terem sido vistas.
    """
    _validar_janela(n)
    return close.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int) -> pd.Series:
    """
    RSI de Wilder: ganhos e perdas suavizados por média exponencial com
    `alpha=1/n` (equivalente a `com=n-1`) — mesmo algoritmo de
    `radar.engines.technical_context._rsi`, não reusado (o lab é
    independente), mas consistente para efeito de comparação.
    """
    _validar_janela(n)
    delta = close.diff()
    ganho = delta.clip(lower=0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    perda = (-delta.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = ganho / perda
    return 100.0 - (100.0 / (1.0 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    """
    ATR de Wilder: true range (contra o fechamento ANTERIOR) suavizado por
    média exponencial com `alpha=1/n` — a mesma constante de suavização do
    `rsi` acima, a convenção original de Wilder para os dois indicadores.
    """
    _validar_janela(n)
    close_anterior = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - close_anterior).abs(),
            (low - close_anterior).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def vwap_sessao(df: pd.DataFrame) -> pd.Series:
    """
    VWAP acumulado POR SESSÃO — o acumulador reinicia a cada troca de dia do
    índice (`df.index` precisa ser um `DatetimeIndex`, tz-aware ou não).

    Preço típico `(High + Low + Close) / 3`, ponderado por volume, acumulado
    (`cumsum`) desde a primeira barra de cada sessão — a soma cumulativa já é
    causal por definição (só soma barras com índice `<= t`).
    """
    # sem ajuste de provento/split aqui, DE PROPÓSITO: o VWAP é intra-sessão
    # (o acumulador reseta a cada dia) e eventos corporativos acontecem ENTRE
    # sessões — dentro da mesma sessão, preço bruto e ajustado têm a mesma
    # forma. A escolha de série ajustada vs bruta é do harness (ADR 0021:
    # B3 diário SEMPRE de precos_adj.parquet; 5m usa price_bars).
    for coluna in ("High", "Low", "Close", "Volume"):
        if coluna not in df.columns:
            raise ValueError(f"vwap_sessao: coluna obrigatória ausente: {coluna!r}")

    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    valor = typical * df["Volume"]
    sessao = df.index.normalize()  # trunca para a data — agrupador de sessão

    valor_acumulado = valor.groupby(sessao).cumsum()
    volume_acumulado = df["Volume"].groupby(sessao).cumsum()
    return valor_acumulado / volume_acumulado


def distancia_vwap_sessao(df: pd.DataFrame) -> pd.Series:
    """
    Distância percentual do close em relação ao VWAP da MESMA sessão:
    `(close - vwap_sessao) / vwap_sessao` — positivo = preço acima do VWAP.

    Reusa `vwap_sessao` internamente (mesma causalidade dela: acumulador que
    reseta por dia, só soma `<= t`). Não é uma janela nova, é uma razão
    PONTUAL na mesma barra — adicionada na Sessão A3 (ADR 0021) para expressar
    como `Cond` de grade a regra `price_vs_vwap >= limiar` de
    `radar.engines.technical_trigger` (ver `radar.lab.catalogo`).
    """
    vwap = vwap_sessao(df)
    return (df["Close"] - vwap) / vwap


def hora_do_dia(df: pd.DataFrame) -> pd.Series:
    """
    Hora UTC (0-23) do timestamp de cada barra — indicador TEMPORAL causal por
    definição (o valor em `t` é uma propriedade do próprio `t`, nunca do
    futuro). Base das specs de SAZONALIDADE do catálogo v3
    (CATALOGO_V3_PESQUISA01: janela 21h-23h UTC do BTC, Monday Asia Open —
    Quantpedia/PMC10015199). Requer `df.index` DatetimeIndex (mesmo contrato
    de `vwap_sessao`); em barra DIÁRIA a hora é constante (00) — a spec que
    usar isto só faz sentido em timeframe intradiário.
    """
    idx = pd.DatetimeIndex(df.index)
    return pd.Series(idx.hour.astype(float), index=df.index)


def dia_da_semana(df: pd.DataFrame) -> pd.Series:
    """
    Dia da semana (0=segunda … 6=domingo) do timestamp de cada barra — causal
    por definição (propriedade do próprio `t`). Base do efeito fim-de-semana
    em altcoins (catálogo v3). Mesmo contrato de índice de `hora_do_dia`.
    """
    idx = pd.DatetimeIndex(df.index)
    return pd.Series(idx.dayofweek.astype(float), index=df.index)


def dia_util_do_mes(df: pd.DataFrame) -> pd.Series:
    """
    Posição ORDINAL (1-based) de cada barra dentro do mês, contando só as
    barras que EXISTEM NA SÉRIE do símbolo — 1 = primeira barra do mês na
    série, 2 = segunda, e assim por diante. Mesmo contrato de `hora_do_dia`/
    `dia_da_semana` (`df.index` precisa ser `DatetimeIndex`).

    Causal por construção: o valor em `t` é uma contagem cumulativa
    (`cumcount`) de barras do MESMO mês vistas até `t` (inclusive) — depende
    só de barras com índice `<= t`; truncar a série depois de `t` nunca muda
    o valor em `t` (é olhar para TRÁS, o oposto de `dias_uteis_ate_fim_mes`
    abaixo, que olha para a FRENTE via calendário, nunca via posição na
    série).

    Não é "dia do mês" (1-31) nem uma contagem de dias úteis de CALENDÁRIO —
    é a posição da barra na série já filtrada (fins de semana/feriados já
    ausentes no dado diário B3 típico). Uma série com barra faltando (gap de
    coleta) reflete a LACUNA da série, não o calendário — efeito CATALOGO_V4_B3_01
    (virada-de-mês, `radar.lab.catalogo::v4b1_virada_de_mes`) usa
    `dias_uteis_ate_fim_mes` para o lado "quantos dias faltam" precisamente
    porque esse lado exige o calendário, não a série (ver docstring dela).
    """
    idx = pd.DatetimeIndex(df.index)
    periodo = pd.Series(idx.to_period("M"))
    contagem = periodo.groupby(periodo).cumcount().to_numpy() + 1
    return pd.Series(contagem.astype(float), index=df.index)


def _fim_de_mes_calendario(d: date) -> date:
    """Último dia CALENDÁRIO (não necessariamente útil) do mês de `d`."""
    if d.month == 12:
        return date(d.year, 12, 31)
    return date(d.year, d.month + 1, 1) - timedelta(days=1)


def _dias_uteis_restantes_no_mes(d: date) -> int:
    """
    Quantos dias úteis B3 (calendário — `radar.utils.b3_calendar.
    is_b3_trading_day`) existem ESTRITAMENTE DEPOIS de `d` até o fim do mês
    de `d` (inclusive) — `0` quando `d` já É o último dia útil do mês.
    """
    fim = _fim_de_mes_calendario(d)
    cursor = d + timedelta(days=1)
    total = 0
    while cursor <= fim:
        if is_b3_trading_day(cursor):
            total += 1
        cursor += timedelta(days=1)
    return total


def dias_uteis_ate_fim_mes(df: pd.DataFrame) -> pd.Series:
    """
    Dias úteis B3 de CALENDÁRIO que faltam até o fim do mês, a partir da data
    de cada barra — `0` na barra que cai no último dia útil do mês, `1` na
    véspera, e assim por diante. Mesmo contrato de `hora_do_dia`/
    `dia_da_semana`/`dia_util_do_mes` (`df.index` precisa ser `DatetimeIndex`).

    ANTI-LOOKAHEAD (o cuidado desta função, documentado porque é sutil):
    "quantos dias faltam até o fim do mês" só é conhecível olhando para a
    FRENTE — mas o calendário B3 (feriados fixos e móveis já tabulados até
    2028 em `radar.utils.b3_calendar`, público e conhecido de antemão)
    permite calcular essa contagem SEM consultar a série de barras, só a
    data da própria barra e um calendário que já existia antes de qualquer
    dado ser coletado. A alternativa ingênua — contar a partir da ÚLTIMA
    ocorrência do mês NA SÉRIE (o que `dia_util_do_mes` faz, de propósito,
    olhando para TRÁS) — só saberia qual foi "a última barra do mês" depois
    que a barra seguinte (já de outro mês) aparecesse: nesse ponto, o valor
    atribuído à barra atual já teria sido decidido usando uma informação que
    só existe no futuro da própria série. Como o efeito virada-de-mês
    (`radar.lab.catalogo::v4b1_virada_de_mes`, CATALOGO_V4_B3_01) é
    justamente sobre ANTECIPAR a proximidade do fim do mês para decidir a
    entrada — nunca confirmá-la depois do fato —, esta função usa só
    `is_b3_trading_day` (calendário), nunca a posição da barra na série.

    Cada valor único de `df.index.date` é resolvido uma vez só (dicionário de
    memoização local) — o calendário nunca varre mais que ~31 dias por data
    distinta, então o custo é desprezível mesmo em séries de 20 anos.
    """
    idx = pd.DatetimeIndex(df.index)
    cache: dict[date, int] = {}
    valores = np.empty(len(idx), dtype=float)
    for i, timestamp in enumerate(idx):
        dia = timestamp.date()
        if dia not in cache:
            cache[dia] = _dias_uteis_restantes_no_mes(dia)
        valores[i] = cache[dia]
    return pd.Series(valores, index=df.index)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On Balance Volume: soma cumulativa do volume assinado pelo sentido do
    retorno diário — `+volume[t]` se `close[t] > close[t-1]`, `-volume[t]` se
    menor, `0` se igual. A primeira barra não tem `close[t-1]`: recebe sinal 0
    (o acumulador começa sem direção, não `NaN` — é a convenção padrão do OBV).
    """
    direcao = np.sign(close.diff()).fillna(0.0)
    return (direcao * volume).cumsum()


def maxima_n(high: pd.Series, n: int) -> pd.Series:
    """
    Máxima das `n` barras ANTERIORES a `t`, EXCLUINDO `t` (`shift(1)` antes do
    `rolling`). Uso típico: nível de rompimento — comparar `high[t]` contra a
    máxima anterior sem contar a própria barra evita o look-ahead óbvio de
    "romper a própria máxima".
    """
    _validar_janela(n)
    return high.shift(1).rolling(window=n, min_periods=n).max()


def minima_n(low: pd.Series, n: int) -> pd.Series:
    """Mínima das `n` barras ANTERIORES a `t`, EXCLUINDO `t` — ver `maxima_n`."""
    _validar_janela(n)
    return low.shift(1).rolling(window=n, min_periods=n).min()


def razao_volume(volume: pd.Series, n: int) -> pd.Series:
    """
    `volume[t] / média das `n` barras ANTERIORES` (excluindo `t` — mesma lógica
    de exclusão de `maxima_n`/`minima_n`: o volume da própria barra não pode
    inflar a média que a compara).
    """
    _validar_janela(n)
    media_anterior = volume.shift(1).rolling(window=n, min_periods=n).mean()
    return volume / media_anterior


def largura_banda(close: pd.Series, n: int, k: float) -> pd.Series:
    """
    Largura de Bollinger relativa: `(2·k·desvio) / média` — usada para detectar
    squeeze (compressão de volatilidade). `desvio` e `média` compartilham a
    mesma janela causal de `n` barras (`rolling`, `min_periods=n`).
    """
    _validar_janela(n)
    media = close.rolling(window=n, min_periods=n).mean()
    desvio = close.rolling(window=n, min_periods=n).std()
    return (2.0 * k * desvio) / media


def razao_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, n_curto: int, n_longo: int
) -> pd.Series:
    """
    Razão de volatilidades (Sessão A7, `radar.lab.catalogo::p9c59_contracao_atr`)
    — a mesma medida do "VCR" do Capítulo 59 do Manual (`σ_curta/σ_longa`),
    aqui expressa via ATR de Wilder (já causal, ver `atr` acima) em vez de
    desvio-padrão de retornos: `ATR(n_curto) / ATR(n_longo)`.

    `< 1` → volatilidade recente ABAIXO da de regime (contração/squeeze,
    Capítulo 57/59); `> 1` → expansão. Herda a causalidade e o `NaN` de
    `atr()` sem preenchimento — enquanto qualquer um dos dois ATRs não tem
    janela completa, o resultado é `NaN`.
    """
    _validar_janela(n_curto)
    _validar_janela(n_longo)
    atr_curto = atr(high, low, close, n_curto)
    atr_longo = atr(high, low, close, n_longo)
    return atr_curto / atr_longo


def macd_hist(close: pd.Series, n_rapida: int, n_lenta: int, n_sinal: int) -> pd.Series:
    """
    Histograma do MACD (Gerald Appel, `radar.learning.feature_store` — MIE-1,
    ADR 0028 §5): `(EMA_rápida − EMA_lenta) − EMA_sinal`, onde `EMA_sinal` é a
    EMA de `n_sinal` períodos sobre a PRÓPRIA linha MACD (`EMA_rápida −
    EMA_lenta`). Devolvido CRU (sem normalizar pelo preço) — normalizar por
    `close` para comparabilidade entre tickers é decisão do CHAMADOR (ver
    `radar.learning.feature_store`, coluna `macd_hist`, que divide o
    resultado desta função pelo close).

    Reusa `ema()` acima (já causal, `adjust=False`/`min_periods=n`) 3 vezes em
    cascata. A segunda EMA (sobre a linha MACD, que carrega um prefixo `NaN`
    do warm-up de `EMA_lenta`) continua causal: `pandas.Series.ewm` conta só
    observações NÃO-nulas para `min_periods` e só começa a recursão a partir
    da primeira barra válida (comportamento verificado empiricamente antes de
    confiar nele aqui — não é óbvio à primeira vista) — nenhuma barra futura
    é usada em nenhuma das 3 EMAs.
    """
    _validar_janela(n_rapida)
    _validar_janela(n_lenta)
    _validar_janela(n_sinal)
    linha_macd = ema(close, n_rapida) - ema(close, n_lenta)
    sinal = ema(linha_macd, n_sinal)
    return linha_macd - sinal


def estocastico_k(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    """
    %K do Oscilador Estocástico (George Lane, `radar.learning.feature_store`
    — MIE-1, ADR 0028 §5): `100 · (close − mínima_n) / (máxima_n − mínima_n)`,
    janela de `n` barras TERMINANDO em `t` (INCLUI a própria barra `t` — ao
    contrário de `maxima_n`/`minima_n` acima, que EXCLUEM `t` de propósito
    para medir nível de ROMPIMENTO; aqui o objetivo é outro — "onde o close de
    hoje está dentro do range recente" — que por definição inclui o próprio
    range de hoje. Reusar `maxima_n`/`minima_n` mudaria a semântica do
    indicador; não é um esquecimento, é uma decisão deliberada).

    `NaN` quando a janela não fecha (`min_periods=n`) OU quando
    `máxima_n == mínima_n` (range degenerado — denominador zero guardado
    explicitamente, nunca `inf` silencioso, mesma convenção de `radar.mie.
    regime._adx` para `+DI+-DI == 0`).
    """
    _validar_janela(n)
    maxima = high.rolling(window=n, min_periods=n).max()
    minima = low.rolling(window=n, min_periods=n).min()
    amplitude = (maxima - minima).replace(0.0, np.nan)
    return 100.0 * (close - minima) / amplitude


def retorno_n(close: pd.Series, n: int) -> pd.Series:
    """
    Retorno acumulado das últimas `n` barras: `close[t]/close[t-n] − 1` —
    momentum do "mastro" da bandeira (Sessão F8-S4,
    `radar.lab.catalogo::p9c56_bandeira`, Capítulo 56). `NaN` enquanto
    `close[t-n]` não existir (as `n` primeiras barras da série).

    CAUSALIDADE: `close.shift(n)` só usa índice `<= t-n` (`< t`), e a razão
    é aritmética elementar na própria barra `t` — o mesmo argumento de
    `maxima_n`/`minima_n` acima, com `shift` em vez de `rolling`.
    """
    _validar_janela(n)
    return close / close.shift(n) - 1.0


def distancia_maxima_n(close: pd.Series, high: pd.Series, n: int) -> pd.Series:
    """
    Distância percentual do fechamento em relação à máxima das `n` barras
    ANTERIORES a `t` (Sessão F8-S4, `radar.lab.catalogo::p9c56_bandeira`,
    Capítulo 56) — reusa `maxima_n` acima, que já EXCLUI `t` via `shift(1)`:
    `(close[t] − maxima_n(high,n)[t]) / maxima_n(high,n)[t]`.

    Negativo quando `close[t]` está abaixo da máxima anterior (o caso comum
    — "quão fundo dentro da bandeira"); zero ou positivo só quando
    `close[t]` rompe a própria máxima anterior.

    CAUSALIDADE: herdada de `maxima_n` (já causal) — a razão é aritmética
    elementar sobre um valor `<= t-1` e `close[t]` (conhecido em `t`), o
    mesmo argumento que `distancia_vwap_sessao` já usa para reaproveitar
    `vwap_sessao` sem reabrir a prova de causalidade.
    """
    _validar_janela(n)
    maxima = maxima_n(high, n)
    return (close - maxima) / maxima


# ── Pivôs causais (topos/fundos confirmados) ──────────────────────────────────
# Fundação dos detectores de padrões gráficos (Sessão F8-S1). Fundo duplo, OCO,
# triângulo e bandeira vêm em S3/S4 e NÃO são implementados aqui — este módulo
# só expõe a PRIMITIVA (pivô confirmado, causal). Definição canônica:
# `docs/manual_trader_ai/parte_09_analise_tecnica/cap_54_resistencias.md` §3.1
# ("swing high de ordem k"; "níveis disponíveis para a decisão em t = topos
# confirmados até t−k−1") e `cap_56_pullbacks.md` §3.1 (mesma armadilha, do
# lado do fundo/impulso).
#
# Definição (pivô de TOPO, folga `n_conf`): `high[i]` é pivô BRUTO em `i` se
# for ESTRITAMENTE maior que as `n_conf` barras de CADA lado —
# `high[i] > max(high[i-n_conf : i])` E `high[i] > max(high[i+1 : i+n_conf+1])`.
# Pivô de FUNDO é o espelho: `low[i]` estritamente MENOR que as `n_conf` barras
# de cada lado. EMPATE EXATO (float) com QUALQUER barra da janela — esquerda
# OU direita — NÃO confirma pivô: as duas comparações são sempre `>`/`<`,
# nunca `>=`/`<=`. Isto é deliberado (não um artefato de ponto flutuante):
# uma barra empatada com o "topo" não domina os dois lados, então não é uma
# única máxima local — é o próprio conceito de pivô que exige.
#
# Causalidade: um pivô bruto em `i` só é CONHECIDO (existe para decisão) em
# `t = i + n_conf` — antes disso, as `n_conf` barras à direita que provam que
# `i` é topo/fundo ainda não existiam na decisão. Regra operacional (mesma do
# Manual): "níveis disponíveis para a decisão em t = pivôs confirmados até
# t − n_conf − 1" (ou, equivalentemente, `topo_idade(t) >= n_conf` sempre —
# ver `topo_idade`/`fundo_idade` abaixo: o pivô "nasce velho").
#
# Nota sobre a fórmula do briefing — decisão além dele, documentada aqui: a
# formulação `high[t-n_conf] == max(high[t-2*n_conf : t])` (rolling de
# `2*n_conf+1` terminando em `t`) descreve o MESMO invariante causal (o
# rolling termina em `t`, logo só usa dado `<= t`), mas uma checagem de
# IGUALDADE única contra o máximo do bloco inteiro não distingue "sou o único
# máximo do bloco" de "empato com outra barra do bloco (à esquerda OU à
# direita) e a `max()` bate por coincidência". A implementação abaixo usa dois
# `rolling` SEPARADOS (esquerda e direita de `i`) com comparação ESTRITA dos
# dois lados — estritamente mais forte que exigir estrito só à direita, e é o
# que de fato implementa "empate exato não gera pivô" em qualquer lado.


def _validar_k(k: int) -> None:
    if k <= 0:
        raise ValueError(f"k precisa ser >= 1, recebeu {k!r}")


def _pivos(serie: pd.Series, n_conf: int, modo: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Núcleo vetorizado dos pivôs causais — comum às 4 famílias `topo_*`/`fundo_*`.

    `modo` é `"topo"` (pivô = ESTRITAMENTE maior que os `n_conf` vizinhos de
    CADA lado) ou `"fundo"` (ESTRITAMENTE menor). Devolve `(posicoes,
    contagem)`:
      - `posicoes`: índices posicionais, em ORDEM CRESCENTE, de cada pivô
        BRUTO da série inteira (`posicoes[j]` = índice `i` do j-ésimo pivô).
      - `contagem`: array do mesmo tamanho de `serie`; `contagem[t]` é quantos
        pivôs já estão CONFIRMADOS na decisão em `t` — quantos `i` de
        `posicoes` satisfazem `i + n_conf <= t`. É o "cumsum de eventos" do
        briefing, já deslocado do tempo de OCORRÊNCIA para o tempo de DECISÃO
        (a barreira causal mora exatamente nesse deslocamento).

    Totalmente vetorizado: dois `rolling` de `n_conf` barras (esquerda e
    direita de cada `i`, ambos com `shift` para excluir a própria barra `i` —
    mesmo truque de `maxima_n`/`minima_n`) mais um `cumsum` — nenhum `for`
    Python sobre as barras. O rolling da direita usa a série revertida
    (espelho do da esquerda) e o resultado é revertido de volta.
    """
    _validar_janela(n_conf)
    if modo not in ("topo", "fundo"):
        raise ValueError(f"modo precisa ser 'topo' ou 'fundo', recebeu {modo!r}")

    valores = serie.to_numpy(dtype=float)
    n = len(valores)
    revertida = pd.Series(valores[::-1])

    if modo == "topo":
        esquerda = serie.shift(1).rolling(window=n_conf, min_periods=n_conf).max().to_numpy()
        direita = (
            revertida.shift(1).rolling(window=n_conf, min_periods=n_conf).max().to_numpy()[::-1]
        )
        bruto = (valores > esquerda) & (valores > direita)
    else:
        esquerda = serie.shift(1).rolling(window=n_conf, min_periods=n_conf).min().to_numpy()
        direita = (
            revertida.shift(1).rolling(window=n_conf, min_periods=n_conf).min().to_numpy()[::-1]
        )
        bruto = (valores < esquerda) & (valores < direita)

    posicoes = np.flatnonzero(bruto)
    contagem_bruta = np.cumsum(bruto)
    contagem = np.zeros(n, dtype=np.int64)
    if n > n_conf:
        contagem[n_conf:] = contagem_bruta[: n - n_conf]
    return posicoes, contagem


def _pivo_kesimo(serie: pd.Series, k: int, n_conf: int, modo: str, campo: str) -> pd.Series:
    """
    `campo="preco"` → preço do k-ésimo pivô mais recente CONFIRMADO em cada
    `t` (`k=1` é o mais recente, `k=2` o penúltimo, ...); `NaN` enquanto não
    existirem `k` pivôs confirmados. `campo="idade"` → `t - posicao_do_pivo`
    (sempre `>= n_conf`, ver docstring de `_pivos` acima).
    """
    _validar_k(k)
    posicoes, contagem = _pivos(serie, n_conf, modo)
    valores = serie.to_numpy(dtype=float)
    n = len(valores)

    resultado = np.full(n, np.nan, dtype=float)
    tem_kesimo = contagem >= k
    if tem_kesimo.any():
        indices_absolutos = posicoes[contagem[tem_kesimo] - k]
        if campo == "preco":
            resultado[tem_kesimo] = valores[indices_absolutos]
        else:  # "idade"
            t_idx = np.flatnonzero(tem_kesimo)
            resultado[tem_kesimo] = (t_idx - indices_absolutos).astype(float)

    return pd.Series(resultado, index=serie.index)


def topo_preco(high: pd.Series, k: int, n_conf: int) -> pd.Series:
    """
    Preço do k-ésimo topo pivô mais recente CONFIRMADO em cada `t` (folga
    `n_conf` de cada lado — ver definição no cabeçalho da seção "Pivôs
    causais" acima). `NaN` enquanto não existirem `k` topos confirmados.
    `k=1` é o mais recente; um novo topo confirmado empurra o anterior para
    `k=2`, e assim por diante.
    """
    return _pivo_kesimo(high, k, n_conf, modo="topo", campo="preco")


def topo_idade(high: pd.Series, k: int, n_conf: int) -> pd.Series:
    """
    Idade (em barras, `t - i`) do k-ésimo topo pivô mais recente CONFIRMADO em
    `t`. Invariante: SEMPRE `>= n_conf` — um pivô só é conhecido a partir de
    `n_conf` barras depois de ocorrer, então, no instante exato em que é
    confirmado, sua idade já é `n_conf` ("o pivô nasce velho").
    """
    return _pivo_kesimo(high, k, n_conf, modo="topo", campo="idade")


def fundo_preco(low: pd.Series, k: int, n_conf: int) -> pd.Series:
    """Preço do k-ésimo fundo pivô mais recente CONFIRMADO em cada `t` — simétrico a `topo_preco` (ver docstring lá)."""
    return _pivo_kesimo(low, k, n_conf, modo="fundo", campo="preco")


def fundo_idade(low: pd.Series, k: int, n_conf: int) -> pd.Series:
    """Idade do k-ésimo fundo pivô mais recente CONFIRMADO em `t` — simétrico a `topo_idade` (mesmo invariante `>= n_conf`)."""
    return _pivo_kesimo(low, k, n_conf, modo="fundo", campo="idade")


# ── Registro nome → construtor ────────────────────────────────────────────────
# O harness (Sessão A2) monta colunas nomeadas (ex. "rsi_14") a partir daqui,
# guiado pelos campos referenciados em `EstrategiaSpec.entrada`.
INDICADORES: dict[str, Callable[..., pd.Series]] = {
    "sma": sma,
    "ema": ema,
    "rsi": rsi,
    "atr": atr,
    "vwap_sessao": vwap_sessao,
    "distancia_vwap_sessao": distancia_vwap_sessao,
    "hora_do_dia": hora_do_dia,
    "dia_da_semana": dia_da_semana,
    "dia_util_do_mes": dia_util_do_mes,
    "dias_uteis_ate_fim_mes": dias_uteis_ate_fim_mes,
    "obv": obv,
    "maxima_n": maxima_n,
    "minima_n": minima_n,
    "razao_volume": razao_volume,
    "largura_banda": largura_banda,
    "razao_atr": razao_atr,
    "macd_hist": macd_hist,
    "estocastico_k": estocastico_k,
    "retorno_n": retorno_n,
    "distancia_maxima_n": distancia_maxima_n,
    "topo_preco": topo_preco,
    "topo_idade": topo_idade,
    "fundo_preco": fundo_preco,
    "fundo_idade": fundo_idade,
}


# ── Indicadores que vieram de `radar.mie.regime` na S169 ────────────────────
#
# Os três abaixo são indicadores PUROS — ADX de Wilder, percentil rolling e
# breadth de mercado — e moravam no pacote do motor por acidente histórico
# (nasceram junto com o MIE-2). A consequência era que `radar.learning.
# feature_store*` importava `radar.mie` para materializar coluna: a camada de
# DADOS dependendo do MOTOR, com o motor declarado congelado.
#
# `_breadth_sma50` virou `breadth_sma`: a janela sempre foi parâmetro, e o
# "50" no nome dizia o contrário. Os nomes antigos seguem disponíveis em
# `radar.mie.regime` como alias, porque `detectar_regime` e a bateria de
# testes do MIE-2 os usam.

def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    """
    ADX de Wilder (força de tendência, 0-100), CAUSAL — mesma convenção de
    suavização (`alpha=1/n`, `adjust=False`, `min_periods=n`) que `atr`/`rsi`
    de `radar.lab.indicators` já usam para as duas médias exponenciais em
    cascata que o ADX exige.

    Fórmula (Wilder, 1978), em 4 passos:
      1. `+DM`/`-DM` (movimento direcional): `+DM[t] = high[t]-high[t-1]` se
         isso for MAIOR que `low[t-1]-low[t]` E positivo, senão `0`; `-DM`
         é o espelho do lado do fundo. Nunca os dois positivos no mesmo `t`
         (um exclui o outro pela comparação `>`).
      2. TR suavizado = `atr(high, low, close, n)` — REUSADO de
         `radar.lab.indicators` (é literalmente o mesmo ATR de Wilder que
         o resto do lab já usa; nenhuma fórmula nova aqui).
      3. `+DI`/`-DI = 100 · suavizado(±DM) / ATR(n)` — mesma suavização de
         Wilder (`alpha=1/n`) aplicada a `±DM`.
      4. `DX = 100 · |+DI − −DI| / (+DI + −DI)`; `ADX` é o `DX` suavizado de
         novo pela mesma constante (a "dupla suavização" padrão do ADX —
         por isso o primeiro valor não-`NaN` só aparece perto de `2n-1`
         barras, não `n`: a segunda EMA só começa a contar depois que a
         primeira já produziu `n` valores).

    Causalidade: `high.diff()`/`low.diff()` comparam `t` com `t-1` (nunca
    `t+1`); `.ewm(adjust=False)` é recursivo (só usa o valor acumulado até
    `t-1` e a observação em `t`) — o mesmo argumento causal que já justifica
    `ema`/`rsi`/`atr` em `radar.lab.indicators`. Divisão por zero (`+DI+−DI
    == 0`, mercado sem NENHUM range em `n` barras — degenerado, não
    observado em dado real) produz `NaN` explicitamente (nunca `inf`
    silencioso que poderia comparar `>= 20` como `True` por acidente).
    """
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index
    )

    atr_n = atr(high, low, close, n)
    plus_dm_suave = plus_dm.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    minus_dm_suave = minus_dm.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()

    plus_di = 100.0 * plus_dm_suave / atr_n
    minus_di = 100.0 * minus_dm_suave / atr_n

    soma_di = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / soma_di

    return dx.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def percentil_rolling(serie: pd.Series, janela: int) -> pd.Series:
    """
    Percentil (fração `<=`) do valor em `t` dentro da janela de `janela`
    observações TERMINANDO em `t` (inclusive) — `1.0` = o maior valor da
    janela, `0.0` = o menor. CAUSAL por construção: a janela nunca inclui
    `t+1` em diante (`rolling` padrão do pandas), e o valor comparado é o
    próprio último elemento da janela (`x[-1]`, que é `serie[t]`).

    `min_periods=janela` garante que só compara quando a janela inteira tem
    `janela` observações NÃO-`NaN` — como o único `NaN` esperado em `serie`
    é um PREFIXO contíguo (o warm-up da vol realizada de 21d que alimenta
    este percentil, ver `detectar_regime`), a partir do primeiro `t` em que
    `min_periods` é satisfeito a janela inteira já está livre de `NaN`
    (o prefixo já ficou inteiramente para trás dela) — nenhum valor de
    `NaN` contamina a comparação `x <= x[-1]`.
    """

    def _fracao_menor_ou_igual(janela_valores: np.ndarray) -> float:
        return float((janela_valores <= janela_valores[-1]).mean())

    return serie.rolling(window=janela, min_periods=janela).apply(_fracao_menor_ou_igual, raw=True)


def breadth_sma(precos_equities: pd.DataFrame, janela: int) -> pd.Series:
    """
    `% dos tickers com Close > SMA_própria(janela)` por `date`, sobre o
    universo de `precos_equities` (já sem `^BVSP`/símbolos não-ação — ver
    `detectar_regime`). Devolve uma `pd.Series` indexada por `date`.

    Cada ticker vira uma coluna (`pivot_table`, `aggfunc="mean"` — só entra
    em jogo se houver linha duplicada `(ticker, date)`, o que não deveria
    acontecer com dado limpo) e sua própria SMA é calculada CAUSALMENTE por
    `radar.lab.indicators.sma` (rolling `min_periods=n`, `NaN` enquanto a
    janela não fecha — inclusive para um ticker recém-listado). O `.where`
    depois da comparação `>` existe porque `valor > NaN` já devolve `False`
    em vez de `NaN` (semântica de comparação do pandas/numpy) — sem ele, um
    ticker sem SMA ainda formada entraria no denominador do breadth como
    "abaixo da própria SMA", o que é falso: não SABEMOS ainda, não é "não".
    """
    if precos_equities.empty:
        return pd.Series(dtype=float)

    pivot_close = precos_equities.pivot_table(index="date", columns="ticker", values="Close")
    sma_por_ticker = pivot_close.apply(lambda coluna: sma(coluna, janela))

    # `.astype(float)` ANTES do `.where` — sem isto, o `.where` que segue
    # (bool + NaN) deixa o DataFrame em dtype `object`, e `breadth_sma50`
    # sairia `object` na saída final em vez de `float64` (achado da bateria
    # de testes: `saida.dtypes` mostrava `object`, não um erro de valor).
    acima = (pivot_close > sma_por_ticker).astype(float)
    acima = acima.where(sma_por_ticker.notna())
    return acima.mean(axis=1, skipna=True)

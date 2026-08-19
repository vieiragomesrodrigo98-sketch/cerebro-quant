"""
src/radar/evidence/celulas_tecnicas.py — convenção ÚNICA de encode/decode da
célula técnica do Laboratório de Estratégias Técnicas (ADR 0021, Decisão 1).

`radar.evidence.ledger` (`Celula`, `medir()`, `aplicar_fdr()`) não muda: a
célula técnica é uma CONVENÇÃO sobre campos que já existem em `Celula`
(`evidencia`, `contexto`, `horizonte`), não um tipo novo. Este módulo é o
único lugar que sabe montar/desmontar essas strings — o mesmo papel que
`radar.evidence.categorias` cumpre para a célula de Fato Relevante: quem MEDE
(`scripts/build_technical_ledger.py`) e quem um dia vier a CONSUMIR o Ledger
técnico (o motor, lendo pesos) precisam concordar byte a byte. Duas cópias da
convenção divergiriam sem ninguém perceber — o mesmo erro de classe que a
docstring de `categorias.py` já descreve.

Zero I/O, zero import de `radar.lab`/pandas/numpy. `horizonte_da_spec` aceita
tanto um objeto com atributo `tempo_max_barras` (ex.:
`radar.lab.harness.SpecResolvida`, casado por DUCK TYPING via
`_TemTempoMaxBarras`) quanto um `int` já extraído pelo chamador — de propósito,
para não precisar importar `radar.lab` (que arrasta pandas/numpy/loguru) num
módulo que deve continuar leve o bastante para o motor de produção importar
um dia sem herdar essas dependências.

Formatos
--------
`evidencia`: ``"tec:<estrategia_id>@<params_hash8>"`` — o namespace ``tec:``
distingue da célula de Fato Relevante (``fr_*``, sem prefixo de namespace).

`contexto`: ``"tf:<timeframe>|mod:<modalidade>|liq:<faixa>|reg:<regime>"`` —
quatro campos, ORDEM fixa ao montar (determinismo do round-trip), mas
`parse_contexto` é tolerante à ordem ao ler (robustez contra reordenação
futura sem quebrar leitura de dado já gravado).

`horizonte`: holding máximo em DIAS ÚTEIS — é o lag do Newey-West
(`radar.evidence.ledger.medir` chama `t_newey_west(por_dia,
lag=max(1, horizonte))`). Day trade nunca dorme (a regra "fim_sessão" de
`radar.lab.harness.simular`), então `horizonte=1` sempre que
`modalidade == "day"`, qualquer que seja o `timeframe`. Para `timeframe="1d"`,
1 barra É 1 dia útil por definição: `horizonte = tempo_max_barras`. Para
timeframe intradiário (1m/5m/1h) fora de day trade (ex.: swing operado em
barras de 1h), convertemos barras -> dias úteis via uma aproximação de
barras-por-sessão de pregão B3 (~6h30 = 390min): 1m->390, 5m->78, 1h->7
(=ceil(390/60)). A aproximação é deliberadamente CONSERVADORA: para cripto
(calendário 24/7, mais barras por dia corrido que a B3), esta conta SUBESTIMA
barras/dia e por isso SUPERESTIMA o horizonte em dias — um lag de Newey-West
maior do que o estritamente necessário só perde poder estatístico, nunca
infla o `t` (o erro está sempre no lado seguro, qualquer que seja o mercado).
"""

from __future__ import annotations

import math
from typing import Protocol


class _TemTempoMaxBarras(Protocol):
    """
    Estrutural: qualquer objeto com `.tempo_max_barras: int` serve —
    `radar.lab.harness.SpecResolvida` casa por duck typing, sem que este
    módulo precise importar `radar.lab` (que traz pandas/numpy/loguru).
    """

    tempo_max_barras: int


# ── evidência: "tec:<estrategia_id>@<params_hash8>" ──────────────────────────

_PREFIXO_EVIDENCIA = "tec:"


def montar_evidencia(estrategia_id: str, params_hash8: str) -> str:
    """
    ``"tec:<estrategia_id>@<params_hash8>"`` — o namespace da célula técnica
    (ADR 0021, Decisão 1). `estrategia_id` é `EstrategiaSpec.id`;
    `params_hash8` é `radar.lab.spec.params_hash8(params)` da combinação
    medida.
    """
    if not estrategia_id:
        raise ValueError("montar_evidencia: estrategia_id não pode ser vazio")
    if not params_hash8:
        raise ValueError("montar_evidencia: params_hash8 não pode ser vazio")
    if "@" in estrategia_id or "@" in params_hash8:
        raise ValueError(
            f"montar_evidencia: nem estrategia_id ({estrategia_id!r}) nem "
            f"params_hash8 ({params_hash8!r}) podem conter '@' — é o separador"
        )
    return f"{_PREFIXO_EVIDENCIA}{estrategia_id}@{params_hash8}"


def parse_evidencia(evidencia: str) -> tuple[str, str]:
    """
    `(estrategia_id, params_hash8)` — inverso de `montar_evidencia`. Levanta
    `ValueError` claro se `evidencia` não seguir o formato
    ``"tec:<estrategia_id>@<params_hash8>"``.
    """
    if not evidencia.startswith(_PREFIXO_EVIDENCIA):
        raise ValueError(
            f"parse_evidencia: {evidencia!r} não começa com {_PREFIXO_EVIDENCIA!r} "
            "— não é uma evidência de célula técnica"
        )
    resto = evidencia[len(_PREFIXO_EVIDENCIA) :]
    partes = resto.split("@")
    if len(partes) != 2 or not partes[0] or not partes[1]:
        raise ValueError(
            f"parse_evidencia: {evidencia!r} — formato esperado "
            f"'{_PREFIXO_EVIDENCIA}<estrategia_id>@<params_hash8>', achou {resto!r}"
        )
    estrategia_id, params_hash8 = partes
    return estrategia_id, params_hash8


# ── contexto: "tf:<tf>|mod:<mod>|liq:<faixa>|reg:<regime>" ───────────────────

TIMEFRAMES_VALIDOS: tuple[str, ...] = ("1m", "5m", "1h", "1d")
MODALIDADES_VALIDAS: tuple[str, ...] = ("day", "swing", "position")
LIQUIDEZ_VALIDA: tuple[str, ...] = ("todos", "liquida", "media", "iliquida")
REGIMES_VALIDOS: tuple[str, ...] = ("alta", "baixa", "lateral", "any")

_PREFIXO_POR_CAMPO: dict[str, str] = {
    "timeframe": "tf",
    "modalidade": "mod",
    "liquidez": "liq",
    "regime": "reg",
}
_CAMPO_POR_PREFIXO: dict[str, str] = {v: k for k, v in _PREFIXO_POR_CAMPO.items()}


def _validar_dominio(campo: str, valor: str, dominio: tuple[str, ...]) -> None:
    if valor not in dominio:
        raise ValueError(
            f"celulas_tecnicas: valor inválido para {campo!r}: {valor!r} — "
            f"precisa ser um de {dominio}"
        )


def montar_contexto(timeframe: str, modalidade: str, liquidez: str, regime: str = "any") -> str:
    """
    ``"tf:<timeframe>|mod:<modalidade>|liq:<liquidez>|reg:<regime>"``
    (ADR 0021, Decisão 1). Valida os quatro domínios antes de montar —
    string malformada nunca sai daqui.
    """
    _validar_dominio("timeframe", timeframe, TIMEFRAMES_VALIDOS)
    _validar_dominio("modalidade", modalidade, MODALIDADES_VALIDAS)
    _validar_dominio("liquidez", liquidez, LIQUIDEZ_VALIDA)
    _validar_dominio("regime", regime, REGIMES_VALIDOS)
    return f"tf:{timeframe}|mod:{modalidade}|liq:{liquidez}|reg:{regime}"


def parse_contexto(contexto: str) -> tuple[str, str, str, str]:
    """
    `(timeframe, modalidade, liquidez, regime)` — inverso de `montar_contexto`.

    Tolerante à ORDEM dos quatro campos (útil para ler um contexto escrito por
    uma versão futura que reordene os campos), mas exige os quatro presentes,
    sem duplicata, e cada valor dentro do domínio válido. Levanta `ValueError`
    claro em qualquer desvio.
    """
    partes = contexto.split("|")
    if len(partes) != 4:
        raise ValueError(
            f"parse_contexto: {contexto!r} — esperado 4 campos "
            f"'tf:..|mod:..|liq:..|reg:..', achou {len(partes)}"
        )

    campos: dict[str, str] = {}
    for parte in partes:
        prefixo, sep, valor = parte.partition(":")
        if not sep or prefixo not in _CAMPO_POR_PREFIXO:
            raise ValueError(
                f"parse_contexto: campo {parte!r} inválido em {contexto!r} — prefixo "
                f"precisa ser um de {sorted(_CAMPO_POR_PREFIXO)}"
            )
        nome_campo = _CAMPO_POR_PREFIXO[prefixo]
        if nome_campo in campos:
            raise ValueError(f"parse_contexto: campo {nome_campo!r} duplicado em {contexto!r}")
        campos[nome_campo] = valor

    faltando = set(_CAMPO_POR_PREFIXO.values()) - set(campos)
    if faltando:
        raise ValueError(f"parse_contexto: {contexto!r} — faltando campo(s) {sorted(faltando)}")

    timeframe = campos["timeframe"]
    modalidade = campos["modalidade"]
    liquidez = campos["liquidez"]
    regime = campos["regime"]

    _validar_dominio("timeframe", timeframe, TIMEFRAMES_VALIDOS)
    _validar_dominio("modalidade", modalidade, MODALIDADES_VALIDAS)
    _validar_dominio("liquidez", liquidez, LIQUIDEZ_VALIDA)
    _validar_dominio("regime", regime, REGIMES_VALIDOS)
    return timeframe, modalidade, liquidez, regime


# ── horizonte: holding máximo em DIAS ÚTEIS (o lag do Newey-West) ────────────

# Barras por sessão de pregão B3 (~6h30 = 390min) — ver docstring do módulo
# para a direção conservadora desta aproximação também fora da B3.
_BARRAS_POR_DIA_PREGAO: dict[str, int] = {"1m": 390, "5m": 78, "1h": 7}


def horizonte_da_spec(
    spec_resolvida_ou_tempo_max_barras: _TemTempoMaxBarras | int,
    timeframe: str,
    modalidade: str,
) -> int:
    """
    Holding máximo em DIAS ÚTEIS — o `horizonte` de `Celula`, e portanto o lag
    do Newey-West (`radar.evidence.ledger.medir` chama `t_newey_west` com
    `lag=max(1, horizonte)`).

    Aceita um `SpecResolvida` (ou qualquer objeto com `.tempo_max_barras`) OU
    o `int` já extraído pelo chamador — ver docstring do módulo para o porquê
    do duck typing.

    Regra (nesta ordem):
      1. `modalidade == "day"` -> 1 (nunca dorme, ver docstring do módulo).
      2. `timeframe == "1d"` -> `tempo_max_barras` (1 barra = 1 dia útil).
      3. timeframe intradiário -> `ceil(tempo_max_barras / barras_por_dia)`.
    """
    _validar_dominio("modalidade", modalidade, MODALIDADES_VALIDAS)
    _validar_dominio("timeframe", timeframe, TIMEFRAMES_VALIDOS)

    tempo_max_barras = (
        spec_resolvida_ou_tempo_max_barras
        if isinstance(spec_resolvida_ou_tempo_max_barras, int)
        else int(spec_resolvida_ou_tempo_max_barras.tempo_max_barras)
    )
    if tempo_max_barras <= 0:
        raise ValueError(
            f"horizonte_da_spec: tempo_max_barras precisa ser > 0, recebeu {tempo_max_barras!r}"
        )

    if modalidade == "day":
        return 1

    if timeframe == "1d":
        return tempo_max_barras

    barras_por_dia = _BARRAS_POR_DIA_PREGAO[timeframe]
    return max(1, math.ceil(tempo_max_barras / barras_por_dia))

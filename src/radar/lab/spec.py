"""
Spec declarativa de estratégia técnica — o "o quê" da grade, sem nenhum "quanto".

Este módulo é PURO: zero I/O, zero import de banco, rede ou `radar.llm`. Ele só
declara a FORMA de uma estratégia (`EstrategiaSpec`) e a grade de parâmetros que
a testa (`combinacoes_da_grade`) — quem executa a estratégia sobre dado real é o
harness (Sessão A2), e quem mede o veredito continua sendo `radar.evidence.ledger`
(ADR 0020, inalterado).

Por que nenhum número do Manual vira default
----------------------------------------------
Uma auditoria encontrou 31 números inventados em `docs/manual_trader_ai/` — limiares
e janelas que soam plausíveis mas nunca foram medidos. A defesa estrutural contra
isso é: `EstrategiaSpec` não tem um único campo numérico com valor fixo de "melhor
prática". Todo número que o Manual sugere (período de RSI, multiplicador de ATR,
razão de risco/retorno) entra em `grade_params` — um EIXO varrido, nunca uma
constante. O que autoriza um valor a virar peso de produção é uma célula 🟢 no
Ledger, nunca a opinião de um capítulo do Manual (ADR 0020, Trava 1; ADR 0021,
Decisão 1).

Congelamento da grade (ADR 0021, Decisão 3)
--------------------------------------------
`catalogo_hash` e `validar_orcamento_grade` existem para que o catálogo inteiro de
estratégias seja congelado — por hash SHA-256 — ANTES da primeira rodada de medição.
A família de FDR (`aplicar_fdr` em `radar.evidence.ledger`) é uma propriedade da
GRADE, não da célula isolada: adicionar uma estratégia depois de medir é escolher o
denominador depois de ver o resultado, o mesmo garimpo que a faca 7 do Ledger existe
para impedir.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field

# ── Cond — condição declarativa de entrada ────────────────────────────────────

_OPS_COND = {">", "<", ">=", "<=", "cruza_acima", "cruza_abaixo"}


@dataclass(frozen=True)
class Cond:
    """
    Uma condição de entrada: `campo <op> ref`.

    `campo` é o nome de uma coluna já calculada pelo harness a partir de
    `radar.lab.indicators` (ex.: "close", "rsi_14", "atr_14" — o sufixo numérico
    é montado pelo harness na Sessão A2, aqui só o nome da coluna importa).
    `ref` é outro campo (str) ou uma constante numérica (float) — permite tanto
    "rsi_14 < 30" quanto "close cruza_acima sma_20".
    """

    campo: str
    op: str
    ref: str | float

    def __post_init__(self) -> None:
        if not self.campo:
            raise ValueError("Cond.campo não pode ser vazio")
        if self.op not in _OPS_COND:
            raise ValueError(
                f"Cond.op inválido: {self.op!r} — precisa ser um de {sorted(_OPS_COND)}"
            )


# ── RegraStop / RegraAlvo — saída declarativa ────────────────────────────────

_TIPOS_STOP = {"atr", "minima_n", "coluna"}
_TIPOS_ALVO = {"rr", "maxima_n", "coluna"}


@dataclass(frozen=True)
class RegraStop:
    """
    Regra de stop. `tipo="atr"` usa `valor` como multiplicador k do ATR;
    `tipo="minima_n"` usa `valor` como N barras (stop na mínima das N anteriores);
    `tipo="coluna"` (Sessão F8-S2) usa `coluna` — nome de coluna já construída
    pelo harness (`radar.lab.indicators`), placeholders `{chave}` permitidos —
    como o NÍVEL de referência, e `valor` como multiplicador/buffer aplicado
    sobre esse nível: `coluna="fundo_preco_1_5"`, `valor=0.99` = stop 1% ABAIXO
    do valor da coluna (lido, como todo stop/alvo do harness, na barra do
    SINAL — nunca da entrada). Pré-requisito dos detectores compostos da S3
    (fundo duplo/OCO): o stop mora no fundo, o alvo na projeção da neckline —
    ambos COLUNAS que o detector produz, não ATR/mínima genérica.

    `coluna` só é válida (e obrigatória) quando `tipo="coluna"` — nos outros
    tipos, `coluna is None` é exigido (nunca uma combinação ambígua).
    """

    tipo: str
    valor: float | int
    coluna: str | None = None

    def __post_init__(self) -> None:
        if self.tipo not in _TIPOS_STOP:
            raise ValueError(
                f"RegraStop.tipo inválido: {self.tipo!r} — precisa ser um de {sorted(_TIPOS_STOP)}"
            )
        if self.valor <= 0:
            raise ValueError(f"RegraStop.valor precisa ser > 0, recebeu {self.valor!r}")
        if self.tipo == "coluna":
            if not self.coluna:
                raise ValueError(
                    "RegraStop.coluna é obrigatória e não pode ser vazia quando "
                    "tipo='coluna' (nome da coluna que fornece o nível de referência)"
                )
        elif self.coluna is not None:
            raise ValueError(
                f"RegraStop.coluna só é permitida quando tipo='coluna' — "
                f"tipo={self.tipo!r} recebeu coluna={self.coluna!r}"
            )


@dataclass(frozen=True)
class RegraAlvo:
    """
    Regra de alvo. `tipo="rr"` usa `valor` como múltiplo de risco:retorno sobre a
    distância do stop; `tipo="maxima_n"` usa `valor` como N barras (alvo na máxima
    das N anteriores); `tipo="coluna"` (Sessão F8-S2) é simétrico a
    `RegraStop.tipo="coluna"` — `coluna` é o nível de referência (placeholders
    `{chave}` permitidos), `valor` o multiplicador/buffer sobre ele (ex.:
    `valor=1.02` = alvo 2% ACIMA do valor da coluna, lido na barra do SINAL).

    `coluna` só é válida (e obrigatória) quando `tipo="coluna"` — nos outros
    tipos, `coluna is None` é exigido.
    """

    tipo: str
    valor: float | int
    coluna: str | None = None

    def __post_init__(self) -> None:
        if self.tipo not in _TIPOS_ALVO:
            raise ValueError(
                f"RegraAlvo.tipo inválido: {self.tipo!r} — precisa ser um de {sorted(_TIPOS_ALVO)}"
            )
        if self.valor <= 0:
            raise ValueError(f"RegraAlvo.valor precisa ser > 0, recebeu {self.valor!r}")
        if self.tipo == "coluna":
            if not self.coluna:
                raise ValueError(
                    "RegraAlvo.coluna é obrigatória e não pode ser vazia quando "
                    "tipo='coluna' (nome da coluna que fornece o nível de referência)"
                )
        elif self.coluna is not None:
            raise ValueError(
                f"RegraAlvo.coluna só é permitida quando tipo='coluna' — "
                f"tipo={self.tipo!r} recebeu coluna={self.coluna!r}"
            )


# ── EstrategiaSpec — a estratégia inteira ────────────────────────────────────

_ID_RE = re.compile(r"^[a-z0-9_]+$")
_MODALIDADES = {"day", "swing", "position"}
_MERCADOS = {"b3", "cripto"}
_TIMEFRAMES = {"1m", "5m", "1h", "1d"}
_UNIVERSOS = {
    "b3_liquida",
    "b3_universo",
    "cripto_top",
    # CATALOGO_V3_PESQUISA01 (S152/S153) — universos cripto MAIS FINOS que
    # "cripto_top" (top-50 inteiro): "cripto_majors" (BTC+ETH),
    # "cripto_top_ex_majors" (top-50 menos majors) e "cripto_btc" (só BTC,
    # `v3c5_janela_2123utc`). O filtro de símbolo por universo é aplicado em
    # `scripts/build_technical_ledger.py::filtrar_universo_por_spec` — ver
    # `radar.lab.data.CRIPTO_MAJORS`/`CRIPTO_TOP50_EX_MAJORS`.
    "cripto_majors",
    "cripto_top_ex_majors",
    "cripto_btc",
}


@dataclass(frozen=True)
class EstrategiaSpec:
    """
    Uma estratégia técnica declarada — sem nenhum número fixo, só a FORMA da regra
    e os eixos (`grade_params`) que a varredura testa.

    `capitulo` amarra a estratégia a um capítulo de `docs/manual_trader_ai/` (ex.:
    "parte_09_analise_tecnica/cap_55_rompimentos.md") — rastreabilidade obrigatória
    entre o que o Manual descreve e o que o laboratório mede.
    """

    id: str
    capitulo: str
    modalidade: str
    mercados: tuple[str, ...]
    timeframes: tuple[str, ...]
    direcao: str
    universo: str
    entrada: tuple[Cond, ...]
    stop: RegraStop
    alvo: RegraAlvo
    tempo_max_barras: int
    grade_params: dict[str, tuple[float | int, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _ID_RE.match(self.id):
            raise ValueError(
                f"EstrategiaSpec.id inválido: {self.id!r} — precisa casar {_ID_RE.pattern!r}"
            )
        if not self.capitulo:
            raise ValueError(
                "EstrategiaSpec.capitulo não pode ser vazio — rastreabilidade ao "
                "Manual (docs/manual_trader_ai/) é obrigatória"
            )
        if self.modalidade not in _MODALIDADES:
            raise ValueError(
                f"EstrategiaSpec.modalidade inválida: {self.modalidade!r} — "
                f"precisa ser um de {sorted(_MODALIDADES)}"
            )
        if not self.mercados or not set(self.mercados) <= _MERCADOS:
            raise ValueError(
                f"EstrategiaSpec.mercados inválido: {self.mercados!r} — precisa ser "
                f"subconjunto não vazio de {sorted(_MERCADOS)}"
            )
        if not self.timeframes or not set(self.timeframes) <= _TIMEFRAMES:
            raise ValueError(
                f"EstrategiaSpec.timeframes inválido: {self.timeframes!r} — precisa "
                f"ser subconjunto não vazio de {sorted(_TIMEFRAMES)}"
            )
        if self.direcao != "long":
            raise ValueError(
                f"EstrategiaSpec.direcao={self.direcao!r} não suportada — v1 do "
                "laboratório é LONG-only (ADR 0021, Decisão 5); long&short (cap. 78) "
                "fica registrado, não implementado"
            )
        if self.universo not in _UNIVERSOS:
            raise ValueError(
                f"EstrategiaSpec.universo inválido: {self.universo!r} — precisa ser "
                f"um de {sorted(_UNIVERSOS)}"
            )
        if not self.entrada:
            raise ValueError(
                "EstrategiaSpec.entrada não pode ser vazia — toda estratégia precisa "
                "de ao menos uma condição de entrada"
            )
        if self.tempo_max_barras <= 0:
            raise ValueError(
                f"EstrategiaSpec.tempo_max_barras precisa ser > 0, recebeu "
                f"{self.tempo_max_barras!r}"
            )


# ── Hash e grade ──────────────────────────────────────────────────────────────


def params_hash8(params: Mapping[str, object]) -> str:
    """
    SHA-256 de `params` serializado em JSON canônico (chaves ordenadas,
    separadores compactos), truncado nos primeiros 8 hex.

    Usado para compor `evidencia="tec:<estrategia_id>@<params_hash8>"` (ADR 0021,
    Decisão 1) — precisa ser determinístico e indiferente à ordem em que as
    chaves de `params` foram construídas.
    """
    canonico = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonico.encode("utf-8")).hexdigest()[:8]


def combinacoes_da_grade(spec: EstrategiaSpec) -> list[dict[str, float | int]]:
    """
    Produto cartesiano ordenado e determinístico dos eixos de `grade_params`.

    As chaves são ordenadas alfabeticamente e cada eixo mantém a ordem em que
    foi declarado — chamar esta função duas vezes com a mesma spec produz
    exatamente a mesma lista, na mesma ordem. Cada combinação vira 1 célula
    do Ledger técnico (multiplicada por mercado × timeframe — ver
    `tamanho_da_grade`).
    """
    if not spec.grade_params:
        return [{}]
    chaves = sorted(spec.grade_params.keys())
    eixos = [spec.grade_params[chave] for chave in chaves]
    return [dict(zip(chaves, combo, strict=True)) for combo in itertools.product(*eixos)]


def _validar_ids_unicos(specs: Sequence[EstrategiaSpec]) -> None:
    """
    Levanta `ValueError` se houver `id` duplicado no catálogo.

    O `id` compõe a chave da célula (`evidencia="tec:<id>@<params_hash8>"`) —
    duas specs com o mesmo `id` colidiriam células silenciosamente e
    corromperiam a contagem da família de FDR.
    """
    vistos: set[str] = set()
    duplicados: set[str] = set()
    for spec in specs:
        if spec.id in vistos:
            duplicados.add(spec.id)
        vistos.add(spec.id)
    if duplicados:
        raise ValueError(f"catálogo com id(s) duplicado(s): {sorted(duplicados)}")


def _serializar_regra(regra: RegraStop | RegraAlvo) -> dict[str, object]:
    """
    Forma canônica de `RegraStop`/`RegraAlvo` para hashing — OMITE a chave
    `coluna` quando `None` (o caso de todas as specs anteriores à Sessão F8-S2,
    que não usam `tipo="coluna"`). Isto é o que garante que `catalogo_hash`
    das specs v1 existentes não mude por causa de um campo novo que elas nunca
    usam — só uma spec que de fato declara `tipo="coluna"` (e portanto tem
    `coluna` não-`None`) entra com a chave extra no hash.
    """
    dados = asdict(regra)
    if dados.get("coluna") is None:
        dados.pop("coluna", None)
    return dados


def _serializar_spec(spec: EstrategiaSpec) -> dict[str, object]:
    """Forma canônica de uma spec para hashing — nunca depende da ordem de construção em memória."""
    return {
        "id": spec.id,
        "capitulo": spec.capitulo,
        "modalidade": spec.modalidade,
        "mercados": list(spec.mercados),
        "timeframes": list(spec.timeframes),
        "direcao": spec.direcao,
        "universo": spec.universo,
        "entrada": [asdict(cond) for cond in spec.entrada],
        "stop": _serializar_regra(spec.stop),
        "alvo": _serializar_regra(spec.alvo),
        "tempo_max_barras": spec.tempo_max_barras,
        "grade_params": {k: list(v) for k, v in sorted(spec.grade_params.items())},
    }


def catalogo_hash(specs: Sequence[EstrategiaSpec]) -> str:
    """
    SHA-256 canônico do catálogo INTEIRO — congela a grade antes da 1ª rodada
    (ADR 0021, Decisão 3).

    Ordena as specs por `id` e serializa cada uma de forma determinística, para
    que o hash não dependa da ordem em que o catálogo foi montado em memória.
    """
    _validar_ids_unicos(specs)
    ordenado = sorted(specs, key=lambda s: s.id)
    canonico = json.dumps(
        [_serializar_spec(s) for s in ordenado], sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonico.encode("utf-8")).hexdigest()


TETO_GRADE = 1000  # teto de células por rodada — ADR 0021, Decisão 3


def tamanho_da_grade(specs: Sequence[EstrategiaSpec]) -> int:
    """
    Total de células que o catálogo inteiro produziria: soma, por spec, de
    |combinações de `grade_params`| × |mercados| × |timeframes|.
    """
    total = 0
    for spec in specs:
        n_combinacoes = len(combinacoes_da_grade(spec))
        total += n_combinacoes * len(spec.mercados) * len(spec.timeframes)
    return total


def validar_orcamento_grade(specs: Sequence[EstrategiaSpec]) -> None:
    """
    Levanta `ValueError` se o catálogo estourar `TETO_GRADE` células.

    O teto não é um limite de performance — é o que torna a família de FDR da
    rodada (ADR 0020, faca 7) auditável e finita: adicionar estratégia depois
    de uma rodada exige re-rodar a grade inteira, nunca só somar uma célula.
    """
    _validar_ids_unicos(specs)
    tamanho = tamanho_da_grade(specs)
    if tamanho > TETO_GRADE:
        raise ValueError(
            f"grade estoura o orçamento: {tamanho} células > TETO_GRADE={TETO_GRADE} "
            "— reduza eixos ou divida em mais de uma rodada (ADR 0021, Decisão 3)"
        )

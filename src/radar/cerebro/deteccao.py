"""
src/radar/cerebro/deteccao.py — o estágio OPPORTUNITY DETECTION.

Pergunta única: **"há algo acontecendo?"** — e nada além disso.

⚠️ DETECÇÃO NÃO IMPLICA SINAL
-----------------------------
Está na skill `raciocinio-quant` e é a trava deste módulo. Uma situação
detectada é candidata a ser avaliada; ela **não** afirma vantagem, não estima
probabilidade e não decide. Quem pergunta *"essa oportunidade tem vantagem?"* é
`expectativa.py`; *"devemos emitir?"* é `decisao.py`.

O estágio existe por economia, não por elegância: avaliar evidência e
expectativa é caro, e a varredura contínua roda sobre o universo INTEIRO. A
detecção é o filtro barato que decide **o que merece o caro** — e por isso ela
precisa ser generosa: falso positivo aqui custa cálculo; falso negativo custa a
oportunidade inteira, e ninguém fica sabendo.

O que este módulo NÃO faz
-------------------------
Não elimina ativo (ADR 0036: nada sai do universo por mérito), não pontua, não
ordena e **não conhece estratégia nem família** — a mesma situação alimenta
famílias diferentes, e amarrá-la a uma delas aqui mataria a
`ativo × estratégia × família` que é a unidade de análise.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from radar.cerebro.monitor import Observacao

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


class Situacao(StrEnum):
    """O vocabulário de situações — deliberadamente **descritivo**, nunca
    avaliativo.

    Nenhum nome aqui contém juízo (`bom`, `forte`, `provável`). A razão é o
    guardrail *resultado da estratégia ≠ propriedade do ativo*: rotular a
    situação com o resultado esperado contaminaria a detecção com a conclusão,
    e o mesmo `BREAKOUT` que funciona numa família falha em outra.
    """

    BREAKOUT = "breakout"
    CONTINUACAO = "continuacao"
    REVERSAO = "reversao"
    COMPRESSAO = "compressao"
    EXPANSAO = "expansao"
    EXAUSTAO = "exaustao"
    MOMENTUM = "momentum"
    REVERSAO_A_MEDIA = "reversao_a_media"
    ANOMALIA = "anomalia"
    EVENTO = "evento"
    CROSS_SECTIONAL = "cross_sectional"


@dataclass(frozen=True, slots=True)
class Candidato:
    """Um ativo com ao menos uma situação detectada, pronto para ser AVALIADO.

    **Não tem score, probabilidade nem direção** — a mesma trava de
    `Observacao`. Se ganhar, a decomposição morreu e a próxima autópsia volta a
    ser impossível.
    """

    observacao: Observacao
    situacoes: frozenset[Situacao]
    #: Números que o detector já calculou e que os estágios seguintes
    #: aproveitariam recalcular. É **cache**, não julgamento: nada aqui pode ser
    #: lido como "quão boa" a oportunidade é.
    evidencias: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.situacoes:
            raise ValueError(
                "Candidato exige ao menos uma situação — ativo sem situação "
                "detectada não é candidato, é observação (e continua no universo)"
            )

    @property
    def ativo_id(self) -> str:
        return self.observacao.ativo.ativo_id

    @property
    def chave_de_particao(self) -> str:
        return self.observacao.chave_de_particao


class DetectorDeSituacao(Protocol):
    """Quem sabe reconhecer UMA classe de situação.

    Injetado, como todo o resto do Cérebro: `radar.cerebro` não conhece
    implementação de motor. Devolve o conjunto de situações que reconheceu —
    vazio é resposta válida e comum.
    """

    def __call__(
        self, observacao: Observacao, precos: pd.DataFrame
    ) -> Iterable[Situacao]:  # pragma: no cover — contrato
        ...


def detectar_situacoes(
    observacao: Observacao,
    precos: pd.DataFrame,
    detectores: Sequence[DetectorDeSituacao],
) -> Candidato | None:
    """Aplica os detectores e devolve um `Candidato`, ou `None` se nada foi
    reconhecido.

    `None` — e não um candidato vazio — porque "nada acontecendo" é o estado
    **normal** da varredura total: com o universo inteiro a cada ciclo, a
    imensa maioria dos ativos não apresenta situação alguma, e criar objeto para
    cada um seria ruído com custo.

    Observação inelegível não é avaliada: sem contexto legível não há partição,
    e sem partição a situação não pode ser interpretada (G-P1). Ela **não sai
    do universo** — apenas não vira candidata neste ciclo.

    Detector que levanta é ignorado e os demais seguem — mas **a falha é
    CONTADA**, não engolida. Isto foi corrigido na revisão da Fase 2: a primeira
    versão fazia `except Exception: continue` em silêncio, e isso contradizia o
    princípio que eu mesmo tinha aplicado ao monitor três arquivos antes (lá o
    ativo inelegível **aparece** no resultado, porque omitir faria "não observei"
    e "observei e estava limpo" ficarem indistinguíveis).

    Um detector com bug que sempre levanta produziria zero situações para
    sempre, e sem contagem isso é indistinguível de "o mercado está calmo" —
    exatamente a família de defeito que o canário do scan de governança existe
    para impedir. `falhas_de_detector` sai preenchido para quem quiser somar.
    """
    if not observacao.elegivel_no_ciclo:
        return None

    encontradas: set[Situacao] = set()
    falhas = 0
    for detector in detectores:
        try:
            encontradas.update(detector(observacao, precos))
        except Exception:
            falhas += 1
    if not encontradas:
        return None
    return Candidato(
        observacao=observacao,
        situacoes=frozenset(encontradas),
        evidencias={"falhas_de_detector": float(falhas)} if falhas else {},
    )

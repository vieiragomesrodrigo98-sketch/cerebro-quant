"""
src/radar/cerebro/oportunidade.py — a unidade de decisão do Cérebro v2.

ADR 0032 §8.3: *"a unidade de decisão passa a ser a **oportunidade**, com
identidade própria (`opportunity_id`) e ciclo de vida explícito"*. O mesmo ativo
acumula oportunidades independentes, e **o histórico não bloqueia o futuro**.

O que o código fazia antes
---------------------------
`fsim_autopilot_open.py` monta `skip = open_tickers | pending_tickers` e descarta
todo sinal cujo ticker já tenha posição ou ordem. Consequências medidas no ADR:
reforço impossível; estratégias concorrentes no mesmo ativo impossíveis (um sinal
de swing em BTC morre porque existe uma posição de scalp em BTC); reentrada após
stop funciona no scalp e não no swing.

Por que dedup e oportunidade nascem no MESMO módulo
----------------------------------------------------
Porque separá-los cria o problema que a decisão resolve. §8.3, trava 1: *"se o
mesmo ativo gerar 5 oportunidades no dia e as 5 entrarem no Ledger como trades
independentes, o `t` infla exatamente como já inflou neste projeto: **5,08 →
0,30** quando o clustering por dia foi corrigido (S142-143)"*.

`opportunity_id` é a unidade de **DEDUP ESTATÍSTICO**, não um identificador de
conveniência. Na execução, 5 ordens são 5 ordens; na estatística, mesmo ativo +
mesmo motor + janela de posição sobreposta contam como **UM** setup.

O que este módulo NÃO faz
--------------------------
Não decide tamanho, não escolhe entre oportunidades, não persiste. É o
vocabulário e a aritmética do dedup — alocação é do Portfolio Engine, e
persistência é da borda.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any, Final

__all__ = [
    "Acao",
    "Estado",
    "Oportunidade",
    "chave_de_setup",
    "dedup_por_setup",
]


class Estado(str, Enum):
    """
    O ciclo de vida do §8.3, sem estado intermediário inventado.

    `CRIADA → ORDEM → ABERTA → (gestão) → FECHADA`. A gestão acontece **dentro**
    de `ABERTA` — aumentar, reduzir e mover stop não mudam o estado, porque a
    oportunidade continua sendo a mesma. Um estado por ação faria o ciclo
    depender do que se fez, não do que se tem.
    """

    CRIADA = "oportunidade_criada"
    ORDEM = "ordem"
    ABERTA = "posicao_aberta"
    FECHADA = "posicao_fechada"

    @property
    def ocupa_o_ativo(self) -> bool:
        """`True` enquanto a oportunidade segura o ativo (§8.3 trava 4).

        `ORDEM` ocupa: uma ordem pendente trava o ativo — e é por isso que ela
        precisa de expiração, senão uma ordem que nunca preenche sequestra o
        ativo para sempre."""
        return self in (Estado.ORDEM, Estado.ABERTA)


class Acao(str, Enum):
    """
    O que o Cérebro decide sobre uma oportunidade.

    §8.3: as ações deixam de ser `COMPRAR/NÃO COMPRAR`. `AUMENTAR` está aqui
    como vocabulário, mas o ADR o marca como **o item mais perigoso da lista** —
    piramidar concentra risco exatamente quando a posição está maior, e este
    projeto já mediu o modo de falha (S165, uma posição consumiu a carteira).
    Nasce como **hipótese a medir**, nunca como funcionalidade a ligar.
    """

    ABRIR = "abrir"
    AUMENTAR = "aumentar"
    MANTER = "manter"
    REDUZIR = "reduzir"
    REALIZAR_PARCIAL = "realizar_parcial"
    MOVER_STOP = "mover_stop"
    FECHAR = "fechar"
    REENTRAR = "reentrar"


#: Motivos de invalidação que ENCERRAM a oportunidade. Espelham
#: `radar.mie.signal_builder.construir_invalidacao()` — mesmo vocabulário, de
#: propósito: quem lê o trail não deve aprender duas palavras para a mesma ideia.
MOTIVOS_DE_FIM: Final[tuple[str, ...]] = (
    "stop_tocado",
    "regime_panico",
    "tempo_esgotado",
)


@dataclass(frozen=True)
class Oportunidade:
    """
    Uma decisão com identidade e fim declarado.

    **O fim é declarado na criação, não descoberto depois** (§8.3 trava 2). Uma
    oportunidade só TERMINA quando uma das suas condições de invalidação
    dispara; até lá, sinal novo do mesmo motor no mesmo ativo é
    **continuidade**, não evento novo. É isto que impede o "comprar 100 vezes
    porque o modelo continua dizendo comprar" — sem inventar cooldown
    arbitrário, que o ADR proíbe explicitamente (o controle de overtrading é
    ECONÔMICO: a oportunidade nova tem de pagar o round-trip que implica).
    """

    ativo: str
    motor: str
    """A família que a originou (`scalp_cripto_v2`, `position_cripto_v1`...).
    Duas famílias no mesmo ativo são oportunidades DIFERENTES."""

    criada_em: datetime
    invalidacao: tuple[str, ...]
    """As condições que a encerram. **Vazio levanta**: oportunidade sem fim
    declarado nunca termina, e o ativo fica travado para sempre."""

    estado: Estado = Estado.CRIADA
    fechada_em: datetime | None = None
    motivo_fim: str | None = None
    contexto: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.invalidacao:
            raise ValueError(
                f"oportunidade em {self.ativo!r} sem condição de invalidação — "
                "sem fim declarado ela nunca termina e o ativo fica travado"
            )
        if self.estado is Estado.FECHADA and self.motivo_fim is None:
            raise ValueError("oportunidade FECHADA exige motivo_fim")

    @property
    def id(self) -> str:
        """`opportunity_id` — determinístico, derivado do que a identifica.

        Determinístico e não aleatório para que a MESMA oportunidade recalculada
        produza o mesmo id: um `uuid4()` faria a reexecução de um dia gerar
        oportunidades "novas" e inflar a contagem, que é exatamente o defeito
        que o dedup existe para impedir."""
        crua = f"{self.motor}|{self.ativo}|{self.criada_em.isoformat()}"
        return hashlib.sha256(crua.encode()).hexdigest()[:16]

    @property
    def viva(self) -> bool:
        return self.estado is not Estado.FECHADA

    def com_estado(self, novo: Estado, *, em: datetime, motivo: str | None = None) -> Oportunidade:
        """Avança o ciclo. Fechar exige motivo — e ele tem de ser declarado."""
        if novo is Estado.FECHADA:
            if motivo is None:
                raise ValueError("fechar exige motivo")
            if motivo not in self.invalidacao:
                raise ValueError(
                    f"motivo {motivo!r} não está entre as invalidações declaradas "
                    f"desta oportunidade: {self.invalidacao}"
                )
            return replace(self, estado=novo, fechada_em=em, motivo_fim=motivo)
        return replace(self, estado=novo)


def chave_de_setup(o: Oportunidade, *, janela: Any) -> tuple[str, str, Any]:
    """
    A chave de DEDUP ESTATÍSTICO: `(motor, ativo, janela)`.

    §8.3 trava 1 — *"mesmo ativo + mesmo motor + janela de posição sobreposta
    contam como UM setup na estatística, ainda que sejam ordens distintas na
    execução"*.

    `janela` é o balde temporal do horizonte da família: duas oportunidades do
    mesmo motor no mesmo ativo dentro do mesmo balde são **um** setup. Quem
    escolhe o balde é a família (o horizonte dela), não este módulo — um balde
    fixo aqui seria o cooldown arbitrário que o ADR proíbe.
    """
    return (o.motor, o.ativo, janela)


def dedup_por_setup(
    oportunidades: list[Oportunidade],
    *,
    janela_de: Any,
) -> list[Oportunidade]:
    """
    Reduz uma lista de oportunidades a **um setup por (motor, ativo, janela)**.

    `janela_de` é uma função `Oportunidade -> balde`. Mantém a **primeira** de
    cada balde, na ordem de criação: a primeira é a que teria sido executada, e
    escolher a melhor seria olhar o resultado para decidir o que contar.

    Isto é o que separa a estatística da execução. Na execução, 5 ordens são 5
    ordens e podem todas existir. No Ledger, elas contam como 1 — senão o `t`
    infla como já inflou: **5,08 → 0,30** (S142-143).
    """
    vistos: set[tuple[str, str, Any]] = set()
    saida: list[Oportunidade] = []
    for o in sorted(oportunidades, key=lambda x: x.criada_em):
        chave = chave_de_setup(o, janela=janela_de(o))
        if chave in vistos:
            continue
        vistos.add(chave)
        saida.append(o)
    return saida

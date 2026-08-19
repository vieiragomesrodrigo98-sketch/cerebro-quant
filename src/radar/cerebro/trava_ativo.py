"""
src/radar/cerebro/trava_ativo.py — a trava exclusiva por ativo (ADR 0032 §8.3.4).

Decisão do DEV: *"ele pode escolher mais de uma [modalidade], mas o mesmo ativo
não entra em outro modelo após entrar em algum tipo de trade"*.

Regra: **enquanto um ativo tiver posição aberta por um motor, ele fica travado
para todos os outros motores daquela conta, até a posição fechar.** O
multi-select no perfil fica liberado (o usuário escolhe scalp *e* swing); a
exclusividade é do **ATIVO**, não da modalidade.

Por que isto resolve o problema físico por construção
------------------------------------------------------
Há sempre **uma posição lógica por ativo por conta**, que é exatamente **uma
posição física** na exchange. O simulador deixa de poder representar algo que a
execução não reproduz — que era a razão do risco, e o que dispensa as três
alternativas cogitadas antes (hedge mode, subcontas, netting com reconciliação).

O que muda em relação ao código de hoje
----------------------------------------
`fsim_autopilot_open.py` já bloqueia por ticker — **por acidente, não por
desenho** (`skip = open_tickers | pending_tickers`). A trava passa a ter **dono**
e **liberação** explícitos: registra qual motor a detém e cai no fechamento,
liberando o ativo para QUALQUER motor no tick seguinte.

**Ordem pendente também trava, e por isso precisa de EXPIRAÇÃO** — uma ordem que
nunca preenche não pode sequestrar o ativo para sempre.

A consequência que o ADR manda MEDIR, não assumir
--------------------------------------------------
Com cadências diferentes, **o motor mais rápido ganha o ativo**: o scalp avalia a
cada minuto e o position uma vez por dia, então na prática o scalp trava BTC
antes de o position sequer olhar. Isso não invalida a regra (a alternativa física
é pior), mas **o custo de oportunidade vira métrica obrigatória** — quantos
sinais de cada motor morreram por ativo travado por outro.

Só depois de medir é que se discute prioridade por motor. Inventar regra de
prioridade antes de ter o número seria complexidade não medida, exatamente o que
o ADR 0032 existe para evitar.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta

__all__ = [
    "CustoDeOportunidade",
    "Trava",
    "TravasPorAtivo",
]


@dataclass(frozen=True)
class Trava:
    """Quem detém o ativo, desde quando, e até quando a ordem pendente vale."""

    ativo: str
    motor: str
    """O **dono**. Era isto que faltava no bloqueio por acidente de hoje."""

    oportunidade_id: str
    desde: datetime
    expira_em: datetime | None = None
    """Só para ordem PENDENTE. `None` = posição aberta, que não expira — ela sai
    quando fecha, e o fechamento tem motivo declarado."""

    def expirada(self, agora: datetime) -> bool:
        return self.expira_em is not None and agora >= self.expira_em


@dataclass
class CustoDeOportunidade:
    """
    Quantos sinais de cada motor morreram por ativo travado por outro.

    **Métrica obrigatória** (§8.3), não telemetria opcional: é o número que
    autoriza — ou não — discutir prioridade por motor depois. Sem ele, qualquer
    regra de prioridade seria complexidade não medida.
    """

    bloqueios: Counter[tuple[str, str]] = field(default_factory=Counter)
    """`(motor_bloqueado, motor_dono) -> quantidade`."""

    def registrar(self, *, bloqueado: str, dono: str) -> None:
        self.bloqueios[(bloqueado, dono)] += 1

    @property
    def total(self) -> int:
        return sum(self.bloqueios.values())

    def por_motor_bloqueado(self) -> dict[str, int]:
        """Quem mais perde sinal — o candidato a ser prejudicado pela cadência."""
        r: Counter[str] = Counter()
        for (bloqueado, _), n in self.bloqueios.items():
            r[bloqueado] += n
        return dict(r)

    def por_motor_dono(self) -> dict[str, int]:
        """Quem mais segura ativo — o motor mais rápido tende a liderar aqui."""
        r: Counter[str] = Counter()
        for (_, dono), n in self.bloqueios.items():
            r[dono] += n
        return dict(r)

    def to_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "por_motor_bloqueado": self.por_motor_bloqueado(),
            "por_motor_dono": self.por_motor_dono(),
            "pares": {f"{b}<-{d}": n for (b, d), n in self.bloqueios.items()},
        }


class TravasPorAtivo:
    """
    O registro de travas de UMA conta.

    Puro e sem I/O — quem persiste é a borda. O estado é pequeno por construção:
    no máximo uma trava por ativo, que é a regra inteira.
    """

    def __init__(self, custo: CustoDeOportunidade | None = None) -> None:
        self._por_ativo: dict[str, Trava] = {}
        self.custo = custo if custo is not None else CustoDeOportunidade()

    def _limpar_expiradas(self, agora: datetime) -> None:
        """Ordem pendente que expirou libera o ativo — senão sequestra sempre."""
        for ativo, t in list(self._por_ativo.items()):
            if t.expirada(agora):
                del self._por_ativo[ativo]

    def dono(self, ativo: str, *, agora: datetime) -> str | None:
        self._limpar_expiradas(agora)
        t = self._por_ativo.get(ativo)
        return t.motor if t else None

    def pode_operar(self, ativo: str, motor: str, *, agora: datetime) -> tuple[bool, str]:
        """
        Autoriza `motor` a abrir em `ativo`.

        **O próprio dono continua podendo** — reforço e reentrada são decisões
        da oportunidade, não da trava. A trava existe para impedir que OUTRO
        motor entre, não para congelar o ativo.
        """
        self._limpar_expiradas(agora)
        t = self._por_ativo.get(ativo)
        if t is None:
            return True, "livre"
        if t.motor == motor:
            return True, "já é o dono"
        self.custo.registrar(bloqueado=motor, dono=t.motor)
        return False, f"travado por {t.motor} desde {t.desde.isoformat()}"

    def travar(
        self,
        *,
        ativo: str,
        motor: str,
        oportunidade_id: str,
        agora: datetime,
        expira_apos: timedelta | None = None,
    ) -> Trava:
        """
        Registra a trava. Ativo já travado por OUTRO motor **levanta**.

        Levanta em vez de devolver `False` porque chegar aqui sem passar por
        `pode_operar` é erro de programação, não estado do mundo — e um retorno
        silencioso deixaria o chamador achar que travou.
        """
        self._limpar_expiradas(agora)
        atual = self._por_ativo.get(ativo)
        if atual is not None and atual.motor != motor:
            raise ValueError(
                f"{ativo} já travado por {atual.motor}; chame pode_operar antes"
            )
        t = Trava(
            ativo=ativo, motor=motor, oportunidade_id=oportunidade_id, desde=agora,
            expira_em=None if expira_apos is None else agora + expira_apos,
        )
        self._por_ativo[ativo] = t
        return t

    def liberar(self, ativo: str) -> None:
        """Libera para QUALQUER motor no tick seguinte. Ativo livre é no-op —
        fechar duas vezes não pode quebrar o fluxo de fechamento."""
        self._por_ativo.pop(ativo, None)

    def promover_para_posicao(self, ativo: str) -> None:
        """Ordem preencheu: a trava perde a expiração.

        Posição aberta não expira — ela sai quando fecha, e o fechamento tem
        motivo declarado (`Oportunidade.com_estado`)."""
        t = self._por_ativo.get(ativo)
        if t is not None:
            self._por_ativo[ativo] = Trava(
                ativo=t.ativo, motor=t.motor, oportunidade_id=t.oportunidade_id,
                desde=t.desde, expira_em=None,
            )

    @property
    def ativos_travados(self) -> set[str]:
        return set(self._por_ativo)

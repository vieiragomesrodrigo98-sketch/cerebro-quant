"""
Camada de execução — RiskManager: o kill switch (G4 do ADR 0014).

Motor puro — sem I/O, no mesmo espírito do `StopEngine`
(`radar.engines.stop_engine`): recebe o estado atual da conta e devolve uma
decisão auditável. A persistência do estado (`RiskState`) é responsabilidade
de quem chama — ver `RiskStateTable` em `radar.database.tables` — porque uma
trava que só existe em memória não sobrevive a um restart do processo, e
"não religar sozinho" só é garantia real se a trava sobreviver a ele.

Por que isto existe
--------------------
Sem limite de perda e sem circuito de parada não existe "robô autônomo" —
existe robô solto. Knight Capital perdeu USD 440M em 30 minutos por não ter
isso (ADR 0001 §2). Os números abaixo são o perfil MODERADO, decisão
explícita do DEV (S144) — constantes auditáveis, não escondidas em código
disperso pelos engines.

O que uma trava FAZ
--------------------
Ao estourar `perda_dia_max_pct` ou `perda_total_max_pct`: `avaliar()` marca a
conta como travada e sinaliza `fechar_tudo=True` — quem chama é responsável
por de fato fechar as posições a mercado (este módulo não toca broker).

Só existem DUAS saídas de uma trava, e nenhuma é automática no sentido de "um
timer disparou": (1) uma trava por PERDA DIÁRIA se autolimpa dentro de
`avaliar()` quando um novo dia útil começa — é o "trava 24h" do perfil,
embutido na própria régua, não um relógio rodando em segundo plano; (2) uma
trava por PERDA TOTAL não tem prazo — permanece travada por qualquer número
de dias até `destravar()` ser chamado, que exige `por=` não vazio (identificar
QUEM destravou) e não é chamado por nenhum outro método deste módulo.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum


# ── Perfil MODERADO — decisão do DEV, S144. Capital inicial fica de fora de
# propósito: não é um número medido nem uma constante de design, é uma
# decisão financeira do DEV que ainda está pendente (ver plano
# elegant-herding-wilkes.md). RiskState.capital_inicial é campo OBRIGATÓRIO
# — não há default aqui — para que essa pendência não vire um valor chutado.
@dataclass(frozen=True)
class RiskLimits:
    risco_por_trade_pct: Decimal = Decimal("0.015")     # 1,5% do capital por trade
    perda_dia_max_pct: Decimal = Decimal("0.03")        # 3% -> fecha tudo, trava até o próximo dia
    perda_total_max_pct: Decimal = Decimal("0.15")      # 15% -> fecha tudo, TRAVA (sem prazo)
    posicoes_max: int = 5                               # posições simultâneas
    exposicao_max_pct: Decimal = Decimal("0.80")        # 80% do capital em risco ao mesmo tempo
    # Concentração setorial (Manual, cap. 73). None = sem teto além de
    # `posicoes_max` — um número aqui seria inventado sem medição (ADR 0020
    # Trava 1); fica explícito como pendência, não como suposição silenciosa.
    posicoes_max_por_setor: int | None = None

    def capital_maximo_da_ordem(
        self, *, equity: Decimal, exposicao_atual: Decimal
    ) -> Decimal:
        """Quanto capital a PRÓXIMA ordem pode usar sem estourar a exposição.

        O buraco que isto fecha (`FSIM_SCALP_CAP_POSICAO01`, P0)
        ---------------------------------------------------------
        `exposicao_max_pct` já dizia "no máximo 80% do capital em risco ao mesmo
        tempo", e `pode_abrir_posicao` já o checava. Só que o checava **antes** de
        a posição existir: com a carteira vazia a exposição atual é 0, o teste
        passa, e nada mais olha o tamanho da ordem que vem em seguida. O autopilot
        pedia `max_capital=balance` — 100% do saldo —, e foi assim que uma posição
        de scalping consumiu a carteira inteira, violando um limite que o sistema
        acreditava estar respeitando.

        A trava aqui é **prospectiva**: olha a exposição que a conta terá DEPOIS,
        não a que ela tem agora.

        O que este método deliberadamente NÃO faz
        ------------------------------------------
        Não impõe teto POR posição. A primeira versão dividia
        `exposicao_max_pct / posicoes_max` (0,80/5 = 16%) e chamava isso de
        "derivado" — não é: os dois limites dizem que a SOMA é ≤ 80% e que há no
        máximo 5, não que cada uma valha um quinto. Uma única posição de 80%
        respeita as duas regras declaradas. Um teto de concentração por posição é
        política nova, que ninguém declarou, e inventá-la aqui seria a Trava 1 do
        ADR 0020 — número sem medição — disfarçada de aritmética.

        O teste que expôs isso: com stop de 2% e risco de 2%, o sizing já pede
        100% do saldo (`risco/distância` = 1), então um teto de 16% estrangularia
        **todo** trade do sistema, não só o scalping do incidente.
        """
        teto = equity * self.exposicao_max_pct - exposicao_atual
        return teto if teto > 0 else Decimal("0")


DEFAULT_LIMITS = RiskLimits()


class MotivoTrava(str, Enum):
    PERDA_DIARIA = "perda_diaria_maxima"
    PERDA_TOTAL = "perda_total_maxima"


@dataclass(frozen=True)
class RiskState:
    """
    Estado que PRECISA sobreviver ao processo — persistido via `RiskStateTable`.

    `capital_inicial` é o teto de referência de `perda_total_max_pct` — o
    capital com que a conta começou, nunca reajustado por aporte/saque sem
    decisão explícita registrada em outro lugar.
    """

    account_id: str
    capital_inicial: Decimal
    equity_inicio_dia: Decimal
    data_inicio_dia: date
    travado: bool = False
    motivo_trava: MotivoTrava | None = None
    travado_em: datetime | None = None
    destravado_em: datetime | None = None
    destravado_por: str | None = None

    def __post_init__(self) -> None:
        if self.capital_inicial <= 0:
            raise ValueError(
                "RiskState.capital_inicial precisa ser > 0 — sem capital "
                "definido o RiskManager não pode calcular nenhum limite"
            )


@dataclass(frozen=True)
class RiskDecision:
    """Veredito de `avaliar()`. `novo_estado` é o que quem chama deve persistir."""

    pode_operar: bool
    motivo: str
    novo_estado: RiskState
    fechar_tudo: bool = False


class RiskManager:
    """O kill switch. Ver docstring do módulo."""

    def __init__(self, limits: RiskLimits = DEFAULT_LIMITS) -> None:
        self.limits = limits

    # ── Avaliação por ciclo ──────────────────────────────────────────────────

    def avaliar(
        self,
        state: RiskState,
        equity_atual: Decimal,
        *,
        hoje: date | None = None,
    ) -> RiskDecision:
        """
        Avalia o estado da conta AGORA. Chamar antes de toda ordem e a cada
        ciclo de monitoramento — é essa checagem repetida que faz o kill
        switch valer mesmo entre ordens, não só no instante de abertura.

        Se já travada, `pode_operar` é sempre False — mesmo que o equity
        tenha se recuperado. Recuperação de equity não destrava: só
        `destravar()`, chamado por um humano, destrava.
        """
        hoje = hoje or datetime.now(tz=UTC).date()

        # Novo dia útil = novo teto de referência para a perda DIÁRIA. A trava
        # por perda DIÁRIA tem prazo embutido nessa própria régua — o início
        # de um novo dia JÁ É o "trava 24h" do perfil — e por isso se
        # autolimpa aqui. A trava por perda TOTAL não tem prazo nenhum: só
        # sai por `destravar()`, chamado por um humano.
        if state.data_inicio_dia != hoje:
            state = replace(state, equity_inicio_dia=equity_atual, data_inicio_dia=hoje)
            if state.travado and state.motivo_trava == MotivoTrava.PERDA_DIARIA:
                state = replace(state, travado=False, motivo_trava=None, travado_em=None)

        if state.travado:
            # Invariante deste módulo: toda transição para travado=True (abaixo,
            # e em `avaliar` de uma chamada anterior) grava motivo_trava junto.
            assert state.motivo_trava is not None, "travado=True sem motivo_trava"
            return RiskDecision(False, f"conta travada: {state.motivo_trava.value}", state)

        perda_total_pct = (state.capital_inicial - equity_atual) / state.capital_inicial
        if perda_total_pct >= self.limits.perda_total_max_pct:
            travado = replace(
                state, travado=True, motivo_trava=MotivoTrava.PERDA_TOTAL,
                travado_em=datetime.now(tz=UTC),
            )
            return RiskDecision(
                False,
                f"perda total {perda_total_pct:.1%} >= limite "
                f"{self.limits.perda_total_max_pct:.0%} — CONTA TRAVADA",
                travado, fechar_tudo=True,
            )

        perda_dia_pct = (state.equity_inicio_dia - equity_atual) / state.equity_inicio_dia
        if perda_dia_pct >= self.limits.perda_dia_max_pct:
            travado = replace(
                state, travado=True, motivo_trava=MotivoTrava.PERDA_DIARIA,
                travado_em=datetime.now(tz=UTC),
            )
            return RiskDecision(
                False,
                f"perda diária {perda_dia_pct:.1%} >= limite "
                f"{self.limits.perda_dia_max_pct:.0%} — travada até o próximo dia",
                travado, fechar_tudo=True,
            )

        return RiskDecision(True, "ok", state)

    def destravar(self, state: RiskState, *, por: str) -> RiskState:
        """
        A ÚNICA forma de sair de uma trava. Não é chamada por nenhum outro
        método deste módulo — é sempre uma ação humana explícita.
        """
        if not por or not por.strip():
            raise ValueError(
                "destravar() exige identificar QUEM destravou — uma trava nunca "
                "sai sozinha, nem por engano de chamada automática"
            )
        return replace(
            state, travado=False, motivo_trava=None, travado_em=None,
            destravado_em=datetime.now(tz=UTC), destravado_por=por,
        )

    # ── Checagem por ORDEM (além da checagem por ciclo em `avaliar`) ─────────

    def pode_abrir_posicao(
        self,
        state: RiskState,
        *,
        equity_atual: Decimal,
        exposicao_atual: Decimal,
        posicoes_abertas: int,
        setor: str | None = None,
        posicoes_por_setor: dict[str, int] | None = None,
    ) -> tuple[bool, str]:
        """
        Autoriza (ou não) abrir UMA posição nova. `avaliar()` decide se a
        conta pode operar agora; esta função decide se ESTA ordem específica
        cabe dentro dos limites de concentração.
        """
        if state.travado:
            assert state.motivo_trava is not None, "travado=True sem motivo_trava"
            return False, f"conta travada: {state.motivo_trava.value}"

        if posicoes_abertas >= self.limits.posicoes_max:
            return False, f"limite de {self.limits.posicoes_max} posições simultâneas atingido"

        exposicao_max = equity_atual * self.limits.exposicao_max_pct
        if exposicao_atual >= exposicao_max:
            return False, (
                f"exposição atual ({exposicao_atual:.2f}) já no teto de "
                f"{self.limits.exposicao_max_pct:.0%} do capital"
            )

        if (
            self.limits.posicoes_max_por_setor is not None
            and setor is not None
            and (posicoes_por_setor or {}).get(setor, 0) >= self.limits.posicoes_max_por_setor
        ):
            return False, f"limite de concentração no setor '{setor}' atingido"

        return True, "ok"

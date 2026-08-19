"""
src/radar/cerebro/diagnostico.py — a árvore de causas do fracasso, mecanizada.

Por que este módulo existe
---------------------------
Ordem do DEV (2026-08-04):

    Não devemos matar uma estratégia simplesmente porque uma medição ruim não
    encontrou resultado. Porém, também não devemos manter uma estratégia
    indefinidamente porque acreditamos que ela deveria funcionar.

As duas metades são igualmente importantes, e a segunda é a que dá dentes à
primeira. Sem ela, "não funcionou → deve ser o instrumento / os dados / o
custo" vira uma máquina de nunca aceitar um negativo — garimpo com passos
extras, só que agora com vocabulário científico. **Este módulo existe tanto
para permitir a reabertura quanto para limitá-la.**

A árvore, na ordem em que é percorrida
---------------------------------------
A ordem NÃO é arbitrária: vai do defeito mais nosso ao defeito menos nosso.
Cada ramo só é alcançado se todos os anteriores foram descartados, e cada um
tem um **predicado mecânico** computável a partir da própria rodada — nunca de
uma opinião posterior ao resultado.

    não funcionou
        ├── instrumento quebrado?   → REABRIR          (o portão do Contrato falhou)
        ├── amostra insuficiente?   → NÃO ESTIMÁVEL    (a estatística não existe)
        ├── custo mal modelado?     → RECALCULAR       (só dentro da banda declarada)
        ├── alvo errado?            → REFORMULAR       (aprendeu o rótulo, o rótulo não paga)
        ├── dados insuficientes?    → MELHORAR DADOS   (o mesmo instrumento aprende noutro alvo)
        └── tudo correto            → EVIDÊNCIA CONTRA A HIPÓTESE

Só o último ramo é **evidência**. Os cinco primeiros são diagnósticos sobre
NÓS, não sobre o mercado, e nenhum deles autoriza concluir coisa alguma sobre a
tese — nem a favor, nem contra.

As três travas contra a reabertura infinita
--------------------------------------------
1. **Predicado antes do resultado.** Cada ramo exige um número que a rodada
   produziu (`dias`, `num_trees`, `custo_aplicado`, `ic_vs_rotulo`), nunca uma
   narrativa. `custo_mal_modelado` em particular só dispara se o custo aplicado
   ficou FORA da banda que o pré-registro declarou ANTES de medir — sem isso,
   "o custo estava alto demais" é ajustável até o resultado virar.
2. **Uma reabertura por causa.** A mesma célula não pode ser reaberta duas
   vezes pela mesma causa sem ADR novo (`MAX_REABERTURAS_POR_CAUSA`). Se
   consertar o instrumento e ainda falhar por instrumento, o problema não é o
   instrumento.
3. **Reabertura conta no denominador.** Toda re-medição é um teste novo na
   família do FDR. Reabrir é barato em código e caro em multiplicidade — que é
   exatamente o incentivo correto.

Instância real que motivou cada ramo (nenhum é hipotético)
-----------------------------------------------------------
* `INSTRUMENTO_QUEBRADO` — os 6 modelos do Cérebro v1 com `num_trees=1`.
* `AMOSTRA_INSUFICIENTE` — `p9c57_triangulo_saida`: metade das células verdes
  do projeto, **11 dias distintos** no walk-forward. Nunca foi refutada.
* `ALVO_ERRADO` — `swing_v1`: Rank IC contra o próprio rótulo **+0,106
  (t +28,2)** e mesmo assim perde dinheiro. O modelo acertou o alvo; o alvo é
  que premiava volatilidade.
* `DADOS_INSUFICIENTES` — `autoref_v1`: 1 a 4 árvores com alvo absoluto, 17 a
  134 com alvo relativo, **mesmo código**. As features carregam posição
  relativa, não direção absoluta.
* `EVIDENCIA_CONTRA` — `p9c57_squeeze`: 15.483 trades, 1.464 dias distintos,
  custo varrido, 19 anos de walk-forward → **+0,044%**. Este está refutado.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

MAX_REABERTURAS_POR_CAUSA: Final[int] = 1
"""
Quantas vezes a MESMA célula pode ser reaberta pela MESMA causa sem ADR novo.

Um é o número certo: consertar o instrumento e falhar de novo por instrumento
significa que a causa foi diagnosticada errado, não que falta mais um conserto.
"""


class Causa(str, Enum):
    """Os seis desfechos possíveis. Só o último é evidência sobre o mercado."""

    INSTRUMENTO_QUEBRADO = "instrumento_quebrado"
    AMOSTRA_INSUFICIENTE = "amostra_insuficiente"
    CUSTO_MAL_MODELADO = "custo_mal_modelado"
    ALVO_ERRADO = "alvo_errado"
    DADOS_INSUFICIENTES = "dados_insuficientes"
    EVIDENCIA_CONTRA = "evidencia_contra"

    @property
    def acao(self) -> str:
        return _ACAO[self]

    @property
    def e_evidencia(self) -> bool:
        """
        `True` só para `EVIDENCIA_CONTRA`.

        É a pergunta que o resto do sistema deve fazer antes de escrever que uma
        tese morreu. Cinco dos seis desfechos respondem `False` — e escrever
        "morreu" em cima de qualquer um deles é o erro que
        `feedback_morrer_de_certeza` descreve.
        """
        return self is Causa.EVIDENCIA_CONTRA


_ACAO: Final[dict[Causa, str]] = {
    Causa.INSTRUMENTO_QUEBRADO: "reabrir — consertar o instrumento e re-medir",
    Causa.AMOSTRA_INSUFICIENTE: "não estimável — acumular amostra, não concluir",
    Causa.CUSTO_MAL_MODELADO: "recalcular com o custo dentro da banda declarada",
    Causa.ALVO_ERRADO: "reformular o alvo — o modelo aprendeu, o rótulo não paga",
    Causa.DADOS_INSUFICIENTES: "melhorar os dados — estes não contêm o fenômeno",
    Causa.EVIDENCIA_CONTRA: "registrar: evidência contra a hipótese",
}


@dataclass(frozen=True)
class Rodada:
    """
    O que uma rodada precisa ter produzido para ser diagnosticada.

    Todo campo é um número ou booleano que a rodada JÁ gerou. Não há campo de
    texto livre e isso é deliberado: o diagnóstico não pode depender de como a
    rodada foi narrada depois de o resultado aparecer.
    """

    celula: str

    # --- ramo 1: instrumento (Contrato de Pensamento) -----------------------
    portoes_falhos: tuple[str, ...] = ()
    """G-P0..G-P4 que falharam. Não-vazio ⇒ o instrumento não está apto."""

    # --- ramo 2: amostra ----------------------------------------------------
    dias_distintos: int = 0
    n_eventos: int = 0
    t_definido: bool = True
    """`False` quando o `t` saiu `NaN` (variância nula, série curta demais)."""

    # --- ramo 3: custo ------------------------------------------------------
    excesso_bruto: float = 0.0
    excesso_liquido: float = 0.0
    custo_aplicado_bps_lado: float | None = None
    banda_custo_declarada_bps: tuple[float, float] | None = None
    """Banda `(min, max)` congelada no pré-registro ANTES de medir. `None`
    significa que ninguém declarou banda — e então este ramo **não pode**
    disparar, porque não há como distinguir "custo errado" de "custo que eu
    gostaria que fosse menor"."""

    # --- ramo 4: alvo -------------------------------------------------------
    t_ic_vs_rotulo: float | None = None
    """`t` do Rank IC da predição contra o PRÓPRIO rótulo. Alto + dinheiro
    negativo é a assinatura de alvo errado: o modelo acertou o que foi mandado
    acertar, e acertar aquilo não paga."""

    # --- ramo 5: dados ------------------------------------------------------
    instrumento_aprendeu_noutro_alvo: bool = False
    """`True` quando o MESMO instrumento produziu modelo saudável em outra
    célula/alvo. Só então a degeneração acusa os DADOS em vez da ferramenta —
    e é o que separa `autoref_v1` de um bug."""

    # --- controle de reabertura --------------------------------------------
    reaberturas_por_causa: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Diagnostico:
    causa: Causa
    acao: str
    justificativa: str
    pode_reabrir: bool
    """`False` quando a causa é terminal OU quando a cota de reaberturas
    daquela causa já foi gasta — e nesse segundo caso o resultado passa a valer
    como evidência, com o motivo dito em `justificativa`."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "causa": self.causa.value,
            "acao": self.acao,
            "justificativa": self.justificativa,
            "pode_reabrir": self.pode_reabrir,
            "e_evidencia": self.causa.e_evidencia,
        }


def diagnosticar(
    r: Rodada,
    *,
    min_dias_distintos: int,
    min_eventos: int,
    t_ic_forte: float = 3.0,
) -> Diagnostico:
    """
    Percorre a árvore na ordem e devolve a PRIMEIRA causa que se aplica.

    `min_dias_distintos`/`min_eventos` são passados explicitamente (e não lidos
    de `radar.evidence.ledger`) para que este módulo não force uma dependência
    de importação sobre o Ledger e para que o chamador declare qual régua está
    usando — a régua é parte do pré-registro, não um default escondido aqui.

    A cota de reabertura é verificada DEPOIS de a causa ser identificada: uma
    causa não terminal cuja cota acabou não vira outra causa — ela permanece
    diagnosticada corretamente, mas deixa de autorizar nova rodada. Trocar a
    causa pela cota apagaria a informação de por que falhou.
    """
    causa, justificativa = _classificar(
        r,
        min_dias_distintos=min_dias_distintos,
        min_eventos=min_eventos,
        t_ic_forte=t_ic_forte,
    )

    if causa is Causa.EVIDENCIA_CONTRA:
        return Diagnostico(causa, causa.acao, justificativa, pode_reabrir=False)

    gastas = int(r.reaberturas_por_causa.get(causa.value, 0))
    if gastas >= MAX_REABERTURAS_POR_CAUSA:
        return Diagnostico(
            causa,
            "cota de reabertura esgotada — o resultado vale como evidência",
            f"{justificativa} — mas esta célula já foi reaberta {gastas}× por "
            f"'{causa.value}' (máx. {MAX_REABERTURAS_POR_CAUSA}). Reabrir de novo "
            "pela mesma causa exige ADR novo: se o conserto não consertou, a "
            "causa foi diagnosticada errado.",
            pode_reabrir=False,
        )

    return Diagnostico(causa, causa.acao, justificativa, pode_reabrir=True)


def _classificar(
    r: Rodada, *, min_dias_distintos: int, min_eventos: int, t_ic_forte: float
) -> tuple[Causa, str]:
    # 1 — instrumento. Primeiro porque um instrumento quebrado torna todo o
    # resto não-interpretável: não dá para acusar os dados com a ferramenta
    # torta.
    if r.portoes_falhos:
        return Causa.INSTRUMENTO_QUEBRADO, (
            f"portões do Contrato de Pensamento falharam: {', '.join(r.portoes_falhos)}"
        )

    # 2 — amostra. Antes de custo e alvo porque sem estatística não há o que
    # interpretar: `nao_estimavel` do Ledger é exatamente este ramo.
    if r.n_eventos < min_eventos:
        return Causa.AMOSTRA_INSUFICIENTE, (
            f"N={r.n_eventos} < {min_eventos} — nenhum teste aconteceu"
        )
    if r.dias_distintos < min_dias_distintos:
        return Causa.AMOSTRA_INSUFICIENTE, (
            f"{r.dias_distintos} dias distintos < {min_dias_distintos} — o `t` de "
            "uma amostra assim é ficção (clustering de dias)"
        )
    if not r.t_definido:
        return Causa.AMOSTRA_INSUFICIENTE, "`t` indefinido — a estatística não existe"

    # 3 — custo. Só dispara com banda DECLARADA e custo FORA dela. Sem banda,
    # este ramo é desligado de propósito.
    if (
        r.excesso_bruto > 0
        and r.excesso_liquido <= 0
        and r.banda_custo_declarada_bps is not None
        and r.custo_aplicado_bps_lado is not None
    ):
        lo, hi = r.banda_custo_declarada_bps
        if not (lo <= r.custo_aplicado_bps_lado <= hi):
            return Causa.CUSTO_MAL_MODELADO, (
                f"bruto {r.excesso_bruto:+.4%} vira líquido {r.excesso_liquido:+.4%} "
                f"com custo de {r.custo_aplicado_bps_lado:.1f} bps/lado, FORA da banda "
                f"declarada [{lo:.1f}, {hi:.1f}] bps"
            )

    # 4 — alvo. O modelo aprendeu forte o que foi mandado aprender, e aquilo
    # não paga. É o diagnóstico da `swing_v1`.
    if (
        r.t_ic_vs_rotulo is not None
        and r.t_ic_vs_rotulo >= t_ic_forte
        and r.excesso_liquido <= 0
    ):
        return Causa.ALVO_ERRADO, (
            f"o modelo prevê o próprio rótulo com t={r.t_ic_vs_rotulo:+.1f} e ainda "
            f"assim entrega {r.excesso_liquido:+.4%} — quem falhou foi o rótulo, "
            "não o modelo"
        )

    # 5 — dados. Só acusável quando o MESMO instrumento já se provou noutro
    # alvo; caso contrário isto é indistinguível de ferramenta quebrada.
    if r.instrumento_aprendeu_noutro_alvo and (
        r.t_ic_vs_rotulo is not None and abs(r.t_ic_vs_rotulo) < 1.0
    ):
        return Causa.DADOS_INSUFICIENTES, (
            "o mesmo instrumento aprende noutro alvo e aqui não aprende nada "
            f"(t do IC contra o próprio rótulo = {r.t_ic_vs_rotulo:+.2f}) — a "
            "informação não está nestes dados"
        )

    # 6 — terminal. A justificativa só pode afirmar o que foi VERIFICADO.
    #
    # Os ramos 3 (custo) e 4 (alvo) desligam quando o insumo não existe — banda
    # de custo não declarada, `t_ic_vs_rotulo` ausente. Até 2026-08-11 este
    # texto dizia "custo dentro da banda, alvo alinhado" mesmo nesses casos, e
    # a frase é o laudo que autoriza escrever que uma tese morreu. Afirmar que
    # dois ramos passaram quando eles nem foram avaliados é a forma mais cara
    # do erro que este módulo existe para impedir: um `EVIDENCIA_CONTRA` que
    # parece completo e não é. Quem lê o veredito não tem como saber a
    # diferença — então o veredito precisa dizê-la.
    verificado = [
        "instrumento apto",
        f"{r.dias_distintos} dias distintos",
        f"N={r.n_eventos}",
    ]
    nao_avaliado: list[str] = []

    if r.banda_custo_declarada_bps is not None and r.custo_aplicado_bps_lado is not None:
        verificado.append("custo dentro da banda declarada")
    else:
        nao_avaliado.append("custo (nenhuma banda declarada no pré-registro)")

    if r.t_ic_vs_rotulo is not None:
        verificado.append("alvo alinhado")
    else:
        nao_avaliado.append("alvo (IC contra o próprio rótulo não medido)")

    texto = (
        f"{', '.join(verificado)} — e o líquido é {r.excesso_liquido:+.4%}."
    )
    if nao_avaliado:
        return Causa.EVIDENCIA_CONTRA, (
            f"{texto} Ressalva: {' e '.join(nao_avaliado)} — "
            "estes ramos NÃO foram avaliados por falta de insumo, então este "
            "resultado é o mais próximo de evidência que esta rodada permite, "
            "não uma refutação completa."
        )
    return Causa.EVIDENCIA_CONTRA, (
        f"{texto} Todos os ramos foram avaliados. "
        "Este resultado é sobre o mercado, não sobre nós."
    )

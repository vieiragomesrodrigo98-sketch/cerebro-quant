"""
src/radar/cerebro/admissao.py — o portão que decide se uma família merece gastar
orçamento de FDR.

FASE 2 do plano aprovado em 2026-08-13. Complementa `operabilidade.py` sem
duplicá-lo: aquele responde **por ativo** (*"este ativo se move o bastante para
pagar o pedágio neste horizonte?"*); este responde **por família** (*"vale
pré-registrar este horizonte, ou a economia já o reprova?"*).

Por que a pergunta agregada precisa existir
--------------------------------------------
O edge medido do `scalp_cripto_v2` foi `0,0021%/trade` contra `0,04%` de custo —
**5% do pedágio**. E no scalp de cripto, o custo é **296% do movimento mediano de
60 s** do BTC. Nos dois casos a economia reprova *antes* de qualquer modelo.

Sem este portão, cada família nova gasta células do orçamento de FDR para medir
uma hipótese que o custo já matou. O orçamento é **derivado, nunca contador**
(ADR 0037), mas ele é finito: teto 180, e cada célula nula consome espaço que uma
célula viável poderia usar.

O pré-registro do scalp V3 fez isto **à mão**, excluindo horizontes abaixo de
30 s por medição. Este módulo transforma aquele julgamento em portão.

⚠️ O QUE ESTE MÓDULO NÃO FAZ
-----------------------------
Não diz que a família é lucrativa, e a distinção é a mesma de `operabilidade`:
*"existe movimento suficiente para pagar o pedágio"* nunca significa *"consigo
prever a direção dele"*. Admissão é condição **necessária**, jamais suficiente.

E não elimina modelo. Reprovar um HORIZONTE de uma família é diferente de
aposentar o especialista — o ADR 0038 é explícito: *ausência de oportunidade ≠
ausência do modelo*. Um horizonte inadmissível hoje pode virar admissível com
custo menor (tier de taxa) ou venue diferente, e a resposta correta é
`SEM OPORTUNIDADE → CONTINUAR MONITORANDO`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from radar.cerebro.operabilidade import Operabilidade

#: Razão máxima entre custo e movimento típico para o horizonte ser admissível.
#:
#: `1,0` significa "o custo consome exatamente o movimento mediano" — metade das
#: operações perde por construção, antes de qualquer questão de previsão. Acima
#: disso a hipótese não é difícil, é aritmeticamente perdedora.
#:
#: Note que é MAIS FROUXO que a margem de `operabilidade` (2,0), e de propósito:
#: aquele classifica o ativo para OPERAR, este decide se vale MEDIR. Medir um
#: horizonte apertado ainda ensina — operar nele não.
RAZAO_MAXIMA_CUSTO_MOVIMENTO: Final[float] = 1.0


@dataclass(frozen=True, slots=True)
class AdmissaoDeHorizonte:
    """O veredito para UM horizonte de UMA família."""

    horizonte: int
    n_operaveis: int
    n_universo: int
    #: Mínimo de ativos operáveis para o ranking não degenerar. Derivado da
    #: `fracao_gate` da própria família — ver `pool_minimo`.
    pool_minimo: int
    #: `custo / movimento típico`, mediana sobre o UNIVERSO INTEIRO (não só
    #: sobre os operáveis — isso seria viés de sobrevivência). `inf` quando
    #: ninguém tem medida — ausência nunca vira benefício da dúvida.
    razao_custo_movimento: float

    @property
    def fracao_do_universo(self) -> float:
        return self.n_operaveis / self.n_universo if self.n_universo else 0.0

    @property
    def pool_suficiente(self) -> bool:
        """O Top-N precisa SELECIONAR, não carimbar.

        Com `fracao_gate=0,10` e 4 ativos operáveis, o topo de 10% é 1 de 4 —
        que é 25%, não 10%. A frequência efetiva de sinal deixa de ser a
        nominal, e o ranking vira um sorteio com nome melhor.
        """
        return self.n_operaveis >= self.pool_minimo

    @property
    def economia_paga(self) -> bool:
        """O ativo TÍPICO do universo paga o pedágio neste horizonte?

        Mediana sobre o universo inteiro, não sobre os operáveis — ver a nota em
        `avaliar_horizonte`. Medir só os aprovados responderia "os aprovados
        pagam?", que é tautologia.
        """
        return self.razao_custo_movimento <= RAZAO_MAXIMA_CUSTO_MOVIMENTO

    @property
    def admissivel(self) -> bool:
        """**Falha-fechado**: precisa das DUAS pernas."""
        return self.pool_suficiente and self.economia_paga

    @property
    def motivo(self) -> str:
        """Por que reprovou — vazio quando passou.

        Reprovação sem motivo legível não deixa ninguém agir, e é a diferença
        entre um portão e um muro.
        """
        if self.admissivel:
            return ""
        partes = []
        if not self.pool_suficiente:
            partes.append(
                f"pool de {self.n_operaveis} operável(is) abaixo do mínimo "
                f"{self.pool_minimo} — o Top-N não selecionaria, carimbaria"
            )
        if not self.economia_paga:
            razao = self.razao_custo_movimento
            legivel = "sem movimento medido" if math.isinf(razao) else f"{razao:.0%}"
            partes.append(
                f"custo é {legivel} do movimento típico — acima de "
                f"{RAZAO_MAXIMA_CUSTO_MOVIMENTO:.0%}, metade das operações perde "
                "por construção"
            )
        return " · ".join(partes)


def pool_minimo_para(fracao_gate: float) -> int:
    """Menor pool em que o Top-N ainda é seleção.

    `ceil(1/fracao)`: com 10% é preciso 10 candidatos para o topo ser 1; com 4,
    o "topo de 10%" é 25% do pool. O piso de 1 selecionado de
    `selecionar_percentil_instante` garante que o ranking devolva algo — este
    portão garante que o que ele devolve signifique alguma coisa.
    """
    if not 0.0 < fracao_gate <= 1.0:
        raise ValueError(f"fracao_gate fora de (0, 1]: {fracao_gate!r}")
    return math.ceil(1.0 / fracao_gate)


def _mediana(valores: list[float]) -> float:
    if not valores:
        return float("inf")
    ordenado = sorted(valores)
    meio = len(ordenado) // 2
    if len(ordenado) % 2:
        return ordenado[meio]
    return (ordenado[meio - 1] + ordenado[meio]) / 2.0


def avaliar_horizonte(
    classificacao: dict[str, Operabilidade],
    horizonte: int,
    *,
    fracao_gate: float,
) -> AdmissaoDeHorizonte:
    """Veredito de admissão para um horizonte, a partir da classificação por ativo.

    Consome `operabilidade.classificar_operabilidade` em vez de recalcular
    movimento e custo: duas fontes para o mesmo número divergem no dia em que
    uma delas muda, e aí nenhuma das duas é confiável.
    """
    universo = len(classificacao)
    operaveis = [o for o in classificacao.values() if horizonte in o.horizontes_operaveis]

    # ⚠️ A razão é medida sobre o UNIVERSO INTEIRO, nunca sobre os operáveis —
    # e a primeira versão deste módulo fazia o contrário, o que o tornava
    # decorativo. Achado por auditoria adversarial em 2026-08-13.
    #
    # A aritmética que o denuncia: `classificar_operabilidade` só marca operável
    # quando `movimento > margem × custo` (margem 2,0), ou seja `custo/mov <
    # 0,5` por construção. Medindo a mediana só entre os aprovados, o valor
    # NUNCA alcançava o limiar de 1,0 — a perna econômica não podia reprovar em
    # nenhum dado real, e o portão era 100% `pool_suficiente` fingindo ser dois.
    #
    # É viés de sobrevivência na forma mais direta: condicionar na aprovação e
    # depois medir o aprovado. A pergunta agregada certa é sobre o ativo TÍPICO
    # do universo — se o ativo mediano não paga o pedágio naquele horizonte, o
    # horizonte é ruim ainda que um punhado de ativos o vença.
    razoes = [
        o.custo_roundtrip / mov
        for o in classificacao.values()
        if math.isfinite(o.custo_roundtrip)
        and (mov := o.movimento_por_horizonte.get(horizonte, 0.0)) > 0
    ]
    return AdmissaoDeHorizonte(
        horizonte=horizonte,
        n_operaveis=len(operaveis),
        n_universo=universo,
        pool_minimo=pool_minimo_para(fracao_gate),
        razao_custo_movimento=_mediana(razoes),
    )


def avaliar_familia(
    classificacao: dict[str, Operabilidade],
    horizontes: tuple[int, ...],
    *,
    fracao_gate: float,
) -> dict[int, AdmissaoDeHorizonte]:
    """Um veredito por horizonte declarado.

    **Não devolve um booleano para a família inteira** — de propósito. Uma
    família com 3 horizontes em que 1 reprova não é uma família reprovada: é uma
    família que pré-registra 2 células em vez de 3, e o horizonte reprovado
    volta a ser avaliado quando o custo mudar. Colapsar isso num booleano
    apagaria justamente a informação que decide o quanto se gasta.
    """
    return {h: avaliar_horizonte(classificacao, h, fracao_gate=fracao_gate) for h in horizontes}


def celulas_admitidas(vereditos: dict[int, AdmissaoDeHorizonte]) -> int:
    """Quantas células a família pode pré-registrar — o custo real em FDR."""
    return sum(1 for v in vereditos.values() if v.admissivel)

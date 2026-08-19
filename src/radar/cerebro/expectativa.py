"""
src/radar/cerebro/expectativa.py — o estágio EXPECTANCY ENGINE.

Pergunta única: **"essa oportunidade tem vantagem?"**

⚠️ NUNCA AVALIAR SÓ PELA PROBABILIDADE DE ACERTO
------------------------------------------------
Está na skill `raciocinio-quant`, e é o erro mais comum de quem vem de
classificação para trading: um setup com 70% de acerto e payoff assimétrico
contra perde dinheiro; um com 35% e payoff a favor ganha.

    EV = P(ganho) × payoff_ganho − P(perda) × payoff_perda − custos

O `− custos` não é rodapé — é a lente do corretor, a que as outras esquecem.
Este projeto já mediu o preço de ignorá-la: o custo do motor anterior era
**uniforme**, devolvia a mesma fração para BTC e para uma altcoin de ponta de
universo, e isso **apagava a única diferença entre os dois braços de alvo** da
família `swing_v1`. `cerebro/custos.py` é round-trip **por ativo** desde então,
e é ele que este módulo consome.

O que este módulo NÃO faz
-------------------------
Não decide (é `decisao.py`), não ordena (é `ranking.py`) e **não estima
probabilidade** — a probabilidade entra por parâmetro, vinda de quem tem
evidência para produzi-la. Estimar aqui misturaria modelo com contabilidade, e
a autópsia do v1 mostrou o preço de estágios fundidos.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

#: Margem mínima sobre o custo para a expectativa ser considerada
#: economicamente distinguível de zero. Não é gate (quem decide é
#: `decisao.py`) — é o piso abaixo do qual o número está dentro do ruído do
#: próprio modelo de custo, e tratá-lo como vantagem seria precisão falsa.
MARGEM_MINIMA_SOBRE_CUSTO: Final[float] = 0.5


@dataclass(frozen=True, slots=True)
class Expectativa:
    """A aritmética da vantagem, com as partes VISÍVEIS.

    Publicar as parcelas e não só o `ev` é deliberado: o ADR 0033 fixou que
    nenhuma dimensão publica só o booleano quando o número existe — o gate
    classifica, o número informa. Sem as parcelas, um `ev` negativo não
    distingue "não tem edge" de "tem edge e o custo comeu".
    """

    p_ganho: float
    payoff_ganho: float
    payoff_perda: float
    custo: float

    @property
    def p_perda(self) -> float:
        return 1.0 - self.p_ganho

    @property
    def bruto(self) -> float:
        """EV antes do custo — o que o modelo acha que existe."""
        return self.p_ganho * self.payoff_ganho - self.p_perda * self.payoff_perda

    @property
    def ev(self) -> float:
        """EV líquido. **É este que vale**, e a diferença para `bruto` é
        exatamente o que o corretor cobra."""
        return self.bruto - self.custo

    @property
    def custo_comeu_o_edge(self) -> bool:
        """Havia vantagem bruta e o custo a consumiu. Distinguir isso de
        'não havia vantagem' importa: o primeiro caso melhora com execução e
        seleção de ativo mais líquido; o segundo não melhora com nada."""
        return self.bruto > 0.0 and self.ev <= 0.0

    @property
    def margem_sobre_custo(self) -> float:
        """Quantas vezes o EV bruto cobre o custo. `inf` quando o custo é zero
        — que só acontece em teste, e devolver `inf` é mais honesto que
        devolver um número grande arbitrário."""
        if self.custo <= 0.0:
            return math.inf
        return self.bruto / self.custo

    @property
    def economicamente_distinguivel(self) -> bool:
        return self.ev > 0.0 and self.margem_sobre_custo >= MARGEM_MINIMA_SOBRE_CUSTO


def avaliar(
    *,
    p_ganho: float,
    payoff_ganho: float,
    payoff_perda: float,
    custo_roundtrip: float,
) -> Expectativa:
    """Monta a expectativa a partir dos insumos já produzidos.

    Não recebe o `Candidato` de propósito, e isso foi corrigido na revisão da
    Fase 2: a primeira versão o recebia e o **descartava** (`_ = candidato`).
    Parâmetro que não é usado é acoplamento sem contrapartida — sugere que o
    módulo lê algo do candidato quando não lê, e o próximo a mexer aqui gastaria
    tempo procurando o uso.

    Todos os insumos entram por parâmetro **de propósito**: a probabilidade vem
    do estágio de evidência, e o custo vem de `cerebro.custos.
    custo_roundtrip_da_faixa(faixa, mercado=...)`, que conhece a faixa de
    liquidez do ativo. Este módulo não os produz — ele os combina, e combinar é
    tudo o que ele deve saber fazer.

    Valida as entradas em vez de confiar: `p_ganho` fora de [0,1] e payoff
    negativo são defeito de programação, não dado ruim, e silenciá-los
    produziria um `ev` com cara de medição.
    """
    if not 0.0 <= p_ganho <= 1.0:
        raise ValueError(f"p_ganho={p_ganho} fora de [0,1] — probabilidade não é score")
    if payoff_ganho < 0.0 or payoff_perda < 0.0:
        raise ValueError(
            "payoffs são MAGNITUDES não-negativas; a direção do resultado já está "
            "na fórmula (o de perda entra subtraindo). Payoff negativo aqui "
            "inverteria o sinal duas vezes."
        )
    if custo_roundtrip < 0.0:
        raise ValueError("custo round-trip não pode ser negativo")
    return Expectativa(
        p_ganho=p_ganho,
        payoff_ganho=payoff_ganho,
        payoff_perda=payoff_perda,
        custo=custo_roundtrip,
    )

def de_trades(
    retorno_bruto: Sequence[float], *, custo_roundtrip: float
) -> Expectativa | None:
    """A expectativa MEDIDA a partir dos trades OOS de uma célula.

    Existe porque o ciclo do ADR 0040 parava em `SEM_EXPECTATIVA`: `avaliar()`
    exige `p_ganho` + `payoff_ganho` + `payoff_perda`, e nenhuma família
    publicava os três — a leitura R1 publicava excesso líquido e persistência,
    que são outra coisa. Os três **sempre estiveram** nos trades; o que faltava
    era publicá-los.

    ⚠️ **Recebe o retorno BRUTO, jamais o líquido.** `_validos` traz as duas
    colunas, e `excesso_liquido` já vem com o custo descontado: passá-lo aqui e
    ainda informar `custo_roundtrip` subtrairia o pedágio **duas vezes**, e o
    `ev` sairia pessimista de um jeito que ninguém notaria — o número continua
    plausível, só está errado.

    `None` quando não há trade dos dois lados: com zero ganhos ou zero perdas,
    `payoff_ganho`/`payoff_perda` seriam a média de um conjunto vazio. Devolver
    `0.0` transformaria ausência de amostra em afirmação sobre o payoff.

    ⚠️ **Custo em orçamento de FDR: ZERO.** É telemetria da medição já paga —
    os mesmos trades, contados de outro jeito. Nenhuma hipótese nova entra no
    denominador. Agir sobre o resultado, como sempre, custa.
    """
    valores = [float(x) for x in retorno_bruto if x == x]  # descarta NaN
    ganhos = [v for v in valores if v > 0]
    perdas = [-v for v in valores if v < 0]
    if not ganhos or not perdas:
        return None
    return avaliar(
        p_ganho=len(ganhos) / len(valores),
        payoff_ganho=sum(ganhos) / len(ganhos),
        payoff_perda=sum(perdas) / len(perdas),
        custo_roundtrip=custo_roundtrip,
    )

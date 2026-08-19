"""
src/radar/cerebro/estabilidade.py — estabilidade OOS além de "anos positivos".

Por que a persistência anual não basta
---------------------------------------
`persistência = fração de anos com média positiva` responde *"o sinal aparece
com frequência?"* e não responde *"o resultado depende de um ano só?"*. As duas
podem divergir de forma perversa: uma estratégia com 8 de 10 anos ligeiramente
positivos e um ano gigantesco tem persistência 80% — e o resultado inteiro é o
ano gigantesco. Isso não é edge, é um evento com testemunhas.

Os dois testes, e por que estes dois
--------------------------------------
**1. Concentração.** Que fração do ganho total vem do MELHOR período?

    concentracao = max(ganho_do_periodo) / soma(ganhos positivos)

Acima de `CONCENTRACAO_MAXIMA` (50%), um único período responde por mais da
metade — e a hipótese está apostando que aquele período se repete, não que o
mecanismo funciona.

**2. Replicação por metade.** Parte o OOS ao meio cronologicamente: as duas
metades têm o mesmo sinal do total?

É o teste mais duro disponível sem coletar dado novo — é literalmente
out-of-sample dentro do out-of-sample. Uma estratégia que só funciona na
primeira metade descobriu um regime que acabou; uma que só funciona na segunda
pode estar surfando um regime que ainda não terminou. Nos dois casos o agregado
mente sobre o que se pode esperar do próximo período.

O que este módulo NÃO decide
-----------------------------
Ele devolve números e dois booleanos; **não** devolve veredito. Quem combina
isso com `t`, custo e capacidade é `mapa_validade.classificar`. Manter separado
importa porque estabilidade instável não é sinônimo de hipótese ruim — pode ser
hipótese CONDICIONAL, que é a doutrina do ADR 0031 (*"o Radar não pergunta se
uma estratégia funciona, e sim em quais condições"*). Um resultado concentrado
num regime é exatamente o que a análise condicional existe para investigar, e
transformá-lo em morte aqui apagaria o lead.

Ausência de dado é `None`, nunca `False`
-----------------------------------------
Fonte sem retorno por período devolve `avaliar(...) is None` — e o chamador tem
de tratar isso como **NÃO MEDIDA**, jamais como reprovação. É o mesmo princípio
do quarto veredito do Ledger, um nível acima.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

CONCENTRACAO_MAXIMA: Final[float] = 0.50
"""Acima disto, um único período responde por mais da metade do ganho."""

MINIMO_PERIODOS: Final[int] = 4
"""Menos que isto não permite partir ao meio de forma informativa: com 3
períodos, cada metade tem 1 ou 2 pontos e o teste vira ruído. Abaixo do mínimo
o resultado é `None` (não medida), nunca reprovação."""


@dataclass(frozen=True)
class Estabilidade:
    n_periodos: int
    concentracao: float
    """Fração do ganho total vinda do melhor período."""
    concentrada: bool
    soma_primeira_metade: float
    soma_segunda_metade: float
    metades_concordam: bool
    """As duas metades têm o mesmo sinal do total."""
    estavel: bool
    """`not concentrada and metades_concordam` — o critério fechado."""
    motivos: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_periodos": self.n_periodos,
            "concentracao": self.concentracao,
            "concentrada": self.concentrada,
            "soma_primeira_metade": self.soma_primeira_metade,
            "soma_segunda_metade": self.soma_segunda_metade,
            "metades_concordam": self.metades_concordam,
            "estavel": self.estavel,
            "motivos": list(self.motivos),
        }


def _ordenar(por_periodo: dict[str, float]) -> list[tuple[str, float]]:
    """
    Ordena por CHAVE (o período), não por valor. Ordenar por valor destruiria a
    cronologia e o teste das metades passaria a comparar "os melhores" com "os
    piores" — que dá reprovação garantida e não significa nada.
    """
    itens = [(k, float(v)) for k, v in por_periodo.items() if v == v]  # descarta NaN
    return sorted(itens, key=lambda kv: kv[0])


def avaliar(por_periodo: dict[str, float] | None) -> Estabilidade | None:
    """
    `por_periodo` mapeia período (ano, como string ordenável) → retorno médio
    daquele período. Devolve `None` quando não há dado suficiente — o chamador
    reporta **NÃO MEDIDA**, nunca reprovação.
    """
    if not por_periodo:
        return None
    itens = _ordenar(por_periodo)
    if len(itens) < MINIMO_PERIODOS:
        return None

    valores = [v for _, v in itens]
    total = sum(valores)
    ganhos = [v for v in valores if v > 0]

    # Concentração sobre a soma dos GANHOS (não do total líquido): com total
    # próximo de zero, dividir pelo total explodiria para infinito e a métrica
    # perderia sentido justamente onde ela mais importa.
    soma_ganhos = sum(ganhos)
    concentracao = (max(ganhos) / soma_ganhos) if soma_ganhos > 0 else 1.0
    concentrada = concentracao > CONCENTRACAO_MAXIMA

    meio = len(valores) // 2
    primeira, segunda = sum(valores[:meio]), sum(valores[meio:])
    sinal = (total > 0) - (total < 0)
    concordam = bool(
        sinal != 0
        and ((primeira > 0) - (primeira < 0)) == sinal
        and ((segunda > 0) - (segunda < 0)) == sinal
    )

    motivos: list[str] = []
    if concentrada:
        motivos.append(
            f"{concentracao:.0%} do ganho vem de um único período "
            f"(> {CONCENTRACAO_MAXIMA:.0%}) — é um evento, não um mecanismo"
        )
    if not concordam:
        motivos.append(
            f"as metades não concordam com o total: "
            f"1ª={primeira:+.4f}, 2ª={segunda:+.4f}, total={total:+.4f}"
        )

    return Estabilidade(
        n_periodos=len(itens),
        concentracao=concentracao,
        concentrada=concentrada,
        soma_primeira_metade=primeira,
        soma_segunda_metade=segunda,
        metades_concordam=concordam,
        estavel=(not concentrada) and concordam,
        motivos=tuple(motivos),
    )

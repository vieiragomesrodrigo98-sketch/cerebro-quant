"""
src/radar/cerebro/diagnostico_desfecho.py — por que perdemos, separado de quanto.

FASE 3 do plano aprovado em 2026-08-13, implementando o **ADR 0039**.

A pergunta que este módulo responde não é *"quanto rendeu?"* — é
**"a leitura de mercado que originou o sinal estava certa, mesmo que o trade
tenha perdido?"**. São perguntas diferentes, e confundi-las é o que faz uma mesa
trocar um modelo que lê bem e executa mal por um que não lê nada.

⚠️ ISTO NÃO TRANSFORMA PREJUÍZO EM ACERTO
------------------------------------------
Um desfecho classificado `LEITURA_CORRETA` continua sendo prejuízo. A
classificação existe para dizer **qual camada precisa melhorar** — leitura,
seleção de ativo, timing, execução ou nada (o regime mudou). Nunca para
reescrever o resultado financeiro.

Por que começa em `time_exit` e não em `stop_hit`
-------------------------------------------------
Emenda 2 do ADR 0039, e ela é medida: sobre 305 sinais com desfecho,
`stop_hit` são **13%** e `time_exit` são **62,6%**. O caso extremo já está
isolado — `operacao_rapida` a 1 dia é 67% da amostra, expira 72%, e a mediana do
caminho percorrido até o alvo é **ZERO**. O maquinário é o mesmo; a ordem de
aplicação inverte, porque é onde está o volume.

⚠️ A JANELA PÓS-SAÍDA É PARÂMETRO OBRIGATÓRIO
----------------------------------------------
Emenda 1 do ADR 0039. A pergunta *"o movimento previsto aconteceu depois da
saída?"* só é respondível contra uma janela **fixada antes**. Sem isso,
`LEITURA_CORRETA` vira a classe que absorve todo desfecho ruim — porque em
alguma janela futura, quase todo movimento acontece. Por isso `janela_pos_saida`
não tem default: quem chama declara, e a família registra no pré-registro dela.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class Diagnostico(StrEnum):
    """As 8 classes do ADR 0039 §3. Nomes fixos: viram chave de painel e de log."""

    LEITURA_CORRETA = "leitura_correta"
    LEITURA_PARCIAL = "leitura_parcial"
    LEITURA_ERRADA = "leitura_errada"
    TIMING_ERRADO = "timing_errado"
    ATIVO_ERRADO = "ativo_errado"
    REGIME_MUDOU = "regime_mudou"
    EXECUCAO_PROBLEMATICA = "execucao_problematica"
    INCONCLUSIVO = "inconclusivo"


#: Fração do caminho até o alvo a partir da qual o movimento previsto
#: **aconteceu**, ainda que não tenha sido capturado.
#:
#: `0,90` não é escolha estética: é onde a distribuição medida separa os casos.
#: Sobre 305 desfechos reais, `target_hit` tem razão **1,00** por definição e
#: `partial_target` fica entre **0,52 e 0,89** — ou seja, acima de 0,90 só vive
#: quem de fato alcançou o alvo. Mudar este valor é decisão de pré-registro.
FRACAO_MOVIMENTO_OCORREU: Final[float] = 0.90

#: Piso para "parte relevante da tese ocorreu". Metade do caminho é o ponto em
#: que o R:R nominal do sinal já teria virado 1:1.
FRACAO_MOVIMENTO_PARCIAL: Final[float] = 0.50

#: Quanto do movimento previsto o benchmark precisa ter feito para que a leitura
#: de MERCADO conte como certa — é o que separa `ATIVO_ERRADO` de
#: `LEITURA_ERRADA`. Mesmo piso do parcial, por simetria declarada.
FRACAO_BENCHMARK_CONFIRMA: Final[float] = 0.50

#: Slippage (em fração do preço de entrada) a partir da qual a execução deixa de
#: ser ruído e passa a explicar o desfecho.
#:
#: `0,003` é o custo round-trip conservador de cripto — o critério é: a execução
#: sozinha comeu mais que um pedágio inteiro.
SLIPPAGE_QUE_EXPLICA: Final[float] = 0.003


@dataclass(frozen=True, slots=True)
class FatosDoDesfecho:
    """Tudo que o classificador precisa — e **só** o que ele precisa.

    Nenhum campo é opcional por conveniência: `None` significa *não medido*, e o
    classificador trata ausência como motivo de `INCONCLUSIVO`, jamais como
    zero. Imputar zero num campo ausente transformaria falta de dado em
    afirmação sobre o mercado.
    """

    #: Fração do caminho até o alvo percorrida DENTRO da operação (`MFE / (alvo −
    #: entrada)`), medida do caminho de preço real.
    fracao_alvo_na_operacao: float | None
    #: Fração do caminho até o alvo alcançada na janela pós-saída declarada.
    #: É o que responde "o movimento veio depois?".
    fracao_alvo_pos_saida: float | None
    #: Fração do movimento previsto que o benchmark (BTC) fez no MESMO intervalo.
    #: Separa "erramos o mercado" de "acertamos o mercado e erramos o ativo".
    fracao_benchmark: float | None
    #: Slippage realizado, em fração do preço de entrada. `None` = não medido.
    slippage: float | None
    #: O regime lido na emissão continuou válido até a saída?
    regime_manteve: bool | None

    @property
    def mensuravel(self) -> bool:
        """Sem o caminho dentro da operação não há o que diagnosticar."""
        return self.fracao_alvo_na_operacao is not None and not math.isnan(
            self.fracao_alvo_na_operacao
        )


@dataclass(frozen=True, slots=True)
class Laudo:
    diagnostico: Diagnostico
    motivo: str

    @property
    def evidencia_contra_a_leitura(self) -> bool:
        """A ÚNICA classe que conta como evidência contra a capacidade de leitura
        do modelo (ADR 0039 §8).

        `TIMING_ERRADO`, `ATIVO_ERRADO`, `EXECUCAO_PROBLEMATICA` e
        `REGIME_MUDOU` apontam para outras camadas; `INCONCLUSIVO` não aponta
        para lugar nenhum. Tratar todos como "o modelo errou" é o que faz uma
        auditoria semanal aposentar o especialista errado.
        """
        return self.diagnostico is Diagnostico.LEITURA_ERRADA


def diagnosticar(fatos: FatosDoDesfecho) -> Laudo:
    """Classifica UM desfecho. A ORDEM das perguntas é a doutrina.

    1. **Dá para medir?** Sem caminho medido não há laudo — `INCONCLUSIVO` vem
       antes de tudo, porque toda pergunta seguinte pressupõe o dado.
    2. **A execução explica sozinha?** Slippage acima de um pedágio inteiro
       torna as perguntas seguintes ruído sobre ruído.
    3. **O regime mudou?** Vem antes de julgar a leitura: uma leitura coerente
       no instante do sinal não vira errada porque o mundo mudou depois.
    4. **O movimento veio DEPOIS da saída?** Se veio, a direção estava certa e o
       problema é de prazo — `TIMING_ERRADO`.
    5. **O mercado foi e o ativo não?** `ATIVO_ERRADO` — a leitura de mercado
       estava certa, a seleção não.
    6. **O movimento aconteceu dentro da operação?** `LEITURA_CORRETA` (≥90%) ou
       `LEITURA_PARCIAL` (≥50%).
    7. Nada disso: `LEITURA_ERRADA`.

    Inverter 4 e 5 mudaria o veredito de um caso real: sinal cujo movimento veio
    depois E cujo benchmark andou seria classificado como problema de seleção
    quando é de prazo.
    """
    if not fatos.mensuravel:
        return Laudo(
            Diagnostico.INCONCLUSIVO,
            "caminho de preço não medido dentro da operação — sem ele nenhuma "
            "das perguntas seguintes tem resposta",
        )

    dentro = float(fatos.fracao_alvo_na_operacao or 0.0)

    if fatos.slippage is not None and abs(fatos.slippage) >= SLIPPAGE_QUE_EXPLICA:
        return Laudo(
            Diagnostico.EXECUCAO_PROBLEMATICA,
            f"slippage de {fatos.slippage:.2%} — sozinho, acima de um custo "
            f"round-trip inteiro ({SLIPPAGE_QUE_EXPLICA:.2%}); a tese pode estar "
            "válida e não é ela que explica o desfecho",
        )

    if fatos.regime_manteve is False:
        return Laudo(
            Diagnostico.REGIME_MUDOU,
            "o regime lido na emissão não valia mais na saída — a leitura pode "
            "ter sido coerente no instante em que foi feita",
        )

    pos = fatos.fracao_alvo_pos_saida
    if pos is not None and pos >= FRACAO_MOVIMENTO_OCORREU > dentro:
        return Laudo(
            Diagnostico.TIMING_ERRADO,
            f"o movimento previsto veio DEPOIS da saída ({pos:.0%} do alvo na "
            f"janela pós-saída, contra {dentro:.0%} dentro da operação) — "
            "direção certa, prazo errado",
        )

    bench = fatos.fracao_benchmark
    if (
        bench is not None
        and bench >= FRACAO_BENCHMARK_CONFIRMA
        and dentro < FRACAO_MOVIMENTO_PARCIAL
    ):
        return Laudo(
            Diagnostico.ATIVO_ERRADO,
            f"o mercado fez {bench:.0%} do movimento previsto e este ativo fez "
            f"{dentro:.0%} — a leitura de mercado estava certa, a escolha do "
            "ativo não",
        )

    if dentro >= FRACAO_MOVIMENTO_OCORREU:
        return Laudo(
            Diagnostico.LEITURA_CORRETA,
            f"o preço percorreu {dentro:.0%} do caminho até o alvo — o movimento "
            "previsto aconteceu e não foi capturado",
        )

    if dentro >= FRACAO_MOVIMENTO_PARCIAL:
        return Laudo(
            Diagnostico.LEITURA_PARCIAL,
            f"o preço percorreu {dentro:.0%} do caminho — parte relevante da "
            "tese ocorreu, com comportamento diferente do esperado",
        )

    return Laudo(
        Diagnostico.LEITURA_ERRADA,
        f"o preço percorreu {dentro:.0%} do caminho até o alvo e nem o mercado "
        "nem a janela pós-saída sustentam a tese",
    )


def fracao_do_alvo(
    maximo_no_intervalo: float | None,
    entrada: float,
    alvo: float,
) -> float | None:
    """Fração do caminho até o alvo que um máximo de preço representa.

    `(máximo − entrada) / (alvo − entrada)`, em reais sobre reais.

    ⚠️ A unidade importa e já custou um erro de ~100× neste projeto: `mfe` e
    `mae` estão em REAIS, não em percentual. Dividir reais por percentual produz
    um resultado errado que **confirma a hipótese** — foi pego só pelo controle
    interno (`target_hit` aparecendo com 10% do caminho quando por definição tem
    de dar ≥ 100%).

    Devolve `None` quando não dá para calcular, jamais zero: alvo abaixo da
    entrada é sinal degenerado, e imputar zero transformaria dado inválido em
    afirmação sobre o mercado.
    """
    if maximo_no_intervalo is None or not entrada or not alvo:
        return None
    distancia = float(alvo) - float(entrada)
    if distancia <= 0:
        return None
    return (float(maximo_no_intervalo) - float(entrada)) / distancia


def fracao_do_benchmark(
    retorno_benchmark: float | None,
    entrada: float,
    alvo: float,
) -> float | None:
    """Quanto do movimento previsto o benchmark fez, na mesma janela.

    O previsto é `(alvo − entrada) / entrada` em fração; o benchmark entra já
    como retorno fracionário. A razão responde *"o mercado foi para onde a tese
    dizia?"* — que é o que separa `ATIVO_ERRADO` de `LEITURA_ERRADA`.

    Note que **não** se compara o retorno do ativo com o do benchmark: compara-se
    o do benchmark com o MOVIMENTO PREVISTO. Um mercado que subiu 0,5% quando a
    tese pedia 5% não confirma a leitura — confirma que o mercado andou pouco.
    """
    if retorno_benchmark is None or not entrada or not alvo:
        return None
    previsto = (float(alvo) - float(entrada)) / float(entrada)
    if previsto <= 0:
        return None
    return float(retorno_benchmark) / previsto


@dataclass(frozen=True, slots=True)
class Cobertura:
    """Quantos desfechos puderam de fato ser checados em cada eixo.

    ⚠️ Existe por causa de uma armadilha concreta: `fracao_alvo_pos_saida=None`
    **não** produz `INCONCLUSIVO` — o classificador apenas pula a pergunta de
    timing, e o desfecho cai em `LEITURA_ERRADA`. Sem publicar cobertura,
    *"timing não é a causa"* fica indistinguível de *"não deu para checar"*.

    É a mesma assimetria do `G-P2`, um nível acima: lá era o sinal que não podia
    ser mudo; aqui é o diagnóstico.
    """

    total: int
    com_pos_saida: int
    com_benchmark: int

    @property
    def fracao_pos_saida(self) -> float:
        return self.com_pos_saida / self.total if self.total else 0.0

    @property
    def fracao_benchmark(self) -> float:
        return self.com_benchmark / self.total if self.total else 0.0

    @property
    def confiavel(self) -> bool:
        """Abaixo de 50% em qualquer eixo, o placar é piso e não retrato."""
        return self.fracao_pos_saida >= 0.5 and self.fracao_benchmark >= 0.5


def consolidar(laudos: list[Laudo]) -> dict[str, object]:
    """O placar da auditoria semanal (ADR 0039 §6).

    Publica `evidencia_contra_a_leitura` separado do total **de propósito**: é a
    única fatia que autoriza conclusão sobre a capacidade de leitura do modelo.
    Somar as 8 classes e chamar o resultado de "taxa de erro" apagaria
    exatamente a distinção que este módulo existe para criar.
    """
    contagem = {d.value: 0 for d in Diagnostico}
    for laudo in laudos:
        contagem[laudo.diagnostico.value] += 1
    total = len(laudos)
    contra = sum(1 for laudo in laudos if laudo.evidencia_contra_a_leitura)
    return {
        "por_diagnostico": contagem,
        "_total": total,
        "_evidencia_contra_a_leitura": contra,
        "_fracao_contra_a_leitura": (contra / total) if total else 0.0,
        "_inconclusivos": contagem[Diagnostico.INCONCLUSIVO.value],
    }

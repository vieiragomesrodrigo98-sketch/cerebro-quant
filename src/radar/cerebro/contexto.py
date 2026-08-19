"""
src/radar/cerebro/contexto.py — o estágio MARKET CONTEXT do Cérebro vivo.

ADR 0036: `UNIVERSO → MARKET MONITOR → **MARKET CONTEXT** → OPPORTUNITY
DETECTION → …`. Este módulo responde *"o que está acontecendo no mercado e
neste ativo?"* — e nada além disso.

Por que ele nasce
-----------------
O contexto existia, mas **espalhado**: `radar.mie.regime.detectar_regime` (B3),
`radar.historical.regime_cripto.detectar_regime_cripto`,
`regime_classifier.py` e `regime_timeline.py`. Não havia módulo de contexto no
pacote `cerebro`, e quem quisesse o contexto precisava saber de qual mercado
está falando **antes** de pedir. Num Cérebro que varre B3 e Cripto no mesmo
laço, isso empurra o `if mercado == ...` para cada chamador — e cada chamador
erra de um jeito diferente.

Aqui há um portão só, e ele roteia.

⚠️ CONTEXTO É PARTIÇÃO, NUNCA FEATURE
-------------------------------------
Portão **G-P1** do Contrato de Pensamento (ADR 0032). O contexto **separa
populações**; ele não entra no vetor de entrada do modelo. A razão é medida, e
custou um motor: o Cérebro v1 tinha 7 de 45 colunas constantes dentro do dia, e
a de maior ganho era uma delas — o modelo aprendeu **o dia**, não o ativo.

Por isso este módulo expõe `chave_de_particao()` e **não** expõe nada que
pareça um vetor numérico pronto para concatenar. Quem quiser usar contexto como
feature vai ter que escrever a violação explicitamente, e o portão G-P1 vai
pegá-la.

O que este módulo NÃO faz
-------------------------
Não detecta oportunidade (é `deteccao.py`), não avalia expectativa (é
`expectativa.py`), não decide (é `decisao.py`) e **não reimplementa detecção de
regime** — as regras vivem nos detectores existentes, testados, e continuam lá.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final, Protocol

from loguru import logger

if TYPE_CHECKING:  # pragma: no cover — só para tipo, evita pandas em import leve
    import pandas as pd


class DetectorDeRegime(Protocol):
    """Quem sabe ler regime de um mercado. O Cérebro define o contrato; o motor
    o implementa e é **injetado na borda**.

    Por que injeção e não import
    ----------------------------
    A primeira versão deste módulo importava `radar.mie.regime.detectar_regime`
    direto — e o teste de camadas reprovou, corretamente. O ADR 0032 põe o
    Cérebro **acima** dos motores: a dependência é `motor → contrato`, nunca o
    inverso, e *"se o símbolo é mesmo compartilhado, ele pertence ao Cérebro (ou
    a um módulo neutro), não ao motor"*.

    A saída fácil seria acrescentar a violação à allowlist. Não foi feita: a
    allowlist existe para **dívida herdada**, e dívida nova acomodada nela é o
    mecanismo virando decoração. O detector entra por parâmetro, a composição
    acontece no script que roda o monitor, e `radar.cerebro` continua sem
    conhecer motor nenhum.
    """

    def __call__(self, precos: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover
        ...


class Mercado(StrEnum):
    """Os dois mercados do universo. Não é `asset_class`: aqui interessa qual
    **detector de regime** se aplica, e essa é a única pergunta que o roteamento
    faz."""

    B3 = "b3"
    CRIPTO = "cripto"


#: Os 4 regimes. Iguais nos dois mercados **de propósito** — o ADR 0032 exige
#: "vocabulário de contexto comum" (G-P4). Sem isso, uma zona de validade medida
#: em B3 não pode nem ser comparada com uma de cripto, e o Router (que o ADR 0032
#: coloca acima dos especialistas) não tem sobre o que rotear.
REGIMES: Final[tuple[str, ...]] = ("panico", "bear", "bull_trending", "bull_lateral")

#: Devolvido quando o detector não consegue classificar — dado insuficiente,
#: janela incompleta, benchmark ausente. **Não é um 5º regime**: é a ausência de
#: leitura, e quem consome precisa tratá-la como tal.
#:
#: Existe porque a alternativa é pior: cair no regime mais comum por default
#: transformaria falta de dado em afirmação sobre o mercado — o mesmo modo de
#: falha que fez a sentinela reportar verde sobre antibot desligado.
DESCONHECIDO: Final[str] = "desconhecido"


@dataclass(frozen=True, slots=True)
class ContextoDeMercado:
    """O contexto de UM instante, para UM mercado.

    Imutável de propósito: contexto é observação, e observação que o consumidor
    pode alterar deixa de servir como chave de partição — duas hipóteses
    poderiam declarar a mesma partição e estar falando de coisas diferentes.
    """

    mercado: Mercado
    regime: str
    #: `None` quando o detector não publica — e `None` é diferente de `0.0`.
    #: Publicar zero seria afirmar "sem confiança medida" como se fosse
    #: "confiança medida em zero".
    confianca: float | None = None

    def __post_init__(self) -> None:
        if self.regime not in REGIMES and self.regime != DESCONHECIDO:
            raise ValueError(
                f"regime {self.regime!r} fora do vocabulário comum {REGIMES} "
                f"(ou {DESCONHECIDO!r}). G-P4 do Contrato de Pensamento exige "
                "vocabulário compartilhado entre mercados — regime local a um "
                "mercado impede o Router de comparar zonas de validade."
            )

    @property
    def legivel(self) -> bool:
        """Houve leitura de regime? `False` obriga o chamador a decidir o que
        fazer, em vez de receber um default que parece medição."""
        return self.regime != DESCONHECIDO

    def chave_de_particao(self) -> str:
        """A chave que PARTICIONA a população — a única forma suportada de usar
        contexto (G-P1).

        Deliberadamente uma string, e não um vetor: quem quiser concatenar isto
        num modelo precisa converter à mão, e a conversão fica visível na
        revisão e no portão.
        """
        return f"{self.mercado.value}:{self.regime}"


#: Os nomes que os detectores REAIS usam para a coluna de regime, na ordem em
#: que são procurados.
#:
#: 🔴 `regime_mie` vem primeiro porque é o que **os dois** detectores do projeto
#: publicam (`radar.mie.regime.COLUNAS_SAIDA` e
#: `radar.historical.regime_cripto.COLUNAS_SAIDA`). A versão anterior procurava
#: **só** `"regime"` — nome que detector nenhum produz —, e o efeito era
#: `contexto.detectar()` devolver `DESCONHECIDO` para todo detector, sempre.
#:
#: Passou despercebido porque este módulo tem **zero importadores fora do
#: pacote**: a etapa de CONTEXTO do Cérebro nunca rodou contra um detector de
#: verdade, então a única evidência de que funcionava eram testes que injetavam
#: um quadro com a coluna `"regime"` — quadro que a produção não gera. Achado
#: ao ligar o ciclo do ADR 0040, e é a razão de "ligar" valer mais que "ter".
#:
#: `"regime"` fica como segundo nome porque é o que os testes e o broadcast do
#: feature store usam; aceitar os dois é o que torna a leitura verdadeira em vez
#: de trocar uma grafia errada por outra.
NOMES_DA_COLUNA_DE_REGIME: Final[tuple[str, ...]] = ("regime_mie", "regime")


def _ultima_linha_regime(quadro: pd.DataFrame) -> tuple[str, float | None]:
    """Extrai (regime, confiança) da ÚLTIMA linha do quadro devolvido pelos
    detectores.

    A última linha é o instante corrente — os detectores publicam a série
    inteira, e usar qualquer outra linha seria olhar para o passado achando que
    é o presente. Se o quadro vier vazio, isso é ausência de leitura, não erro:
    o Cérebro varre o universo inteiro e ativo sem histórico suficiente é
    esperado, não excepcional.
    """
    if quadro is None or len(quadro) == 0:
        return DESCONHECIDO, None
    linha = quadro.iloc[-1]
    regime = DESCONHECIDO
    for nome in NOMES_DA_COLUNA_DE_REGIME:
        if nome in quadro.columns:
            regime = str(linha.get(nome, DESCONHECIDO) or DESCONHECIDO)
            break
    if regime not in REGIMES:
        return DESCONHECIDO, None
    bruto = linha.get("confianca", None)
    try:
        confianca = None if bruto is None else float(bruto)
    except (TypeError, ValueError):
        confianca = None
    return regime, confianca


def detectar(
    precos: pd.DataFrame,
    mercado: Mercado | str,
    detector: DetectorDeRegime,
) -> ContextoDeMercado:
    """Contexto do instante corrente, usando o detector INJETADO.

    **Não reimplementa regra nenhuma e não importa motor nenhum.** Os detectores
    reais (`radar.mie.regime.detectar_regime` para B3,
    `radar.historical.regime_cripto.detectar_regime_cripto` para cripto) são
    rule-based, sem look-ahead e já testados — trazer as regras para cá
    duplicaria doutrina e criaria duas verdades sobre o mesmo regime; importá-los
    inverteria a dependência que o ADR 0032 fixa.

    Quem compõe é a borda. Ver `radar.cerebro.contexto.detectores_padrao` — que
    mora **fora** deste módulo, no script que roda o monitor.

    Falha do detector vira `DESCONHECIDO`, não exceção: o Cérebro vivo varre o
    universo inteiro continuamente, e um ativo sem dado suficiente não pode
    derrubar o laço. Foi o modo de falha do ciclo overnight de 2026-08-08, em
    que um único ticker com fechamento ausente matou a rodada inteira.
    """
    alvo = Mercado(mercado) if not isinstance(mercado, Mercado) else mercado
    try:
        quadro = detector(precos)
    # `except Exception` largo é DELIBERADO: ver docstring. O Cérebro vivo varre
    # o universo inteiro, e nenhum ativo com dado ruim pode derrubar o laço.
    #
    # O log NÃO é decoração, e a distinção importa: engolir a exceção é a
    # decisão certa (o laço não pode cair); engolir em SILÊNCIO não é. Um ativo
    # cujo detector falha sempre viraria `DESCONHECIDO` para sempre, e ninguém
    # saberia — que é a definição de "feature morre em silêncio". A degradação
    # continua idêntica; só deixa de ser invisível.
    except Exception as exc:
        logger.warning(
            "CONTEXTO: detector de regime falhou para {} ({}: {}) — "
            "devolvendo DESCONHECIDO e seguindo a varredura",
            alvo.value if isinstance(alvo, Mercado) else alvo,
            type(exc).__name__,
            exc,
        )
        return ContextoDeMercado(mercado=alvo, regime=DESCONHECIDO)

    regime, confianca = _ultima_linha_regime(quadro)
    return ContextoDeMercado(mercado=alvo, regime=regime, confianca=confianca)

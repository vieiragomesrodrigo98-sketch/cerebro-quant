"""
src/radar/cerebro/monitor.py — o estágio MARKET MONITOR do Cérebro vivo.

ADR 0036: observação **contínua** do universo elegível inteiro. Hoje o projeto
roda cron em horário fixo; este módulo é a peça que torna o Cérebro *vivo* em
vez de um lote que roda e devolve score.

⚠️ O MONITOR É OBSERVADOR, NÃO DECISOR
--------------------------------------
Ordem explícita do DEV, e é a trava mais importante deste arquivo:

    UNIVERSO → NORMALIZAÇÃO → OBSERVAÇÃO → CONTEXTO → CANDIDATOS      ✅
    UNIVERSO → MONITOR → SCORE → SINAL                                ❌

O monitor responde **"o que está acontecendo?"** e para aí. Quem pergunta
*"existe oportunidade compatível?"* é `deteccao.py`; *"essa oportunidade tem
vantagem?"* é `expectativa.py`; *"devemos emitir?"* é `decisao.py`.

A razão é histórica e cara: o motor anterior era monolítico, e quando ele
adoeceu não dava para saber **qual** estágio estava errado — foram necessários
4 defeitos encadeados medidos um a um para entender que o v1 estava quebrado.
Um monitor que pontua volta a fundir observação com julgamento, e a próxima
autópsia volta a ser impossível.

Por isso a saída deste módulo é `Observacao`, que **não tem score, nem
probabilidade, nem direção**. Se algum dia tiver, a decomposição morreu.

Cadência é parâmetro pré-registrado
-----------------------------------
`intervalo_s` não é detalhe de implementação: avaliar a cada instante multiplica
**tentativas** mesmo com a hipótese fixa, e o ADR 0031 cobra penalização por
tentativas. Trocar a cadência muda o denominador do FDR — por isso ela é
explícita, e mudá-la é decisão registrada, não ajuste de configuração.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from radar.cerebro.contexto import (
    DESCONHECIDO,
    ContextoDeMercado,
    DetectorDeRegime,
    Mercado,
    detectar,
)
from radar.cerebro.universo import AtivoRef, ProvedorDeUniverso, unir

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

#: Cadência default, em segundos. 300 s = 5 min, a mesma do monitor sintético
#: que já roda na VPS — escolhida por simetria operacional, não por medição de
#: mercado. Quem apertar precisa declarar o efeito no denominador do FDR.
INTERVALO_PADRAO_S: Final[float] = 300.0

#: Assinatura de quem entrega o histórico de preços de um ativo. Injetada pela
#: mesma razão de `DetectorDeRegime` e `ProvedorDeUniverso`: `radar.cerebro` não
#: conhece `radar.lab`, `radar.historical` nem motor algum.
FonteDePrecos = Callable[[AtivoRef], "pd.DataFrame"]


@dataclass(frozen=True, slots=True)
class Observacao:
    """O que o monitor viu sobre UM ativo, num instante.

    **Não contém score, probabilidade nem direção** — ver a trava no cabeçalho
    do módulo. É insumo de `deteccao.py`, não conclusão.
    """

    ativo: AtivoRef
    contexto: ContextoDeMercado
    #: `False` quando o regime não pôde ser lido. O ativo **continua no
    #: universo** (ADR 0036: nada sai por mérito) — apenas esta observação não
    #: sustenta avaliação, e quem consome precisa decidir o que fazer com isso
    #: em vez de receber um default que parece medição.
    elegivel_no_ciclo: bool

    @property
    def chave_de_particao(self) -> str:
        """Atalho para a partição do contexto — a única forma suportada de usar
        contexto (G-P1)."""
        return self.contexto.chave_de_particao()


def observar_um(
    ativo: AtivoRef,
    fonte_de_precos: FonteDePrecos,
    detectores: dict[Mercado, DetectorDeRegime],
) -> Observacao:
    """Observa UM ativo. Nunca levanta.

    A varredura total encontra ativo sem histórico, com dado corrompido e com
    janela incompleta — isso é **esperado**, não excepcional. Deixar a exceção
    subir mataria o ciclo inteiro por causa de um ativo, que foi exatamente o
    que o `Decimal("NaN")` fez com a rodada overnight de 2026-08-08.
    """
    detector = detectores.get(ativo.mercado)
    if detector is None:
        # Mercado sem detector configurado é erro de COMPOSIÇÃO, e ainda assim
        # não derruba a varredura: vira observação inelegível, e o chamador vê
        # o buraco no relatório em vez de perder o ciclo.
        return Observacao(
            ativo=ativo,
            contexto=ContextoDeMercado(mercado=ativo.mercado, regime=DESCONHECIDO),
            elegivel_no_ciclo=False,
        )
    try:
        precos = fonte_de_precos(ativo)
    except Exception:
        precos = None  # type: ignore[assignment]

    contexto = (
        ContextoDeMercado(mercado=ativo.mercado, regime=DESCONHECIDO)
        if precos is None
        else detectar(precos, ativo.mercado, detector)
    )
    return Observacao(ativo=ativo, contexto=contexto, elegivel_no_ciclo=contexto.legivel)


def varrer(
    provedores: Sequence[ProvedorDeUniverso],
    fonte_de_precos: FonteDePrecos,
    detectores: dict[Mercado, DetectorDeRegime],
) -> list[Observacao]:
    """Um ciclo de observação sobre o universo INTEIRO.

    Devolve uma `Observacao` por ativo — **inclusive as inelegíveis**. Omitir as
    que falharam faria "não observei" e "observei e estava limpo" ficarem
    indistinguíveis, que é o defeito que o canário do scan de governança existe
    para impedir. Quem quiser só as elegíveis filtra explicitamente.
    """
    return [
        observar_um(ativo, fonte_de_precos, detectores) for ativo in unir(*provedores)
    ]


def ciclos(
    provedores: Sequence[ProvedorDeUniverso],
    fonte_de_precos: FonteDePrecos,
    detectores: dict[Mercado, DetectorDeRegime],
    *,
    intervalo_s: float = INTERVALO_PADRAO_S,
    maximo: int | None = None,
    dormir: Callable[[float], None] | None = None,
) -> Iterator[list[Observacao]]:
    """O laço vivo: um ciclo de `varrer` a cada `intervalo_s`.

    Gerador, e não `while True` interno, de propósito: quem consome decide o que
    fazer com cada ciclo e pode parar quando quiser, sem o monitor precisar
    conhecer alerta, persistência ou sinal. `maximo` e `dormir` existem para
    teste — rodar N ciclos sem `time.sleep` real.

    ⚠️ `intervalo_s` mexe no denominador do FDR (ver cabeçalho). Não é ajuste
    de configuração.
    """
    import time

    dormir_fn = dormir if dormir is not None else time.sleep
    ciclo = 0
    while maximo is None or ciclo < maximo:
        ciclo += 1
        yield varrer(provedores, fonte_de_precos, detectores)
        if maximo is None or ciclo < maximo:
            dormir_fn(intervalo_s)


def elegiveis(observacoes: Iterable[Observacao]) -> list[Observacao]:
    """As observações que sustentam avaliação. Filtro explícito, e não default:
    ver `varrer`."""
    return [o for o in observacoes if o.elegivel_no_ciclo]

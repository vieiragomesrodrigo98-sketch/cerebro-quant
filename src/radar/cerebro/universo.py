"""
src/radar/cerebro/universo.py — o CONTRATO de universo do Cérebro vivo.

ADR 0036: *"todo ativo elegível de B3 e Cripto permanece no universo"*. Este
módulo diz **o que é um ativo** para o Cérebro, e **quem sabe enumerá-los** —
sem saber como nenhum dos dois mercados faz isso.

> Correspondência com o desenho do DEV: `AssetRef` → `AtivoRef`;
> `UniverseProvider` → `ProvedorDeUniverso`. Nomes em português por coerência
> com o resto do pacote (`ContextoDeMercado`, `DetectorDeRegime`); os conceitos
> são os mesmos.

Por que um contrato, e não um `if mercado == ...`
--------------------------------------------------
Medido antes de escrever: os dois universos vêm de fontes **diferentes**, com
formatos **diferentes** — `radar.universe.get_active_tickers()` devolve
`TickerMeta` (B3, do COTAHIST) e `radar.lab.data.cripto_universo_completo()`
devolve `tuple[str, ...]` (cripto, do `exchangeInfo` da Binance). Sem contrato,
essa diferença vaza para cada chamador do Cérebro, e cada um a resolve de um
jeito.

**E a alternativa óbvia estava proibida:** importar `radar.lab` daqui esbarraria
na mesma regra de camada que já reprovou `contexto.py` ao importar
`radar.mie.regime`. A resposta é a mesma que funcionou lá — o Cérebro define o
contrato, a implementação é **injetada na borda**:

    Cérebro conhece  →  ProvedorDeUniverso  →  Iterable[AtivoRef]
    Cérebro NÃO conhece  →  como B3 ou Cripto enumeram

⚠️ Este módulo não elimina ativo nenhum por mérito
--------------------------------------------------
ADR 0036: existem **dois** filtros e só um vive aqui. O de **universo** é
higiene — ativo inexistente, dado inválido, instrumento degenerado — e é
responsabilidade do provedor, que conhece seu mercado. O de **oportunidade** é
o coração do Cérebro e acontece muito depois, em `deteccao.py` e adiante.

Nada neste módulo decide que um ativo "não vale a pena". Se algum dia decidir, é
o guardrail *resultado da estratégia ≠ propriedade do ativo* sendo violado.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from radar.cerebro.contexto import Mercado


class TipoDeInstrumento(StrEnum):
    """O que o ativo **é**, não o quão bom ele é.

    Existe porque microestrutura, custo e execução mudam com o instrumento — a
    lente do corretor, que as outras esquecem. Uma ação à vista e um par
    spot de cripto não compartilham spread, horário nem mecanismo de
    liquidação, e tratá-los como intercambiáveis é o erro que o custo uniforme
    do motor anterior cometia (`custos.py` já é round-trip **por ativo**).
    """

    ACAO = "acao"
    PAR_SPOT = "par_spot"
    INDICE = "indice"
    OUTRO = "outro"


@dataclass(frozen=True, slots=True)
class AtivoRef:
    """A referência normalizada de um ativo — a única forma que o Cérebro
    conhece.

    Imutável pela mesma razão do `ContextoDeMercado`: é identidade, e identidade
    que o consumidor altera deixa de identificar.
    """

    #: Identidade estável e única no universo INTEIRO, com o mercado embutido.
    #: `PETR4` e um hipotético `PETR4` de outro mercado não podem colidir — e
    #: colisão de identidade é o tipo de falha que o mapa de validade já teve
    #: de expor uma vez (hipóteses homônimas exibindo números da primeira).
    ativo_id: str
    #: Como o mercado o chama (`PETR4`, `BTCUSDT`). Pode repetir entre mercados.
    simbolo: str
    mercado: Mercado
    tipo: TipoDeInstrumento = TipoDeInstrumento.OUTRO
    #: Metadado do provedor (setor, nome, quoteAsset…). O Cérebro **não** lê
    #: isto para decidir — está aqui para diagnóstico e para o provedor não
    #: precisar de um canal paralelo.
    metadados: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.ativo_id or not self.simbolo:
            raise ValueError("AtivoRef exige `ativo_id` e `simbolo` não vazios")

    @classmethod
    def de(
        cls,
        simbolo: str,
        mercado: Mercado | str,
        tipo: TipoDeInstrumento = TipoDeInstrumento.OUTRO,
        **metadados: str,
    ) -> AtivoRef:
        """Construtor que **deriva** `ativo_id` de `mercado:simbolo`.

        Derivar em vez de aceitar por parâmetro é deliberado: `ativo_id` montado
        à mão em cada provedor produziria convenções divergentes, e a primeira
        divergência só apareceria quando duas hipóteses colidissem.
        """
        m = Mercado(mercado) if not isinstance(mercado, Mercado) else mercado
        return cls(
            ativo_id=f"{m.value}:{simbolo}",
            simbolo=simbolo,
            mercado=m,
            tipo=tipo,
            metadados=dict(metadados),
        )


class ProvedorDeUniverso(Protocol):
    """Quem sabe enumerar os ativos elegíveis de UM mercado.

    Implementações concretas vivem **fora** de `radar.cerebro` — no script ou na
    borda que compõe o Cérebro. É a mesma inversão de `DetectorDeRegime`, e pela
    mesma razão: o teste de camadas proíbe `radar.cerebro` conhecer motor, e a
    allowlist existe para dívida herdada, não para acomodar dívida nova.
    """

    @property
    def mercado(self) -> Mercado:  # pragma: no cover — contrato
        ...

    def ativos(self) -> Iterable[AtivoRef]:  # pragma: no cover — contrato
        """Os ativos elegíveis AGORA. Higiene já aplicada pelo provedor, que é
        quem conhece o que é degenerado no seu mercado."""
        ...


def unir(*provedores: ProvedorDeUniverso) -> list[AtivoRef]:
    """Junta os provedores num universo único, ordenado e sem duplicata.

    Ordenado por `ativo_id` de propósito: varredura em ordem não-determinística
    produz relatório que muda de execução para execução sem o mercado ter
    mudado — e este projeto já gastou tempo perseguindo diferença que era
    ordenação, não fenômeno.

    Duplicata é resolvida pela PRIMEIRA ocorrência, e isso é escolha
    conservadora: dois provedores anunciando o mesmo `ativo_id` é sintoma de
    configuração errada, não caso de uso. Se acontecer, a ordem estável faz o
    efeito ser reproduzível em vez de aleatório.
    """
    vistos: dict[str, AtivoRef] = {}
    for provedor in provedores:
        for ativo in provedor.ativos():
            vistos.setdefault(ativo.ativo_id, ativo)
    return sorted(vistos.values(), key=lambda a: a.ativo_id)

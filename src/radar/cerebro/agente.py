"""
src/radar/cerebro/agente.py — o ciclo cognitivo.

Pergunta única: **"aconteceu algo; o que, se alguma coisa, devo fazer?"**

⚠️ O QUE MUDA AQUI É A NATUREZA, NÃO A CAMADA
---------------------------------------------
Antes do ADR 0037 o Cérebro era isto:

    def executar():
        varrer(); detectar(); avaliar(); decidir()

Arquitetura de software boa, e ainda assim um **pipeline determinístico**. Não
percebia, não escolhia como investigar, não revisava o que sabia e não aprendia
com o resultado. **Executava.**

O erro que este módulo evita é o óbvio: acrescentar um `Orquestrador` por cima e
chamá-lo de agente. Isso seria `pipeline + camada + mais código` — o mesmo
programa procedural com nome melhor. A diferença real é que existe **estado que
sobrevive**, **política que restringe** e **ciclo que fecha**:

    evento → percepção → memória → missão → decisão → ação → registro → novo estado

Os cinco módulos das Fases 1–2 (`contexto` · `monitor` · `deteccao` ·
`expectativa` · `decisao`) **não são reescritos** (emenda 6). Eles viram
**ferramentas**: mudam de papel, não de código. Foi a catraca de camadas que os
deixou em funções puras com dependências injetadas — a forma exata que uma
ferramenta precisa ter.

⚠️ TODO CICLO PRODUZ REGISTRO, INCLUSIVE O QUE NÃO FAZ NADA
-----------------------------------------------------------
Emenda 7. `NENHUMA_ACAO` é gravada como qualquer outra. A razão é a assimetria
do ADR 0037 §5 — *abster custa zero, afirmar custa*: se ignorar fosse grátis
**e invisível**, o agente ignoraria em todo lugar e pareceria seletivo. Foi
literalmente o que aconteceu com o emissor v1, mudo por um mês e lido como
criterioso.

⚠️ O AGENTE NÃO ADULTERA METODOLOGIA
------------------------------------
Ele decide *o que fazer*. Não decide que a régua era outra depois de ver o
resultado. Toda restrição vive em `politica.py`, e este módulo **não tem
permissão para contorná-la**: não existe caminho aqui que produza ação sem
passar por `proxima_acao`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from loguru import logger

from radar.cerebro.estado import Estado
from radar.cerebro.politica import Acao, AcaoPermitida, Evento, Validade, proxima_acao
from radar.cerebro.teses import EstadoDaTese, Tese, despertar_vencidas


@dataclass(frozen=True, slots=True)
class Percepcao:
    """O que o Cérebro viu — **e nada além disso**.

    Sem score, sem probabilidade, sem direção: a mesma trava de `Observacao`,
    `Candidato` e `Barra1s`, e pela mesma razão. Percepção que já traz juízo
    torna impossível separar, na autópsia, "o dado estava errado" de "a leitura
    do dado estava errada".
    """

    evento: Evento
    momento: datetime
    assunto: str | None = None
    #: Números crus que o disparador já tinha em mãos. Cache, não julgamento.
    sinais: Mapping[str, float] | None = None


@dataclass(frozen=True, slots=True)
class RegistroDeCiclo:
    """O que aconteceu num ciclo. Existe **sempre**, mesmo quando nada foi feito."""

    percepcao: Percepcao
    decisao: AcaoPermitida
    executada: bool
    erro: str | None = None

    @property
    def linha_de_diario(self) -> str:
        estado = "executou" if self.executada else ("FALHOU" if self.erro else "não agiu")
        base = (
            f"{self.percepcao.momento.isoformat()} · {self.percepcao.evento.value} → "
            f"{self.decisao.acao.value} ({estado}): {self.decisao.motivo}"
        )
        return f"{base} | erro: {self.erro}" if self.erro else base


class Capacidade(Protocol):
    """Uma ferramenta que o agente pode invocar.

    Recebe a percepção e o estado, devolve o que quiser. É `Protocol` porque
    `cerebro` não conhece implementação — `monitor.varrer`, `deteccao`,
    `expectativa` e `decisao` entram por aqui, injetadas na borda.
    """

    def __call__(
        self, percepcao: Percepcao, estado: Estado
    ) -> object: ...  # pragma: no cover — contrato


#: Onde o chamador pluga as ferramentas. Ação sem capacidade registrada é
#: decidida e **não** executada — e isso aparece no registro em vez de virar
#: exceção: um Cérebro incompleto precisa poder rodar e mostrar o que lhe falta.
Capacidades = Mapping[Acao, Capacidade]

ConsultarValidade = Callable[[str], Validade | None]


def ciclo(
    percepcao: Percepcao,
    estado: Estado,
    *,
    capacidades: Capacidades | None = None,
    consultar_validade: ConsultarValidade | None = None,
    ha_metrica_faltando: bool = False,
) -> tuple[Estado, RegistroDeCiclo]:
    """Um giro completo: percebe, consulta, decide, age e registra.

    Devolve o **estado novo** junto com o registro, em vez de mutar: estado
    imutável é o que impede que um erro no meio do ciclo deixe metade da memória
    atualizada — que é a "superfície de bug nova" declarada nas consequências
    negativas do ADR 0037.

    Ordem, e ela não é arbitrária:

    1. **Despertar teses vencidas primeiro.** O prazo é um evento por si só, e
       processá-lo antes garante que o que já se decidiu esperar tenha
       precedência sobre o que acabou de acontecer.
    2. **Consultar validade** do assunto, se houver.
    3. **Perguntar à política.** É a **única** porta: não há caminho neste
       módulo que produza ação sem passar por `proxima_acao`.
    4. **Agir**, se houver capacidade registrada para a ação.
    5. **Registrar sempre** — inclusive `NENHUMA_ACAO`, inclusive falha.

    Capacidade que levanta **não derruba o ciclo**: o erro entra no registro e o
    estado avança. Um agente que morre na primeira ferramenta com defeito perde
    também a memória do que estava fazendo, e volta ao problema que este ADR
    existe para resolver.
    """
    despertas: list[Tese] = []
    if estado.teses is not None:
        # `despertar_vencidas` transiciona AGUARDANDO → DESPERTA e PERSISTE. A
        # leitura seguinte pega tanto as que acabaram de acordar quanto as que
        # já estavam despertas de ciclos anteriores — e a segunda metade
        # importa: uma tese que despertou e não pôde ser resolvida ficaria
        # órfã para sempre se só se olhasse a transição.
        list(despertar_vencidas(estado.teses, percepcao.momento))
        despertas = [
            t for t in estado.teses.abertas() if t.estado is EstadoDaTese.DESPERTA
        ]

    validade = (
        consultar_validade(percepcao.assunto)
        if consultar_validade is not None and percepcao.assunto
        else None
    )

    decisao = proxima_acao(
        evento=percepcao.evento,
        missao=estado.missao,
        orcamento=estado.orcamento,
        assunto=percepcao.assunto,
        teses_despertas=despertas,
        validade=validade,
        ha_metrica_faltando=ha_metrica_faltando,
    )

    executada = False
    erro: str | None = None
    capacidade = (capacidades or {}).get(decisao.acao)
    if decisao.age and capacidade is not None:
        try:
            capacidade(percepcao, estado)
            executada = True
        except Exception as exc:  # a ferramenta falha; o agente não morre
            erro = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "capacidade {a} falhou: {e}", a=decisao.acao.value, e=erro
            )

    registro = RegistroDeCiclo(
        percepcao=percepcao, decisao=decisao, executada=executada, erro=erro
    )
    novo = estado.com(nota=registro.linha_de_diario, momento=percepcao.momento)
    return novo, registro

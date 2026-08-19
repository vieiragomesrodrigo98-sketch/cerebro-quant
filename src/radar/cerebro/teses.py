"""
src/radar/cerebro/teses.py — o que o Cérebro está esperando acontecer.

Pergunta única: **"o que eu decidi aguardar, por quê, e o que precisa mudar para
eu reavaliar?"**

⚠️ `AGUARDAR` É TESE, NÃO STATUS
--------------------------------
Emenda 4 do DEV ao ADR 0037. Hoje `decisao.AGUARDAR` existe e **não sobrevive ao
ciclo**:

    dia 1 — "não há amostra suficiente. AGUARDAR."
    dia 2 — "não sei o que eu estava esperando."

Isso é pipeline sem memória operacional, e é a diferença concreta entre um
agente e uma função pura chamada repetidamente. Uma tese guarda o **motivo**, a
**condição de despertar** e a **amostra exigida** — declarados ANTES, e é o
"antes" que os torna verificáveis depois.

⚠️ E POR QUE ISSO É UM CONTROLE, NÃO UMA CONVENIÊNCIA
-----------------------------------------------------
G-P2 do Contrato de Pensamento: **mudez é FALHA, nunca abstenção**. A assimetria
que cria o risco está no ADR 0037 §5 — *abster custa zero, afirmar custa*. Se
abster fosse grátis **e invisível**, o agente tomaria a opção grátis em todo
lugar e pareceria seletivo enquanto simplesmente não decidia. Foi exatamente o
que aconteceu com o emissor v1: mudo por um mês, lido como criterioso.

Persistir a tese é o que torna a mudez **contável**. Um Cérebro com 400 teses
vivas e zero emissões não está sendo prudente — está travado, e agora dá para
ver.

O que este módulo NÃO faz
-------------------------
Não decide (é `politica.py`), não mede e não escolhe formato de armazenamento: o
repositório é injetado. `RepositorioDeTeses` é `Protocol` pela mesma razão que
todo o `cerebro` é — a catraca de camadas proíbe o pacote de conhecer
implementação, e é ela que mantém isto testável sem disco.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol


class EstadoDaTese(StrEnum):
    """Ciclo de vida. Cada saída é registrada — inclusive as que não dão em nada.

    Sem `EXPIRADA` e `ABANDONADA` distintas de `RESOLVIDA`, uma tese que morreu
    de velhice ficaria indistinguível de uma que foi respondida, e a contagem de
    mudez perderia o sentido.
    """

    AGUARDANDO = "aguardando"
    #: A condição de despertar ocorreu e a tese voltou para avaliação.
    DESPERTA = "desperta"
    #: Foi medida e virou conhecimento — o desfecho que se quer.
    RESOLVIDA = "resolvida"
    #: O prazo venceu sem a condição ocorrer. **Não é fracasso**: é a informação
    #: de que a condição declarada era rara ou impossível.
    EXPIRADA = "expirada"
    #: Descartada deliberadamente, com motivo.
    ABANDONADA = "abandonada"


ABERTAS: frozenset[EstadoDaTese] = frozenset(
    {EstadoDaTese.AGUARDANDO, EstadoDaTese.DESPERTA}
)


@dataclass(frozen=True, slots=True)
class Tese:
    """Uma espera declarada, com tudo o que a torna verificável depois.

    Todos os campos obrigatórios existem porque a ausência de cada um já
    produziu, neste projeto, um estado que ninguém conseguiu auditar.
    """

    identificador: str
    #: A hipótese ou o ativo × família a que a espera se refere.
    assunto: str
    #: **Por que** se está aguardando. Sem isto, aguardar é indistinguível de
    #: mudez — e é a distinção que o G-P2 cobra.
    motivo: str
    #: **O que precisa mudar** para reavaliar. Declarado antes, em texto que um
    #: humano confere. Condição vaga aqui vira tese imortal.
    condicao_de_despertar: str
    criada_em: datetime
    #: Prazo. `None` seria tese imortal, então **não é permitido**: o prazo é o
    #: que transforma a espera em evento, e sem ele o Cérebro nunca acorda
    #: sozinho para reavaliar.
    acordar_em: datetime
    #: Quantas observações independentes se espera precisar. `None` quando a
    #: espera não é por amostra (ex.: aguardando um evento de mercado).
    amostra_exigida: int | None = None
    amostra_atual: int = 0
    estado: EstadoDaTese = EstadoDaTese.AGUARDANDO
    #: Trilha do que aconteceu com ela. Append-only.
    historico: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.motivo.strip():
            raise ValueError(
                "tese sem motivo é mudez com nome melhor — G-P2 do Contrato de "
                "Pensamento trata mudez como FALHA, nunca abstenção"
            )
        if not self.condicao_de_despertar.strip():
            raise ValueError(
                "tese sem condição de despertar nunca é reavaliada e vira espera "
                "eterna. Declare o que precisa mudar."
            )
        if self.acordar_em <= self.criada_em:
            raise ValueError("o prazo de despertar precisa ser depois da criação")
        if self.amostra_exigida is not None and self.amostra_exigida < 1:
            raise ValueError("amostra exigida é contagem positiva ou None")

    @property
    def aberta(self) -> bool:
        return self.estado in ABERTAS

    @property
    def amostra_suficiente(self) -> bool:
        """`True` também quando não se espera por amostra — nesse caso a espera
        é por evento, e a amostra não é o gargalo."""
        if self.amostra_exigida is None:
            return True
        return self.amostra_atual >= self.amostra_exigida

    def vencida_em(self, agora: datetime) -> bool:
        return agora >= self.acordar_em

    def com(self, *, nota: str, **campos: object) -> Tese:
        """Nova versão da tese com uma nota no histórico.

        Devolve cópia porque `Tese` é imutável: reescrever uma tese no lugar
        apagaria o que ela dizia antes, e é justamente o "o que ela dizia antes"
        que permite auditar se a espera fazia sentido.
        """
        return replace(self, historico=(*self.historico, nota), **campos)  # type: ignore[arg-type]


class RepositorioDeTeses(Protocol):
    """Onde as teses vivem entre ciclos. Injetado."""

    def salvar(self, tese: Tese) -> None: ...  # pragma: no cover — contrato

    def abertas(self) -> Sequence[Tese]: ...  # pragma: no cover — contrato


class TesesEmMemoria:
    """Implementação de referência — e a que os testes usam.

    Fica aqui, e não em `tests/`, porque também serve ao Cérebro rodando em
    modo de pesquisa, onde persistir não faz sentido.
    """

    def __init__(self, iniciais: Iterable[Tese] = ()) -> None:
        self._por_id: dict[str, Tese] = {t.identificador: t for t in iniciais}

    def salvar(self, tese: Tese) -> None:
        self._por_id[tese.identificador] = tese

    def abertas(self) -> Sequence[Tese]:
        return [t for t in self._por_id.values() if t.aberta]

    def todas(self) -> Sequence[Tese]:
        return list(self._por_id.values())


def agora_utc() -> datetime:
    return datetime.now(UTC)


def despertar_vencidas(
    repositorio: RepositorioDeTeses, agora: datetime
) -> Iterator[Tese]:
    """As teses cujo prazo venceu — **o evento** que acorda o Cérebro.

    Devolve e **persiste** como `DESPERTA`. O agente decide depois o que fazer
    com cada uma: medir, continuar aguardando com prazo novo, ou abandonar. Este
    módulo não escolhe — a escolha é de `politica.py`.
    """
    for tese in list(repositorio.abertas()):
        if tese.estado is EstadoDaTese.AGUARDANDO and tese.vencida_em(agora):
            desperta = tese.com(
                estado=EstadoDaTese.DESPERTA,
                nota=f"{agora.isoformat()}: prazo venceu; amostra {tese.amostra_atual}"
                + (f"/{tese.amostra_exigida}" if tese.amostra_exigida else ""),
            )
            repositorio.salvar(desperta)
            yield desperta


def resumo_de_mudez(teses: Iterable[Tese]) -> dict[str, int]:
    """Quantas esperas existem, por estado.

    É a contagem que torna a mudez visível. Um Cérebro com centenas de
    `AGUARDANDO` e nada resolvido não está sendo seletivo — está travado, e o
    número é o que permite distinguir os dois. Publicar sempre todos os estados,
    inclusive os zerados, é deliberado: chave ausente e valor zero têm de ser a
    mesma coisa para quem lê o painel.
    """
    contagem = dict.fromkeys((e.value for e in EstadoDaTese), 0)
    for tese in teses:
        contagem[tese.estado.value] += 1
    return contagem

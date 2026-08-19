"""
src/radar/cerebro/politica.py — o que o Cérebro PODE fazer.

Pergunta única: **"dado estado + evento + teses + orçamento + mapa de validade,
quais ações são permitidas?"**

⚠️ AGÊNCIA E GUARDRAIL SÃO COISAS DIFERENTES
--------------------------------------------
Emenda 1 do DEV ao ADR 0037. Esta pergunta estava implícita e espalhada, e não
pertence a `agente.py`:

    agente.py    → o que eu QUERO fazer
    politica.py  → o que eu POSSO fazer

Não é um segundo cérebro. É a fronteira, e mantê-la num módulo **sem estado** é
o que a torna testável isolada — que é a diferença prática entre um guardrail e
uma intenção. Guardrail que só existe dentro do fluxo que ele governa é
verificável apenas rodando o fluxo inteiro, e por isso nunca é verificado.

A frase que resume a doutrina inteira:

> **Autonomia operacional total. Liberdade metodológica zero.**

O agente escolhe *o que fazer*. Ele não decide que a régua era outra depois de
ver o resultado.

⚠️ EVENTO NÃO É ORDEM
---------------------
Emenda 7. Nunca `BARRA_FECHOU → EXECUTAR`:

    BARRA_FECHOU → Cérebro → estado? tese? regime? validade? risco? orçamento?
                           → AÇÃO / AGUARDAR / IGNORAR

`NENHUMA_ACAO` é resposta legítima **e é registrada**. Se ignorar fosse grátis e
invisível, o agente ignoraria em todo lugar e pareceria seletivo — a mesma
mudez do emissor v1, com nome melhor (G-P2).

O que este módulo NÃO faz
-------------------------
Não age, não mede, não persiste e não tem estado. Recebe tudo por parâmetro.
Nenhuma função aqui pode escrever em lugar nenhum — se pudesse, o guardrail
teria efeito colateral e deixaria de ser auditável por leitura.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from radar.cerebro.orcamento import Orcamento
from radar.cerebro.teses import Tese


class Evento(StrEnum):
    """O que aconteceu. **Nunca o que fazer** — ver a emenda 7."""

    BARRA_1S = "barra_1s"
    BARRA_DIARIA = "barra_diaria"
    DESFECHO_NO_LEDGER = "desfecho_no_ledger"
    PRAZO_DE_TESE = "prazo_de_tese"
    NOTICIA_CLASSIFICADA = "noticia_classificada"


class Missao(StrEnum):
    """Emenda 5 — dois problemas temporais, um Cérebro.

    Não são modos de configuração: são objetivos com prioridades incompatíveis.
    PESQUISA otimiza validade e aceita esperar; EXECUÇÃO otimiza latência e não
    pode. Fundir as duas produziria um sistema que mede mal e opera devagar.
    """

    PESQUISA = "pesquisa"
    EXECUCAO = "execucao"


class Acao(StrEnum):
    """O espaço de ação — maior que "emitir sinal", e é essa a diferença entre
    pipeline e agente (ADR 0037 §5)."""

    #: Completar medição sobre dado que já existe. **Não custa multiplicidade** —
    #: não é tentativa nova, é terminar de olhar o que já foi coletado.
    MEDIR_METRICA = "medir_metrica"
    #: A tese vale, a amostra ainda não chegou. Renova o prazo.
    AGUARDAR_AMOSTRA = "aguardar_amostra"
    #: A tese despertou com amostra suficiente: hora de julgá-la.
    AVALIAR_TESE = "avaliar_tese"
    #: Há vantagem líquida, gatilho e validade. Só na missão de EXECUÇÃO.
    EMITIR_SINAL = "emitir_sinal"
    #: Desfecho chegou; o conhecimento muda. É o laço de retorno do ADR 0036.
    ATUALIZAR_ESTADO = "atualizar_estado"
    #: Nada a fazer — **e isso é registrado**.
    NENHUMA_ACAO = "nenhuma_acao"


@dataclass(frozen=True, slots=True)
class Validade:
    """O que o mapa de validade diz sobre um assunto."""

    assunto: str
    #: FDR é PORTÃO de promoção desde 2026-08-12 (decisão do DEV). Sem isto,
    #: `EMITIR_SINAL` nunca é permitido.
    sobrevive_fdr: bool
    estado: str = "nao_comprovado"


class MapaDeValidade(Protocol):
    """Injetado — `cerebro` não conhece o formato em disco."""

    def consultar(self, assunto: str) -> Validade | None: ...  # pragma: no cover


@dataclass(frozen=True, slots=True)
class AcaoPermitida:
    """A resposta da política. `motivo` é **obrigatório**.

    Ação sem motivo é decisão sem rastro, e a autópsia de um erro passa a
    depender de reconstruir o que o sistema estava pensando — que é exatamente
    o que o v1 não permitiu fazer.
    """

    acao: Acao
    motivo: str
    assunto: str | None = None

    def __post_init__(self) -> None:
        if not self.motivo.strip():
            raise ValueError("ação sem motivo é decisão sem rastro")

    @property
    def age(self) -> bool:
        return self.acao is not Acao.NENHUMA_ACAO


class HipoteseImutavelError(RuntimeError):
    """Tentou mexer na definição de hipótese de um lote em execução.

    É a brecha pela qual autonomia viraria garimpo automatizado, e por isso é
    exceção e não aviso.
    """


def exigir_hipotese_imutavel(
    *, hipotese: str, lote_em_execucao: Sequence[str] | None
) -> None:
    """Emenda 3 — hipótese não muda depois que o lote começou a rodar.

    O agente pode escolher, priorizar, aguardar, medir e registrar. **Não pode
    alterar retrospectivamente a definição de uma hipótese sob avaliação**, nem
    criar hipótese nova dentro do mesmo experimento depois de ver resultado.
    Ideia nova segue o caminho longo:

        nova hipótese → fila de pesquisa → pré-registro → NOVO lote → teste

    Sem esta trava, a sequência que destrói qualquer inferência é trivial de
    executar sem má-fé nenhuma: rodar, olhar, ajustar o corte, rodar de novo — e
    o pré-registro vira decoração, porque o que foi registrado não é o que foi
    testado.
    """
    if lote_em_execucao and hipotese in lote_em_execucao:
        raise HipoteseImutavelError(
            f"`{hipotese}` está no lote em execução e sua definição não pode mudar. "
            "Mande a ideia para a fila de pesquisa e pré-registre um lote NOVO — "
            "ajustar hipótese depois de ver resultado é garimpo, e acha alpha "
            "falso com probabilidade ~1 (ADR 0031 §4)."
        )


def proxima_acao(
    *,
    evento: Evento,
    missao: Missao,
    orcamento: Orcamento,
    assunto: str | None = None,
    teses_despertas: Sequence[Tese] = (),
    validade: Validade | None = None,
    ha_metrica_faltando: bool = False,
) -> AcaoPermitida:
    """A única porta de decisão do Cérebro.

    A ordem das cláusulas **é** a política, e cada uma existe por uma razão
    medida:

    1. **Tese desperta vem primeiro.** O que já se decidiu esperar tem
       precedência sobre o que acabou de acontecer — senão o Cérebro fica
       perpetuamente reagindo ao mais recente e nunca fecha nada. É a diferença
       entre ter missão e ter reflexo.
    2. **Desfecho atualiza o estado**, sempre. É o laço de retorno do ADR 0036,
       o que hoje não existe (o Ledger produz mapa e o motor nunca lê).
    3. **Medir métrica faltante é grátis** e vem antes de qualquer coisa que
       custe: `pesquisa.avaliar_necessidade` já distingue *falta métrica* (medir
       já, custo zero) de *falta amostra* (esperar, custa FDR).
    4. **Emitir exige missão de EXECUÇÃO e sobrevivência ao FDR.** As duas
       condições, sempre.
    5. **`NENHUMA_ACAO` com motivo** — nunca `None`, nunca silêncio.
    """
    for tese in teses_despertas:
        if tese.amostra_suficiente:
            return AcaoPermitida(
                acao=Acao.AVALIAR_TESE,
                motivo=f"tese desperta com amostra suficiente ({tese.amostra_atual}"
                + (f"/{tese.amostra_exigida}" if tese.amostra_exigida else "")
                + f"): {tese.motivo}",
                assunto=tese.assunto,
            )
        return AcaoPermitida(
            acao=Acao.AGUARDAR_AMOSTRA,
            motivo=f"prazo venceu e a amostra não chegou ({tese.amostra_atual}"
            + (f"/{tese.amostra_exigida}" if tese.amostra_exigida else "")
            + f"): {tese.condicao_de_despertar}",
            assunto=tese.assunto,
        )

    if evento is Evento.DESFECHO_NO_LEDGER:
        return AcaoPermitida(
            acao=Acao.ATUALIZAR_ESTADO,
            motivo="desfecho real chegou ao Ledger; o conhecimento muda antes de "
            "qualquer nova decisão (laço de retorno, ADR 0036)",
            assunto=assunto,
        )

    if ha_metrica_faltando:
        return AcaoPermitida(
            acao=Acao.MEDIR_METRICA,
            motivo="há métrica não calculada sobre dado que JÁ existe — medir não "
            "é tentativa nova e não custa multiplicidade",
            assunto=assunto,
        )

    if missao is Missao.EXECUCAO:
        if validade is None:
            return AcaoPermitida(
                acao=Acao.NENHUMA_ACAO,
                motivo=f"sem entrada no mapa de validade para `{assunto}` — emitir "
                "sobre assunto não medido é afirmar sem evidência",
                assunto=assunto,
            )
        if not validade.sobrevive_fdr:
            return AcaoPermitida(
                acao=Acao.NENHUMA_ACAO,
                motivo=f"`{validade.assunto}` não sobrevive ao FDR (estado "
                f"{validade.estado}); FDR é portão de promoção desde 2026-08-12",
                assunto=assunto,
            )
        return AcaoPermitida(
            acao=Acao.EMITIR_SINAL,
            motivo=f"`{validade.assunto}` sobrevive ao FDR e o evento {evento.value} "
            "abriu janela de execução",
            assunto=assunto,
        )

    if orcamento.esgotado:
        return AcaoPermitida(
            acao=Acao.NENHUMA_ACAO,
            motivo=f"orçamento estatístico esgotado ({orcamento}) — tentativa nova "
            "derrubaria o alvo por multiplicidade",
            assunto=assunto,
        )

    return AcaoPermitida(
        acao=Acao.NENHUMA_ACAO,
        motivo=f"evento {evento.value} não abriu lacuna: nenhuma tese desperta, "
        "nenhuma métrica faltando, missão de pesquisa sem alvo imediato",
        assunto=assunto,
    )

"""
src/radar/cerebro/decisao.py — o estágio SIGNAL DECISION.

Pergunta única: **"devemos emitir o sinal?"**

⚠️ `WAIT` É DECISÃO REGISTRADA, NUNCA SILÊNCIO
----------------------------------------------
Trava do ADR 0036, e ela existe por um episódio caro: o emissor v1 passou **um
mês inteiro sem emitir** em produção, e isso foi lido como seletividade em vez
de defeito. O portão **G-P2** do Contrato de Pensamento diz que *mudez é FALHA,
nunca abstenção*.

`WAIT` e `NO_SIGNAL` são dois modos de não operar. Se ambos forem estados
livres e não medidos, o Cérebro ganha uma **opção grátis**: deixa de errar
porque deixa de decidir. Por isso toda decisão — inclusive as de não operar —
sai daqui como `DecisaoRegistrada`, com o motivo e a expectativa que a
sustentou, e vai para avaliação junto com as demais.

A diferença entre os dois não é de grau, é de natureza:

| saída | significa | volta a ser avaliada? |
|---|---|---|
| `COMPRAR` / `VENDER` | há vantagem líquida e o momento é agora | vira posição |
| `AGUARDAR` | a tese vale, o **momento** não | **sim**, nos próximos ciclos |
| `SEM_SINAL` | não há vantagem líquida nesta configuração | não, até a situação mudar |
| `INVALIDADO` | a tese que existia **deixou** de valer | não, encerrada |

O que este módulo NÃO faz
-------------------------
Não dimensiona posição (é do Portfolio/`alocacao.py`), não ordena entre
candidatos (é `ranking.py`) e **não reavalia expectativa** — ela chega pronta.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from radar.cerebro.deteccao import Candidato
from radar.cerebro.expectativa import Expectativa
from radar.cerebro.mapa_validade import Direcao


class Decisao(StrEnum):
    """As quatro saídas possíveis, mais a invalidação.

    Nomes em português por coerência com o pacote; correspondem a
    `BUY / SELL / WAIT / NO_SIGNAL / INVALIDATED` do desenho do DEV.
    """

    COMPRAR = "comprar"
    VENDER = "vender"
    AGUARDAR = "aguardar"
    SEM_SINAL = "sem_sinal"
    INVALIDADO = "invalidado"


#: As saídas que NÃO viram ordem. Nomeadas para que "não operar" seja um
#: conjunto explícito — contar quantas decisões caíram aqui é o que impede a
#: mudez de se disfarçar de seletividade.
NAO_OPERA: frozenset[Decisao] = frozenset(
    {Decisao.AGUARDAR, Decisao.SEM_SINAL, Decisao.INVALIDADO}
)


@dataclass(frozen=True, slots=True)
class DecisaoRegistrada:
    """Uma decisão COM rastro. Toda decisão vira uma destas — inclusive as de
    não operar.

    `motivo` é obrigatório e livre de propósito: enum de motivos convidaria a
    escolher o rótulo mais próximo em vez de escrever o que aconteceu, e a
    autópsia depende do que aconteceu.
    """

    ativo_id: str
    decisao: Decisao
    motivo: str
    expectativa: Expectativa | None
    chave_de_particao: str

    def __post_init__(self) -> None:
        if not self.motivo.strip():
            raise ValueError(
                "toda decisão exige motivo — decisão sem motivo é silêncio com "
                "nome de decisão, e G-P2 proíbe (mudez é FALHA, nunca abstenção)"
            )

    @property
    def opera(self) -> bool:
        return self.decisao not in NAO_OPERA


def decidir(
    candidato: Candidato,
    expectativa: Expectativa,
    *,
    direcao: Direcao,
    momento_favoravel: bool,
    tese_ainda_vale: bool = True,
) -> DecisaoRegistrada:
    """Decide sobre UM candidato já avaliado.

    A ordem das perguntas é a doutrina, e ela importa:

    1. **A tese ainda vale?** Se caiu, é `INVALIDADO` — e isso vem antes de
       qualquer conta, porque avaliar vantagem de uma tese morta é gastar
       cálculo para chegar num número que não significa nada.
    2. **Há vantagem líquida distinguível?** Se não, é `SEM_SINAL`. Note que a
       pergunta é sobre o EV **líquido** e a margem sobre o custo — não sobre a
       probabilidade de acerto, que sozinha não decide nada.
    3. **O momento é agora?** Se a tese vale e a vantagem existe mas o gatilho
       não veio, é `AGUARDAR` — e volta nos próximos ciclos. Separar *"tese
       válida"* de *"momento adequado"* é o estágio TIMING; aqui só se consome
       o veredito dele.

    ⚠️ A DIREÇÃO É PARÂMETRO, e isto foi um bug corrigido na revisão da Fase 2.
    A primeira versão fazia `COMPRAR if payoff_ganho >= payoff_perda else
    VENDER` — **errado**, e errado exatamente na classe que o ADR 0034 bloqueou.
    Assimetria de payoff não tem relação com direção: um short com
    `payoff_ganho=0,10` e `payoff_perda=0,02` sairia classificado como COMPRAR.

    Direção é propriedade da **hipótese**, não da aritmética: é a família/braço
    que declara se opera comprado ou vendido, e o `#base` do `swing_v1` grava
    `Direcao.LONG` justamente porque isso precisou ser declarado, não inferido.
    Quem chama informa; este módulo não adivinha.
    """
    particao = candidato.chave_de_particao
    if not tese_ainda_vale:
        return DecisaoRegistrada(
            ativo_id=candidato.ativo_id,
            decisao=Decisao.INVALIDADO,
            motivo="a tese que sustentava a oportunidade deixou de valer",
            expectativa=expectativa,
            chave_de_particao=particao,
        )

    if not expectativa.economicamente_distinguivel:
        motivo = (
            f"custo comeu o edge (bruto={expectativa.bruto:+.5f}, "
            f"custo={expectativa.custo:.5f})"
            if expectativa.custo_comeu_o_edge
            else f"sem vantagem líquida (ev={expectativa.ev:+.5f})"
        )
        return DecisaoRegistrada(
            ativo_id=candidato.ativo_id,
            decisao=Decisao.SEM_SINAL,
            motivo=motivo,
            expectativa=expectativa,
            chave_de_particao=particao,
        )

    if not momento_favoravel:
        return DecisaoRegistrada(
            ativo_id=candidato.ativo_id,
            decisao=Decisao.AGUARDAR,
            motivo=(
                f"tese válida e vantagem líquida (ev={expectativa.ev:+.5f}), "
                "mas o gatilho de entrada não ocorreu"
            ),
            expectativa=expectativa,
            chave_de_particao=particao,
        )

    if direcao is Direcao.LONG:
        lado = Decisao.COMPRAR
    elif direcao is Direcao.SHORT:
        lado = Decisao.VENDER
    else:
        # NEUTRO/DESCONHECIDA não viram ordem. Emitir com direção indefinida é
        # o que o ADR 0034 chamou de bloqueio metodológico — e ele custou uma
        # trava de projeto inteira até a semântica ser declarada.
        return DecisaoRegistrada(
            ativo_id=candidato.ativo_id,
            decisao=Decisao.SEM_SINAL,
            motivo=(
                f"há vantagem líquida (ev={expectativa.ev:+.5f}) mas a direção é "
                f"{direcao.value} — ordem exige direção declarada (ADR 0034)"
            ),
            expectativa=expectativa,
            chave_de_particao=particao,
        )

    return DecisaoRegistrada(
        ativo_id=candidato.ativo_id,
        decisao=lado,
        motivo=(
            f"vantagem líquida ev={expectativa.ev:+.5f}, "
            f"margem sobre custo {expectativa.margem_sobre_custo:.2f}×, "
            f"direção {direcao.value}, gatilho ocorrido"
        ),
        expectativa=expectativa,
        chave_de_particao=particao,
    )


def resumo_de_mudez(decisoes: list[DecisaoRegistrada]) -> dict[str, int]:
    """Quantas decisões caíram em cada saída — a métrica que impede a mudez de
    se disfarçar de seletividade (G-P2).

    Um ciclo em que **tudo** cai em `NAO_OPERA` não é necessariamente errado,
    mas precisa ser **visível**. Foi a invisibilidade, e não a abstenção, que
    deixou o emissor v1 mudo por um mês sem ninguém notar.
    """
    contagem = {d.value: 0 for d in Decisao}
    for registro in decisoes:
        contagem[registro.decisao.value] += 1
    contagem["_total"] = len(decisoes)
    contagem["_opera"] = sum(1 for r in decisoes if r.opera)
    return contagem

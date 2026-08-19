"""
src/radar/cerebro/orcamento.py — quanto poder estatístico ainda existe.

Pergunta única: **"cabe mais uma tentativa sem matar o que já sabemos?"**

⚠️ ORÇAMENTO É DERIVADO, NUNCA CONTADOR
---------------------------------------
Emenda 2 do DEV ao ADR 0037, e ela é uma **correção técnica**, não uma
preferência de estilo. `restante -= 1` estaria errado porque o espaço admissível
do Benjamini-Hochberg depende da **distribuição** dos p-valores, não da
contagem:

- uma tentativa **nula** custa espaço — empurra o limiar de todo mundo para
  baixo;
- uma tentativa que **acha algo forte** DEVOLVE espaço, porque entra no topo da
  ordenação e ergue a escada inteira.

Um contador não consegue representar a segunda metade dessa frase. Por isso a
função central recalcula do estado, e não existe operação de decremento neste
módulo.

O número, e de onde ele sai
---------------------------
`estrategia_alfa` é o **único achado positivo do projeto**. Ele está no **posto 5**
do BH com `p = 0,001389`, e sobrevive enquanto `p₅ ≤ (5/m)·q`, isto é
**`m ≤ 180`**. Com `m = 57` medido hoje, **restam 123**.

⚠️ E há uma dependência que parece paradoxo e não é: os postos 1 a 4 são cripto
fortemente **negativo** (−7,98, −4,96, −4,36, −3,71). São eles que erguem a
escada até o posto 5. Sem eles, `estrategia_alfa` viraria posto 1, precisaria de
`p ≤ q/53 = 0,00094` e **falharia**. O que parece o fracasso do projeto é o que
sustenta seu único acerto — e é a razão de o alvo ser calculado sobre a grade
INTEIRA, refutações incluídas, nunca sobre o subconjunto que interessa.

O que este módulo NÃO faz
-------------------------
Não lê arquivo, não conhece o mapa de validade e não decide o que medir. Recebe
a grade por parâmetro, como todo o `cerebro` — a catraca de camadas proíbe o
contrário, e é ela que mantém isto testável sem banco.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

#: Taxa de descoberta falsa. É a mesma do mapa de validade e do ADR 0031 §4 —
#: dois valores de `q` vivos produziriam duas verdades sobre a mesma grade.
Q_PADRAO: Final[float] = 0.05

#: Teto quando não há nenhuma hipótese positiva para proteger. **Zero, não
#: infinito.** Sem alvo o orçamento não tem significado, e devolver "ilimitado"
#: transformaria ausência de conhecimento em licença para gastar — que é
#: exatamente o movimento que o ADR 0031 existe para impedir.
SEM_ALVO_NAO_HA_ORCAMENTO: Final[int] = 0


@dataclass(frozen=True, slots=True)
class HipoteseNaGrade:
    """Uma tentativa que já entrou no denominador.

    `p_valor=None` é **não-estimável**, e ela continua contando: o ADR 0032 fixou
    que motor novo não zera o contador, e hipótese que não pôde ser medida
    consumiu tentativa do mesmo jeito. Ela entra como `1.0` no cálculo — o pior
    caso, que nunca é rejeitada e só ocupa espaço.
    """

    nome: str
    p_valor: float | None = None
    t: float | None = None

    @property
    def p_efetivo(self) -> float:
        return 1.0 if self.p_valor is None else self.p_valor

    @property
    def positiva(self) -> bool:
        """Efeito na direção que vira produto.

        Refutação forte é conhecimento valioso e **não** é candidata a alvo: não
        se protege poder estatístico para continuar podendo afirmar que algo não
        funciona. `t=None` não é positiva — ausência de medição não é evidência
        favorável.
        """
        return self.t is not None and self.t > 0.0


@dataclass(frozen=True, slots=True)
class Orcamento:
    """A fotografia do espaço estatístico num instante."""

    #: Quantas tentativas cabem no total antes de `alvo` cair.
    teto: int
    #: Quantas já foram gastas — o tamanho da grade.
    gasto: int
    #: Nome da hipótese que está sendo protegida, se houver.
    alvo: str | None
    #: `q` usado. Publicado porque um orçamento sem o `q` que o gerou é número
    #: sem unidade.
    q: float

    @property
    def restante(self) -> int:
        return max(0, self.teto - self.gasto)

    @property
    def esgotado(self) -> bool:
        return self.restante <= 0

    def __str__(self) -> str:
        alvo = self.alvo or "nenhum alvo positivo"
        return f"teto {self.teto} · gasto {self.gasto} · restante {self.restante} (protegendo: {alvo}, q={self.q})"


class OrcamentoEsgotadoError(RuntimeError):
    """Registrar mais uma hipótese mataria o alvo por multiplicidade.

    Levantar em vez de devolver `False` é deliberado: a emenda 2 exige que a
    recusa seja **visível**. Um `if` ignorado silenciosamente reproduziria a
    família de defeito que o projeto já pagou — a defesa que existe, não dispara,
    e é lida como saúde.
    """


def _postos_bh_rejeitados(p_ordenados: Sequence[float], q: float) -> int:
    """`k = max{i : p₍ᵢ₎ ≤ (i/m)·q}` — o degrau mais alto que a escada alcança.

    Procedimento *step-up*: achado o `k`, **todos** os postos até ele são
    rejeitados, inclusive os que individualmente falhariam. É por isso que um
    achado forte no topo carrega os de baixo junto.
    """
    m = len(p_ordenados)
    if m == 0:
        return 0
    k = 0
    for i, p in enumerate(p_ordenados, start=1):
        if p <= (i / m) * q:
            k = i
    return k


def escolher_alvo(grade: Iterable[HipoteseNaGrade], *, q: float = Q_PADRAO) -> HipoteseNaGrade | None:
    """A melhor hipótese POSITIVA que hoje sobrevive ao FDR.

    É ela que o orçamento protege, porque é a única classe que pode virar
    produto. Devolve `None` quando não há nenhuma — e nesse caso não há orçamento
    a defender, o que é informação, não erro.
    """
    itens = sorted(grade, key=lambda h: h.p_efetivo)
    k = _postos_bh_rejeitados([h.p_efetivo for h in itens], q)
    for h in itens[:k]:
        if h.positiva:
            return h
    return None


def espaco_restante(
    grade: Iterable[HipoteseNaGrade],
    *,
    alvo: str | None = None,
    q: float = Q_PADRAO,
) -> Orcamento:
    """Quantas tentativas NULAS ainda cabem antes de `alvo` deixar de ser rejeitado.

    Uma hipótese nula entra com `p = 1,0`, portanto vai para o fim da ordenação e
    **não muda o posto do alvo** — só aumenta `m`. O alvo, no posto `r`,
    sobrevive se existir algum posto `i ≥ r` com `p₍ᵢ₎ ≤ (i/(m+n))·q`, o que dá:

        n ≤ i·q/p₍ᵢ₎ − m

    e o espaço é o **máximo sobre todo `i ≥ r`**, não apenas `i = r`. A diferença
    importa: às vezes um achado mais fraco alguns postos abaixo sustenta a escada
    por mais tempo que o próprio alvo, e olhar só para `r` subestimaria o
    orçamento — apertando a régua além do que a matemática pede.

    Tentativa que ACHA algo forte não é contabilizada aqui, e é justamente esse o
    ponto: ela muda a grade, e a próxima chamada devolve um número **maior**.
    Nunca decrementamos.
    """
    itens = sorted(grade, key=lambda h: h.p_efetivo)
    m = len(itens)
    escolhido = (
        next((h for h in itens if h.nome == alvo), None)
        if alvo is not None
        else escolher_alvo(itens, q=q)
    )
    if escolhido is None:
        return Orcamento(teto=SEM_ALVO_NAO_HA_ORCAMENTO, gasto=m, alvo=None, q=q)

    posto_do_alvo = itens.index(escolhido) + 1
    teto = 0
    for i in range(posto_do_alvo, m + 1):
        p = itens[i - 1].p_efetivo
        if p <= 0.0:
            # p exatamente zero não é medição, é subfluxo de ponto flutuante num
            # `t` enorme. Tratar como "cabe tudo" seria deixar o orçamento
            # depender de um artefato numérico; ignora-se o posto.
            continue
        cabe = int(i * q / p)
        teto = max(teto, cabe)
    return Orcamento(teto=teto, gasto=m, alvo=escolhido.nome, q=q)


def exigir_espaco(
    grade: Iterable[HipoteseNaGrade],
    *,
    quantas: int = 1,
    alvo: str | None = None,
    q: float = Q_PADRAO,
) -> Orcamento:
    """Autoriza `quantas` tentativas novas, ou **levanta**.

    A porta única por onde o agente registra hipótese. Chamar isto é o que
    transforma o controle nº 6 do ADR 0031 (*penalização por tentativas*) de
    parágrafo em mecanismo.
    """
    orcamento = espaco_restante(grade, alvo=alvo, q=q)
    if orcamento.restante < quantas:
        raise OrcamentoEsgotadoError(
            f"pedido de {quantas} tentativa(s) e só há {orcamento.restante}. "
            f"{orcamento}. Passar disso derruba `{orcamento.alvo}` do FDR por "
            "multiplicidade — e ela é o achado que sustenta o resto. "
            "Corte escopo ANTES de medir; afrouxar a régua depois de ver o "
            "resultado é o que o ADR 0031 existe para impedir."
        )
    return orcamento

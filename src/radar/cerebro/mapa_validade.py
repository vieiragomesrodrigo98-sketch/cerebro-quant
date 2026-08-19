"""
src/radar/cerebro/mapa_validade.py — o EVIDENCE MAP do Cérebro.

O que este módulo é, na arquitetura do DEV (04/08/2026)
--------------------------------------------------------
É a camada `EVIDÊNCIA OOS → VALIDADE ESTATÍSTICA → CUSTO → CAPACIDADE` que fica
ENTRE as famílias de modelo e o Strategy Allocator. Ele não escolhe estratégia:
ele responde *quais fontes de edge têm evidência suficiente, e em que zona*.

    "O objetivo do Cérebro não é provar que existe uma estratégia vencedora. O
    objetivo é construir um sistema de seleção e alocação entre fontes de edge,
    onde cada especialista tem uma zona de validade conhecida."

Consequência direta e libertadora: **família que falha não mata o projeto — sai
do pool.** Família que ainda não pode ser medida fica em observação. Só a
família validada recebe capital. Este módulo é o que torna essas três frases
executáveis em vez de retóricas.

O erro que este módulo existe para impedir
-------------------------------------------
O anti-padrão, nas palavras do DEV:

    Mercado → "qual estratégia está funcionando melhor agora?" → escolher a
    vencedora.

Isso é **overfitting adaptativo**: seleciona o vencedor do ruído recente e
chama de adaptação. A pergunta correta é outra, e a ordem dela é o módulo
inteiro:

    Quais têm evidência VÁLIDA? → quais estão dentro da zona de validade? →
    qual a expectativa LÍQUIDA? → qual tem capacidade de execução? → como
    distribuir capital entre as válidas?

Note que "ganhou recentemente" não aparece em lugar nenhum dessa cadeia.

Por que o `verde` do Ledger NÃO é `VALIDADO` — e isto é um achado, não um detalhe
---------------------------------------------------------------------------------
O `Celula.veredito` do `radar.evidence.ledger` é **in-sample**: as facas 1-6 são
aplicadas sobre o mesmo período que gerou a hipótese, e a faca 7 (FDR) controla
multiplicidade dentro daquele período — não vaza para fora dele.

Medido no Ledger Técnico em 2026-08-04: **252 células verdes in-sample, ZERO
sobrevivem ao walk-forward independente** (melhor `t` OOS de todo o projeto:
**+1,34**, contra a régua de 2,0). O rótulo "verde" estava prometendo o que não
tinha.

Por isso os dois níveis são **deliberadamente separados** e o vocabulário é
diferente em cada um:

| nível | quem produz | vocabulário | significa |
|---|---|---|---|
| 1 · célula, in-sample | `evidence.ledger` | verde/amarelo/vermelho/⚪ | sobreviveu às facas NAQUELE período |
| 2 · hipótese, OOS | **este módulo** | VALIDADO/NÃO COMPROVADO/REFUTADO/NÃO ESTIMÁVEL | pode ou não receber capital |

O nível 1 continua existindo e continua necessário: é ele que o walk-forward usa
**dentro de cada ano de treino** para escolher o que testar no ano seguinte.
Fundi-los seria circular — o walk-forward precisa de um veredito in-sample para
poder produzir o OOS. O que não se pode é chamar o nível 1 de validação.

A distinção REFUTADO × NÃO COMPROVADO exige PODER, não só `t`
-------------------------------------------------------------
Esta é a parte que faz a taxonomia funcionar em vez de virar carimbo. `t` baixo
tem duas causas completamente diferentes:

* **o efeito não existe** — e a amostra era grande o bastante para tê-lo visto;
* **a amostra era pequena demais** — o efeito pode existir e não haver poder.

Confundir as duas é o erro que mata hipóteses boas e mantém hipóteses mortas, na
mesma tacada. A separação é mecânica: com erro-padrão `ep = excesso / t`, o
menor efeito que a amostra conseguiria detectar na régua é

    efeito_minimo_detectavel = T_MINIMO × ep

Se `efeito_minimo_detectavel <= efeito_minimo_relevante` (o menor retorno que
valeria a pena operar, **declarado antes**), então o teste TINHA poder para achar
o que interessa e não achou → **REFUTADO**. Se não tinha, o resultado negativo
não decide nada → **NÃO COMPROVADO**, e a ação é coletar mais dado, não enterrar.

Aplicado aos casos reais do projeto (medidos em 2026-08-04):

* `p9c57_squeeze` — 1.464 dias, `t` +0,14, líquido +0,044%/trade. `ep` = 0,311%
  → só detectaria **0,622%/trade**. Um efeito 14× maior do que o medido.
  **NÃO tinha poder → NÃO COMPROVADO**, apesar da amostra ser a maior do
  projeto.
* `p9c59_contracao_atr_saida` — 160 dias, `t` +1,34, `ep` = 0,446% → detectaria
  0,892%. **Também sem poder → NÃO COMPROVADO**, e a ação é paper trading +
  coleta.

O achado incômodo que isso revela
----------------------------------
Reescrever a fórmula deixa claro por que nada é refutável aqui:

    efeito_minimo_detectavel / efeito_medido  =  2 / |t|

Logo, **"a amostra teve poder" e "`t` chegou perto de 2" são a mesma
afirmação**, e o fator de amostra que falta é `(2/t)²`. Medido nas estratégias
de sinal positivo do projeto:

| estratégia | dias OOS | `t` | fator | dias necessários |
|---|---|---|---|---|
| `p9c59_contracao_atr_saida` | 160 | +1,34 | **2,2×** | **357** (~1,4 ano de B3) |
| `p8c43_volume_obv` | 404 | +0,68 | 8,7× | 3.523 (14 anos) |
| `p9c56_bandeira` | 299 | +0,40 | 25× | 7.533 (30 anos) |
| `p9c57_squeeze` | 1.464 | +0,14 | **203×** | 296.992 (**1.178 anos**) |

Duas consequências que mudam a estratégia do projeto:

1. **`REFUTADO` é quase inalcançável neste dado.** Com técnico diário na B3, o
   sistema consegue dizer "não provei" mas quase nunca "provei que não". Por
   isso `REFUTADO` exige `efeito_minimo_relevante` declarado: sem ele o estado
   nunca dispara, e é assim que deve ser.
2. **`p9c59_contracao_atr_saida` é a única hipótese do projeto com caminho
   viável** — 2,2× de amostra, não 200×. É o caso de manual de `NÃO
   COMPROVADO → paper trading + coleta → nova validação`.

E sustenta cripto como primeiro laboratório por um motivo aritmético, não de
preferência: 365 dias/ano contra 252, e horizontes intradiários que multiplicam
observações independentes. O gargalo do projeto é **dias independentes**, e
cripto é onde eles acumulam mais rápido.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

T_MINIMO: Final[float] = 2.0
"""A mesma régua do Ledger (`radar.evidence.ledger.T_MINIMO`), repetida aqui
como constante local para que este módulo permaneça puro — mas ela é a MESMA
régua por decisão, não por coincidência. Divergir seria ter duas verdades."""

MIN_DIAS_INDEPENDENTES: Final[int] = 15
MIN_TRADES: Final[int] = 50
PERSISTENCIA_MINIMA: Final[float] = 0.70


class Estado(str, Enum):
    """
    Os quatro estados do mapa de validade (taxonomia do DEV, 04/08/2026).

    A ordem de declaração é a de severidade decrescente da conclusão, não a de
    avaliação (ver `classificar`).
    """

    VALIDADO = "validado"
    NAO_COMPROVADO = "nao_comprovado"
    NAO_ESTIMAVEL = "nao_estimavel"
    REFUTADO = "refutado"

    @property
    def acao(self) -> str:
        return _ACAO[self]

    @property
    def recebe_capital(self) -> bool:
        """Só `VALIDADO`. É a única porta para o Strategy Pool."""
        return self is Estado.VALIDADO

    @property
    def opera_em_paper(self) -> bool:
        """
        `NAO_COMPROVADO` opera em paper — é assim que ele vira VALIDADO ou
        REFUTADO em vez de ficar preso para sempre no meio. Paper trading aqui
        não é consolo: é o mecanismo de coleta que dá poder à próxima medição.
        """
        return self is Estado.NAO_COMPROVADO

    @property
    def e_evidencia_contra(self) -> bool:
        return self is Estado.REFUTADO


class Dimensao(str, Enum):
    """
    O estado de UM critério. **Três valores, não dois** — é o princípio
    `não estimável ≠ refutado` aplicado um nível acima: dizer `NAO` sobre um
    critério que ninguém mediu afirma um teste que não houve.
    """

    SIM = "sim"
    NAO = "nao"
    NAO_MEDIDA = "nao_medida"


class Direcao(str, Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRO = "neutro"
    DESCONHECIDA = "desconhecida"


@dataclass(frozen=True)
class PerfilDoProduto:
    """
    O que o produto CONSEGUE executar hoje. É declaração de engenharia, não de
    evidência — e é por isso que mora num eixo separado.

    Motivo de existir (achado de 2026-08-04): as duas hipóteses com maior `t`
    OOS do projeto são **short**, e o produto é **long-only**. Sem este eixo,
    elas cairiam em "não validado" — o que leria como *"a evidência é fraca"*
    quando a verdade é *"a evidência existe e o produto não a alcança"*. São
    conclusões diferentes com ações diferentes: a primeira pede mais dado, a
    segunda pede decisão de arquitetura.
    """

    direcoes_suportadas: frozenset[Direcao] = frozenset({Direcao.LONG})
    mercados_suportados: frozenset[str] = frozenset({"b3", "cripto"})
    horizonte_maximo: int | None = None

    def avaliar(self, direcao: Direcao, mercado: str, horizonte: int) -> tuple[str, ...]:
        """Motivos de INELEGIBILIDADE. Vazio = elegível."""
        motivos: list[str] = []
        if direcao is Direcao.DESCONHECIDA:
            motivos.append("direção da operação não declarada pela fonte")
        elif direcao not in self.direcoes_suportadas:
            motivos.append(
                f"o produto não executa {direcao.value} "
                f"(suporta: {', '.join(sorted(d.value for d in self.direcoes_suportadas))})"
            )
        if mercado not in self.mercados_suportados and mercado != "?":
            motivos.append(f"mercado '{mercado}' fora do produto")
        if self.horizonte_maximo is not None and horizonte > self.horizonte_maximo:
            motivos.append(
                f"horizonte {horizonte}d acima do máximo suportado "
                f"({self.horizonte_maximo}d)"
            )
        return tuple(motivos)


PRODUTO_ATUAL: Final[PerfilDoProduto] = PerfilDoProduto()
"""Long-only nos dois mercados, sem teto de horizonte (ADR 0001 + estado do
produto em 2026-08-04). Mudar isto é decisão de arquitetura do DEV."""


def equilibrio_slippage_bps_lado(excesso_liquido: float) -> float:
    """
    Slippage ADICIONAL por lado que zera o resultado, em bps.

    Adicionar `s` por lado encarece o round-trip em `2s`, então o líquido zera
    em `s = excesso_liquido / 2`. É aritmética sobre número que os ledgers já
    publicam — foi por isso que esta lacuna, das três, era a barata de fechar.

    Note que ela responde uma pergunta DIFERENTE de `t`: `t` pergunta se o
    efeito se distingue de zero, isto pergunta se ele sobrevive à corretora.
    Uma hipótese pode passar numa e reprovar na outra, e as duas importam.
    """
    return excesso_liquido / 2.0 * 10_000.0


class Bloqueio(str, Enum):
    """
    A TERCEIRA dimensão do mapa (pedido do DEV, 04/08/2026): estado diz *onde a
    hipótese está*, `criterios_que_faltam` diz *quão perto*, e isto diz **o que
    exatamente a impede**.

    Sem ela, "faltam 3 critérios" não distingue uma hipótese que precisa de mais
    dado de uma que precisa de outra infraestrutura — e essas duas coisas vão
    para filas diferentes, com donos diferentes e prazos diferentes.
    """

    EVIDENCIA = "evidencia"
    ECONOMIA = "economia"
    EXECUCAO = "execucao"
    RISCO = "risco"
    NENHUM = "nenhum"


CATEGORIA_DO_CRITERIO: Final[dict[str, Bloqueio]] = {
    "evidencia_oos": Bloqueio.EVIDENCIA,
    "persistencia": Bloqueio.EVIDENCIA,
    "estabilidade_oos": Bloqueio.EVIDENCIA,
    "retorno_liquido": Bloqueio.ECONOMIA,
    "margem_de_custo": Bloqueio.ECONOMIA,
    "capacidade": Bloqueio.EXECUCAO,
    "risco_drawdown": Bloqueio.RISCO,
}

PRIORIDADE_DE_BLOQUEIO: Final[tuple[Bloqueio, ...]] = (
    Bloqueio.EVIDENCIA, Bloqueio.ECONOMIA, Bloqueio.EXECUCAO, Bloqueio.RISCO,
)
"""
A ordem em que se ataca, e ela não é arbitrária: não adianta otimizar execução
de algo que não tem edge, nem discutir capacidade de algo que não paga o custo.

`RISCO` entra por ÚLTIMO, e só passou a entrar quando o drawdown virou medível
(`radar.cerebro.risco` + trades OOS persistidos). Enquanto ele era `NÃO MEDIDA`
em 100% das hipóteses, ficava fora da disputa de propósito — teria sido o
dominante de todas as 50, verdade inútil que afogaria o que distingue uma da
outra. Agora que há número, ele disputa; e vai por último porque não adianta
avaliar risco de algo sem edge, sem margem e sem execução.
"""

RISCO_NAO_MEDIDO: Final[str] = (
    "drawdown NÃO MEDIDO — a fonte não publica a curva de equity desta hipótese "
    "(re-rodar o ledger com trades persistidos resolve)"
)

QUADRANTES: Final[tuple[str, ...]] = (
    "VALIDADA", "ALPHA_NAO_EXECUTAVEL", "CANDIDATA", "CANDIDATA_NAO_EXECUTAVEL",
    "NAO_ESTIMAVEL", "REFUTADA",
)

_PREFIXO: Final[dict[str, str]] = {
    "evidencia_oos": "OOS:", "retorno_liquido": "custos: líquido",
    "margem_de_custo": "custos: ", "persistencia": "estabilidade: persistência",
    "estabilidade_oos": "estabilidade: OOS", "capacidade": "escala:",
    "risco_drawdown": "risco:",
}

_ACAO_QUADRANTE: Final[dict[str, str]] = {
    "VALIDADA": "escada do ADR 0032: shadow → paper → champion/challenger → capital",
    "ALPHA_NAO_EXECUTAVEL": (
        "PRESERVAR no mapa como conhecimento do sistema — o alpha existe e o "
        "produto não o alcança. Sair daqui é DECISÃO DE ARQUITETURA (habilitar "
        "short/mercado/horizonte), nunca mais pesquisa"
    ),
    "CANDIDATA": "medir melhor — falta poder ou estabilidade, não falta ideia",
    "CANDIDATA_NAO_EXECUTAVEL": (
        "medir melhor E resolver a elegibilidade; as duas frentes são "
        "independentes e podem correr em paralelo"
    ),
    "NAO_ESTIMAVEL": (
        "buscar dado — mais histórico, mais trades, outro timeframe, outro "
        "mercado. NÃO mexer na hipótese: ela não foi testada"
    ),
    "REFUTADA": (
        "não gastar mais tempo com a MESMA hipótese nas MESMAS condições — "
        "reabrir exige informação nova declarada"
    ),
}

_ACAO: Final[dict[Estado, str]] = {
    Estado.VALIDADO: "entra no Strategy Pool e pode receber capital",
    Estado.NAO_COMPROVADO: "paper trading + coleta de dados → nova validação",
    Estado.NAO_ESTIMAVEL: "observação — buscar mais dados; NÃO é evidência contra",
    Estado.REFUTADO: "fora do pool — não reabrir sem informação NOVA",
}


@dataclass(frozen=True)
class EvidenciaOOS:
    """
    O que uma hipótese precisa ter medido FORA DA AMOSTRA para ser classificada.

    Todo campo vem de walk-forward. Nada aqui é in-sample — se fosse, o mapa
    herdaria a inflação que fez 252 células verdes virarem zero sobreviventes.
    """

    hipotese: str
    mercado: str
    horizonte: int

    trades: int
    dias_independentes: int
    excesso_liquido: float
    """Já líquido de custo. Bruto não classifica nada: o corretor não é opcional."""
    t_nw: float

    direcao: Direcao = Direcao.DESCONHECIDA
    """LONG, SHORT ou NEUTRO. `DESCONHECIDA` (default) torna a hipótese
    inelegível **com motivo explícito** — nunca elegível por omissão. Uma
    hipótese cuja direção ninguém declarou não pode ser tratada como long por
    conveniência."""

    anos_positivos: int = 0
    anos_testados: int = 0

    persistencia: float | None = None
    """
    Fração de anos com resultado positivo, quando a fonte já a traz pronta (o
    Ledger do Cérebro traz; o Ledger Técnico dá `anos_positivos`/`anos_testados`
    e ela é derivada). **`None` significa NÃO MEDIDA e nunca passa no critério.**

    Esta ressalva não é preciosismo — é o buraco que este campo existe para
    tapar. Antes, uma fonte sem informação de anos fazia o critério de
    estabilidade ser **pulado em silêncio**, e a hipótese saía parecendo melhor
    do que foi medida. "Não medido" tratado como "passou" é exatamente como um
    backtest passa a mentir a favor.
    """

    efeito_minimo_relevante: float | None = None
    """O menor retorno por trade que valeria a pena operar, **declarado antes de
    medir**. Sem ele não há teste de poder, e sem teste de poder não se pode
    distinguir REFUTADO de NÃO COMPROVADO — então a ausência degrada para NÃO
    COMPROVADO, nunca para REFUTADO. Falha para o lado de não enterrar."""

    equilibrio_slippage_bps: float | None = None
    """Slippage/lado que zera o resultado. Margem fina = frágil a execução."""
    margem_custo_minima_bps: float = 5.0

    risco_dentro_da_tolerancia: bool | None = None
    """Drawdown dentro da tolerância (`radar.cerebro.risco`). Três estados:
    `True` ok, `False` estourou, **`None` NÃO MEDIDA** — e não medida bloqueia.

    ⚠️ Esta dimensão só REBAIXA, nunca promove: é critério que hoje não é
    avaliado, e ao passar a ser ele ADICIONA uma porta. Portão de segurança
    antes de capital, não caminho de validação."""

    max_drawdown: float | None = None
    """O número ao lado do booleano — mesmo princípio do `capacidade_estimada`:
    nenhuma dimensão publica só o veredito quando o valor existe."""

    estabilidade_detalhe: dict[str, Any] | None = None
    """`concentracao`, `soma_primeira_metade`, `soma_segunda_metade`. Publicados
    porque "instável" não diz se falhou por um ano dominante (evento) ou por
    regime que acabou (metades discordantes) — e as duas mandam para
    investigações diferentes."""

    capacidade_estimada: float | None = None
    """
    Capital total que a hipótese comporta, na moeda do mercado, estimado por
    `radar.cerebro.capacidade` sob modelo declarado. `None` = **não medida**,
    e não medida bloqueia VALIDADO: capacidade desconhecida é chutar o tamanho
    da posição.

    Deixou de ser `bool` de propósito. `capacidade_medida=True` respondia
    "alguém olhou?" quando a pergunta que decide capital é "cabe quanto?" — e
    um booleano não distingue uma estratégia que comporta R$ 3.000 de uma que
    comporta R$ 3.000.000.
    """

    capital_alvo: float | None = None
    """Quanto se pretende alocar. Precisa ser DECLARADO: sem alvo, "capacidade
    suficiente" não é uma afirmação verificável. `None` faz o critério passar
    com qualquer capacidade estimada > 0, e isso fica visível no relatório."""

    estavel_oos: bool | None = None
    """
    Concentração + réplica por metade (`radar.cerebro.estabilidade`). Três
    estados de propósito: `True` estável, `False` instável, **`None` NÃO
    MEDIDA**. Os dois últimos bloqueiam VALIDADO, mas pedem coisas diferentes —
    `False` pede outra hipótese (ou análise condicional), `None` pede medição.
    """


@dataclass(frozen=True)
class Veredito:
    estado: Estado
    hipotese: str
    justificativa: str
    criterios_faltantes: tuple[str, ...] = ()
    poder_suficiente: bool | None = None
    efeito_minimo_detectavel: float | None = None

    dimensoes: dict[str, Dimensao] = field(default_factory=dict)
    """Critério a critério, com SIM/NAO/NAO_MEDIDA. O `estado` é o resumo; isto
    é o detalhe que diz O QUE fazer — `NAO_MEDIDA` pede medição, `NAO` pede
    outra hipótese."""

    elegivel_no_produto: bool = True
    motivos_inelegibilidade: tuple[str, ...] = ()
    """EIXO SEPARADO da evidência. Uma hipótese pode ter evidência impecável e
    ser inelegível (short num produto long-only) — e isso **não é refutação**."""

    @property
    def entra_no_pool(self) -> bool:
        """A conjunção dos dois eixos: só recebe capital quem é VALIDADO **e**
        operacionalmente elegível."""
        return self.estado.recebe_capital and self.elegivel_no_produto

    @property
    def quadrante(self) -> str:
        """
        A MATRIZ evidência × elegibilidade (desenho do DEV, 04/08/2026).

        |                     | elegível | inelegível |
        |---------------------|----------|------------|
        | **validado**        | 🟢 `VALIDADA` | 🟡 `ALPHA_NAO_EXECUTAVEL` |
        | **não comprovado**  | 🔵 `CANDIDATA` | 🟠 `CANDIDATA_NAO_EXECUTAVEL` |
        | **não estimável**   | ⚪ `NAO_ESTIMAVEL` | ⚪ `NAO_ESTIMAVEL` |
        | **refutado**        | 🔴 `REFUTADA` | 🔴 `REFUTADA` |

        **Uma correção à matriz original, e ela é do mesmo tipo que o DEV
        defendeu**: o desenho tinha 3 linhas, com "evidência insuficiente"
        numa só. Mas *"medi e ficou abaixo da régua"* e *"não consegui medir"*
        são exatamente as duas coisas que este sistema inteiro existe para
        separar — colapsá-las aqui reintroduziria, no topo, o erro que o quarto
        veredito corrigiu na base. Por isso são **4 linhas**.

        As duas últimas linhas **não** se dividem por coluna, e isso é
        deliberado: sem estatística (⚪) ou com refutação (🔴), a elegibilidade
        do produto não muda nada que se possa fazer com a hipótese. Dividi-las
        criaria quadrantes sem ação distinta — e quadrante sem ação é ruído de
        relatório.
        """
        if self.estado is Estado.NAO_ESTIMAVEL:
            return "NAO_ESTIMAVEL"
        if self.estado is Estado.REFUTADO:
            return "REFUTADA"
        if self.estado is Estado.VALIDADO:
            return "VALIDADA" if self.elegivel_no_produto else "ALPHA_NAO_EXECUTAVEL"
        return "CANDIDATA" if self.elegivel_no_produto else "CANDIDATA_NAO_EXECUTAVEL"

    @property
    def acao_do_quadrante(self) -> str:
        return _ACAO_QUADRANTE[self.quadrante]

    @property
    def bloqueios(self) -> dict[str, tuple[str, ...]]:
        """
        O que impede esta hipótese, agrupado por natureza do impedimento.

        A inelegibilidade operacional (short, mercado, horizonte) entra em
        `EXECUCAO` porque é isso que ela é — restrição de infraestrutura, não de
        evidência. `RISCO` aparece sempre, com a lacuna declarada.
        """
        por_categoria: dict[str, list[str]] = {b.value: [] for b in Bloqueio
                                               if b is not Bloqueio.NENHUM}
        for criterio, dim in self.dimensoes.items():
            if dim is Dimensao.SIM:
                continue
            categoria = CATEGORIA_DO_CRITERIO.get(criterio, Bloqueio.EVIDENCIA)
            motivo = next(
                (f for f in self.criterios_faltantes if f.startswith(_PREFIXO.get(criterio, ""))),
                criterio,
            )
            por_categoria[categoria.value].append(motivo)
        if self.estado is Estado.NAO_ESTIMAVEL:
            # Sem dimensões computadas o laço acima não produz nada, e o
            # bloqueio sairia `NENHUM` — leitura errada e perigosa: a hipótese
            # não está desimpedida, ela não foi medida. Falta de amostra é
            # bloqueio de EVIDÊNCIA.
            por_categoria[Bloqueio.EVIDENCIA.value].append(self.justificativa)
        por_categoria[Bloqueio.EXECUCAO.value].extend(self.motivos_inelegibilidade)
        # `RISCO` NÃO é mais acrescentado incondicionalmente. Enquanto drawdown
        # não era medido em lugar nenhum, a linha fixa era o jeito de manter a
        # lacuna visível; agora `risco_drawdown` é dimensão de verdade e o laço
        # acima já a trata. Manter o append faria a categoria nunca esvaziar —
        # e, com `RISCO` na lista de prioridade, ela dominaria TODA hipótese,
        # inclusive as que passam. Foi o que o teste pegou.
        return {k: tuple(v) for k, v in por_categoria.items() if v}

    @property
    def bloqueio_dominante(self) -> Bloqueio:
        """
        Por onde começar. `RISCO` fica fora da disputa — ver
        `PRIORIDADE_DE_BLOQUEIO`.
        """
        b = self.bloqueios
        for categoria in PRIORIDADE_DE_BLOQUEIO:
            if b.get(categoria.value):
                return categoria
        return Bloqueio.NENHUM

    @property
    def criterios_que_faltam(self) -> int:
        """Quantos critérios separam esta hipótese de `VALIDADA`. É o número que
        transforma o quadrante em fila de trabalho: 1 é um passo, 5 é outra
        pesquisa."""
        return len(self.criterios_faltantes)

    @property
    def rotulo_operacional(self) -> str:
        """Mantido como alias do quadrante para quem já lia este campo."""
        return self.quadrante

    def to_dict(self) -> dict[str, Any]:
        return {
            "hipotese": self.hipotese,
            "estado": self.estado.value,
            "quadrante": self.quadrante,
            "acao_do_quadrante": self.acao_do_quadrante,
            "criterios_que_faltam": self.criterios_que_faltam,
            "bloqueio_dominante": self.bloqueio_dominante.value,
            "bloqueios": {k: list(v) for k, v in self.bloqueios.items()},
            "acao": self.estado.acao,
            "recebe_capital": self.estado.recebe_capital,
            "elegivel_no_produto": self.elegivel_no_produto,
            "motivos_inelegibilidade": list(self.motivos_inelegibilidade),
            "entra_no_pool": self.entra_no_pool,
            "justificativa": self.justificativa,
            "dimensoes": {k: v.value for k, v in self.dimensoes.items()},
            "criterios_faltantes": list(self.criterios_faltantes),
            "poder_suficiente": self.poder_suficiente,
            "efeito_minimo_detectavel": self.efeito_minimo_detectavel,
        }


def persistencia_de(e: EvidenciaOOS) -> float | None:
    """
    A persistência efetiva: explícita se a fonte a trouxe, derivada de
    `anos_positivos/anos_testados` se não. `None` quando **nenhum dos dois**
    caminhos existe — e aí o critério FALHA, nunca é pulado.
    """
    if e.persistencia is not None:
        return e.persistencia
    if e.anos_testados:
        return e.anos_positivos / e.anos_testados
    return None


def erro_padrao(excesso: float, t: float) -> float | None:
    """`ep = excesso / t`. `None` quando `t` é zero ou indefinido."""
    if t is None or math.isnan(t) or t == 0.0 or math.isnan(excesso):
        return None
    return abs(excesso / t)


def avaliar_dimensoes(e: EvidenciaOOS) -> tuple[dict[str, Dimensao], list[str]]:
    """
    Os seis critérios, cada um com **SIM / NÃO / NÃO MEDIDA**, mais a lista
    legível do que faltou.

    A distinção `NAO` × `NAO_MEDIDA` é o ponto do módulo inteiro repetido um
    nível abaixo, e ela muda a AÇÃO: `NAO` diz *esta hipótese não serve*;
    `NAO_MEDIDA` diz *vá medir*. Ambos bloqueiam `VALIDADO` — mas mandar
    coletar dado e mandar descartar a hipótese não são a mesma instrução.
    """
    dim: dict[str, Dimensao] = {}
    faltantes: list[str] = []

    # 1 · evidência OOS
    if e.t_nw < T_MINIMO:
        dim["evidencia_oos"] = Dimensao.NAO
        faltantes.append(f"OOS: t={e.t_nw:+.2f} < {T_MINIMO}")
    else:
        dim["evidencia_oos"] = Dimensao.SIM

    # 2 · o líquido é positivo
    if e.excesso_liquido <= 0:
        dim["retorno_liquido"] = Dimensao.NAO
        faltantes.append(f"custos: líquido {e.excesso_liquido:+.4%} ≤ 0")
    else:
        dim["retorno_liquido"] = Dimensao.SIM

    # 3 · margem sobre o custo de execução
    if e.equilibrio_slippage_bps is None:
        dim["margem_de_custo"] = Dimensao.NAO_MEDIDA
        faltantes.append("custos: slippage de equilíbrio NÃO MEDIDO")
    elif e.equilibrio_slippage_bps < e.margem_custo_minima_bps:
        dim["margem_de_custo"] = Dimensao.NAO
        faltantes.append(
            f"custos: equilíbrio em {e.equilibrio_slippage_bps:.1f} bps/lado < "
            f"margem mínima {e.margem_custo_minima_bps:.1f}"
        )
    else:
        dim["margem_de_custo"] = Dimensao.SIM

    # 4 · persistência anual
    persist = persistencia_de(e)
    if persist is None:
        dim["persistencia"] = Dimensao.NAO_MEDIDA
        faltantes.append("estabilidade: persistência anual NÃO MEDIDA")
    elif persist < PERSISTENCIA_MINIMA:
        dim["persistencia"] = Dimensao.NAO
        faltantes.append(
            f"estabilidade: persistência {persist:.0%} < {PERSISTENCIA_MINIMA:.0%}"
            + (f" ({e.anos_positivos}/{e.anos_testados} anos)" if e.anos_testados else "")
        )
    else:
        dim["persistencia"] = Dimensao.SIM

    # 5 · estabilidade OOS (concentração + réplica por metade)
    if e.estavel_oos is None:
        dim["estabilidade_oos"] = Dimensao.NAO_MEDIDA
        faltantes.append("estabilidade: OOS NÃO MEDIDA (concentração/metades)")
    elif not e.estavel_oos:
        dim["estabilidade_oos"] = Dimensao.NAO
        faltantes.append("estabilidade: OOS instável (ver motivos)")
    else:
        dim["estabilidade_oos"] = Dimensao.SIM

    # 6-bis · risco (drawdown). Entra DEPOIS da capacidade e antes do retorno
    # porque a ordem do dict e a de leitura do relatorio, nao a de avaliacao.
    if e.risco_dentro_da_tolerancia is None:
        dim["risco_drawdown"] = Dimensao.NAO_MEDIDA
        faltantes.append("risco: drawdown NÃO MEDIDO")
    elif not e.risco_dentro_da_tolerancia:
        dim["risco_drawdown"] = Dimensao.NAO
        faltantes.append(
            "risco: drawdown "
            + (f"{e.max_drawdown:.1%}" if e.max_drawdown is not None else "")
            + " fora da tolerância"
        )
    else:
        dim["risco_drawdown"] = Dimensao.SIM

    # 6 · capacidade
    if e.capacidade_estimada is None:
        dim["capacidade"] = Dimensao.NAO_MEDIDA
        faltantes.append("capacidade: NÃO MEDIDA")
    elif e.capital_alvo is not None and e.capacidade_estimada < e.capital_alvo:
        dim["capacidade"] = Dimensao.NAO
        # NAO ATENDE AO GATE, nunca "capacidade insuficiente". A diferenca nao e
        # de estilo: "82.996 < 100.000" le como DEFEITO da estrategia, quando o
        # fato e que ela nao atende ao requisito de ESCALA declarado para esta
        # rodada. Nao reprovou por falta de alpha nem por execucao — reprovou
        # por escala, e quem le o mapa daqui a um ano precisa ver isso.
        faltantes.append(
            f"escala: NÃO ATENDE AO GATE — capacidade estimada "
            f"{e.capacidade_estimada:,.0f} contra gate oficial "
            f"{e.capital_alvo:,.0f} (a estratégia funciona; não comporta ESTE capital)"
        )
    else:
        dim["capacidade"] = Dimensao.SIM

    return dim, faltantes


def classificar(
    e: EvidenciaOOS, *, perfil_produto: PerfilDoProduto = PRODUTO_ATUAL
) -> Veredito:
    """
    A ordem de avaliação é fixa: **estimável? → validado? → refutado? → resto**.

    `NAO_ESTIMAVEL` vem primeiro porque sem estatística nada mais é
    interpretável. `VALIDADO` vem antes de `REFUTADO` porque exige que TODOS os
    seis critérios passem — e listar quais faltaram é mais informativo do que
    um "não" — e porque um resultado que passa em tudo não pode, por
    construção, ser candidato a refutação.
    """
    # O eixo de elegibilidade e o de evidência são ORTOGONAIS, então a
    # elegibilidade é computada ANTES de qualquer ramo — inclusive nos
    # não-estimáveis. Saber que uma hipótese é short num produto long-only não
    # depende de ela ter estatística.
    inelegibilidade = perfil_produto.avaliar(e.direcao, e.mercado, e.horizonte)
    elegivel = not inelegibilidade

    # 1 ── estimável?
    if e.trades < MIN_TRADES:
        return Veredito(Estado.NAO_ESTIMAVEL, e.hipotese,
                        f"{e.trades} trades OOS < {MIN_TRADES} — nenhum teste aconteceu",
                        elegivel_no_produto=elegivel,
                        motivos_inelegibilidade=inelegibilidade)
    if e.dias_independentes < MIN_DIAS_INDEPENDENTES:
        return Veredito(Estado.NAO_ESTIMAVEL, e.hipotese,
                        f"{e.dias_independentes} dias independentes < "
                        f"{MIN_DIAS_INDEPENDENTES} — o `t` de uma amostra com esse "
                        "clustering é ficção",
                        elegivel_no_produto=elegivel,
                        motivos_inelegibilidade=inelegibilidade)
    if math.isnan(e.t_nw):
        return Veredito(Estado.NAO_ESTIMAVEL, e.hipotese,
                        "`t` indefinido — a estatística não existe", elegivel_no_produto=elegivel,
                        motivos_inelegibilidade=inelegibilidade)

    # 2 ── validado? Os seis critérios, cada um com TRÊS estados possíveis.
    dimensoes, faltantes = avaliar_dimensoes(e)

    if not faltantes:
        return Veredito(
            Estado.VALIDADO, e.hipotese,
            f"OOS t={e.t_nw:+.2f} com {e.dias_independentes} dias independentes, "
            f"líquido {e.excesso_liquido:+.4%}, custo/estabilidade/capacidade "
            "satisfeitos"
            + ("" if not inelegibilidade
               else " — MAS não é operacionalmente elegível: "
                    + "; ".join(inelegibilidade)),
            dimensoes=dimensoes,
            elegivel_no_produto=elegivel,
            motivos_inelegibilidade=inelegibilidade,
        )

    # 3 ── refutado? Só com PODER. Sem poder, negativo não decide.
    ep = erro_padrao(e.excesso_liquido, e.t_nw)
    emd = None if ep is None else T_MINIMO * ep
    tinha_poder = (
        emd is not None
        and e.efeito_minimo_relevante is not None
        and emd <= e.efeito_minimo_relevante
    )
    if tinha_poder and e.t_nw < T_MINIMO:
        return Veredito(
            Estado.REFUTADO, e.hipotese,
            f"a amostra detectaria um efeito de {emd:+.4%} (≤ o mínimo relevante de "
            f"{e.efeito_minimo_relevante:+.4%}) e mediu t={e.t_nw:+.2f} sobre "
            f"{e.dias_independentes} dias — o teste tinha poder e não achou",
            criterios_faltantes=tuple(faltantes),
            poder_suficiente=True,
            efeito_minimo_detectavel=emd,
            dimensoes=dimensoes,
            elegivel_no_produto=elegivel,
            motivos_inelegibilidade=inelegibilidade,
        )

    # 4 ── o resto: há sinal, falta prova. Paper trading + coleta.
    motivo_sem_poder = (
        "efeito mínimo relevante não declarado — sem ele não há teste de poder, "
        "e sem poder um negativo não refuta"
        if e.efeito_minimo_relevante is None
        else (
            f"a amostra só detectaria {emd:+.4%}, acima do mínimo relevante de "
            f"{e.efeito_minimo_relevante:+.4%} — o teste não teve poder"
            if emd is not None
            else "erro-padrão não estimável"
        )
    )
    return Veredito(
        Estado.NAO_COMPROVADO, e.hipotese,
        f"{'; '.join(faltantes)} — e não é refutação: {motivo_sem_poder}",
        criterios_faltantes=tuple(faltantes),
        poder_suficiente=False,
        efeito_minimo_detectavel=emd,
        dimensoes=dimensoes,
            elegivel_no_produto=elegivel,
            motivos_inelegibilidade=inelegibilidade,
    )


@dataclass(frozen=True)
class MapaValidade:
    """O pool, particionado. É isto que o Strategy Allocator recebe."""

    vereditos: tuple[Veredito, ...] = field(default_factory=tuple)

    @property
    def pool(self) -> tuple[Veredito, ...]:
        """
        Os DOIS eixos: validado E operacionalmente elegível. Uma hipótese
        validada mas inelegível fica em `evidencia_sem_elegibilidade` — visível,
        contável e explicitamente NÃO refutada.
        """
        return tuple(v for v in self.vereditos if v.entra_no_pool)

    @property
    def evidencia_sem_elegibilidade(self) -> tuple[Veredito, ...]:
        """Evidência estatística encontrada, produto não alcança. É uma decisão
        de arquitetura pendente, não um resultado de pesquisa."""
        return tuple(
            v for v in self.vereditos
            if v.estado.recebe_capital and not v.elegivel_no_produto
        )

    @property
    def em_paper(self) -> tuple[Veredito, ...]:
        return tuple(v for v in self.vereditos if v.estado is Estado.NAO_COMPROVADO)

    @property
    def em_observacao(self) -> tuple[Veredito, ...]:
        return tuple(v for v in self.vereditos if v.estado is Estado.NAO_ESTIMAVEL)

    @property
    def fora(self) -> tuple[Veredito, ...]:
        return tuple(v for v in self.vereditos if v.estado is Estado.REFUTADO)

    def por_quadrante(self) -> dict[str, tuple[Veredito, ...]]:
        """O mapa particionado pela matriz. É esta a visão que o Cérebro
        consome: cada quadrante tem UMA ação, e ações diferentes não se
        misturam na mesma fila."""
        return {
            q: tuple(v for v in self.vereditos if v.quadrante == q)
            for q in QUADRANTES
        }

    def resumo(self) -> dict[str, int]:
        return {
            "validado": len(self.pool),
            "evidencia_sem_elegibilidade": len(self.evidencia_sem_elegibilidade),
            "nao_comprovado": len(self.em_paper),
            "nao_estimavel": len(self.em_observacao),
            "refutado": len(self.fora),
            "total": len(self.vereditos),
            **{f"quadrante_{q.lower()}": len(vs)
               for q, vs in self.por_quadrante().items()},
        }


def montar_mapa(
    evidencias: list[EvidenciaOOS], *, perfil_produto: PerfilDoProduto = PRODUTO_ATUAL
) -> MapaValidade:
    return MapaValidade(
        tuple(classificar(e, perfil_produto=perfil_produto) for e in evidencias)
    )

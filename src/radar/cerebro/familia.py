"""
src/radar/cerebro/familia.py — o contrato que torna família um DADO.

Ordem do DEV (2026-08-06): *"o próprio Cérebro separava as regras das famílias
dentro dele. Isso faz com que o Cérebro não precise toda hora ser alterado para
seguir uma família. Construímos as famílias, e o Cérebro só decide em qual vai
encaixar o ativo."* É o **Router** do ADR 0032 §2, que já avisava: *"o erro a
evitar é o silo sem Router."*

O que o código fazia antes deste módulo
----------------------------------------
`scripts/cerebro/rodar_scalp_cripto_v2.py` carregava `rodar_swing_v1.py` por
`importlib` e fazia **monkeypatch de `swing._CFG["cripto"]`** — mutava um global
privado de outro módulo para caber a família nova. Funcionava, e era
literalmente alterar o Cérebro para seguir a família. Com Day e Position seriam
mais dois remendos empilhados sobre o mesmo global, cada perfil novo
**aumentando** o acoplamento.

Uma `Familia` é uma **declaração**, não um script
-------------------------------------------------
Tudo que distingue um especialista do outro vira campo: espaço de seleção,
alvo, custo (venue e tipo de ordem), horizontes e sua unidade, embargo,
holdout, amostragem, store de origem, primeiro ano de teste. O runner é um só e
consome a declaração.

**O que NÃO entra aqui, de propósito**: as 7 facas, os 5 portões do Contrato de
Pensamento, o FDR e a régua econômica. Eles são do **Cérebro**, valem para toda
família, e uma família que pudesse escolher a própria faca não estaria sendo
medida — estaria se autoavaliando.

A unidade do horizonte é campo, e isso não é detalhe
-----------------------------------------------------
Embargo e holdout são declarados na **unidade do fenômeno** (pregões, dias
corridos, minutos) e convertidos para a unidade **amostrada** no ponto de uso.
Sem essa separação, um holdout de 43.200 "barras" sobre um dataset amostrado de
17.537 marcos é maior que o dataset inteiro — e o gerador de splits devolve zero
folds **em silêncio**, que é como este projeto perdeu duas rodadas.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

__all__ = [
    "PERFIS",
    "REGISTRO",
    "UNIDADES_HORIZONTE",
    "Familia",
    "familia",
    "hipoteses_duplicadas",
    "horizonte_em_dias",
    "registrar",
]

PERFIS: Final[tuple[str, ...]] = ("scalp", "day", "swing", "position")
"""Os quatro perfis da MISSÃO CORE. **São o produto — nada os invalida**
(ordem do DEV): resultado negativo elimina *aquela estratégia naquele perfil*,
jamais o perfil. Uma família declara a qual perfil serve."""

UNIDADES_HORIZONTE: Final[tuple[str, ...]] = ("pregoes", "dias_corridos", "barras")
"""As mesmas de `radar.mie.dataset._UNIDADES_EMBARGO_VALIDAS` — reexportadas
aqui para que a família declare a unidade sem importar o pipeline."""


@dataclass(frozen=True)
class Familia:
    """
    Um especialista, declarado inteiro.

    Congelado (`frozen=True`) porque a configuração de uma família é
    pré-registrada: mudar um valor **cria família nova** (ADR 0032 §2), não
    ajusta a existente. Um `dataclass` mutável convidaria exatamente o
    monkeypatch que este módulo existe para eliminar.
    """

    nome: str
    """Identificador da família, e é ele que aparece no Evidence Map como
    prefixo da hipótese (`<familia>:<celula>#<ponta>`)."""

    perfil: str
    """Um de `PERFIS`. É o que o Router usa para rotear o ativo, e o que o
    perfil do usuário casa (§8.3 trava 3)."""

    mercado: str
    """`b3` ou `cripto`. Separado do perfil porque os mecanismos
    cross-sectionais têm **sinal oposto** entre os dois (ADR 0032 §2) — a mesma
    estratégia em mercados diferentes é família diferente."""

    store: Path
    """Parquet ou diretório de parquets. Diretório é lido como dataset único."""

    chave_ativo: str = "ticker"
    """Nome da coluna de ativo no store (`symbol` em cripto). A tradução para
    `ticker` acontece na borda; o núcleo nunca ouviu falar de `symbol`."""

    colunas_selecao: tuple[str, ...] = ()
    """O espaço de SELEÇÃO — o que o modelo pode usar para escolher entre
    ativos. Passa pelo G-P1: nada constante no instante entra aqui."""

    colunas_treino: tuple[str, ...] = ()
    """O espaço de TREINO. Igual ao de seleção na maioria das famílias; difere
    no braço auto-referente, onde contexto-do-dia é sinal legítimo porque o
    alvo não é relativo."""

    horizontes: tuple[int, ...] = ()
    """Na unidade de `unidade_horizonte`."""

    unidade_horizonte: str = "pregoes"
    """Um de `UNIDADES_HORIZONTE`. `barras` é o caso intradiário."""

    segundos_por_barra: int | None = None
    """Duração de uma barra, em segundos. **Obrigatório** quando
    `unidade_horizonte == "barras"`; proibido nos demais casos.

    Existe porque `"barras"` sozinho é AMBÍGUO, e a ambiguidade só apareceu
    quando a segunda família intradiária nasceu: `day_cripto_v1` conta barras de
    1 hora e `scalp_cripto_v1` conta barras de 1 segundo. Sem este campo,
    `horizontes=(30, 60, 300)` do scalp e `(24, 72)` do day são números sem
    unidade comum — e qualquer código que compare as duas famílias (o teste de
    sobreposição de horizonte é o caso concreto) lê 30 segundos como 30 horas e
    acusa uma invasão que não existe."""

    embargo: int = 0
    """≥ o maior horizonte, na **unidade do fenômeno**. O embargo existe para o
    treino não ver o desfecho do teste."""

    holdout: int | None = None
    """Na unidade do fenômeno. `None` usa o default do pipeline (12 meses),
    correto para dado diário e destrutivo para 1m — com 2 anos de minuto ele
    engole o treino inteiro do primeiro fold."""

    passo_amostragem: int = 1
    """Treina em barras espaçadas deste passo. `1` = todas.

    Não é atalho de desempenho: com horizonte de 240 barras, dois minutos
    adjacentes têm rótulos ~99,6% **sobrepostos**, e o modelo veria o mesmo
    exemplo 240 vezes. É a mesma aritmética que justifica contar janelas não
    sobrepostas ao medir potência. **Vale só no treino** — a medição OOS usa
    todas as barras, senão a frequência de sinal muda e o controle aleatório
    deixa de ser controle."""

    normalizar_data: bool = True
    """`False` para intradiário. Normalizar colapsa as 1.440 barras de um dia
    numa data só, e o gerador de splits — que conta embargo e holdout em barras
    — passa a contar dias."""

    ano_primeiro_teste: int = 0
    """Primeiro ano-calendário de teste no walk-forward."""

    venue: str | None = None
    tipo_ordem: str | None = None
    """Quando declarados, o custo vem de `CryptoCostModel.da_venue`. `None`
    mantém o roteamento por faixa de liquidez. Declarar ANTES de medir é o que
    impede escolher o custo depois de ver o resultado."""

    fracao_gate: float = 0.05
    """A fração do Top-N. A escala de emissão é ranking (ADR 0032 §4)."""

    pre_registro: str = ""
    """Caminho do documento congelado. Família sem pré-registro não mede."""

    notas: dict[str, Any] = field(default_factory=dict)
    """Espaço para o que é específico de uma família e não merece campo — nunca
    para o que deveria ser campo."""

    def __post_init__(self) -> None:
        if self.perfil not in PERFIS:
            raise ValueError(f"perfil desconhecido: {self.perfil!r} — use um de {PERFIS}")
        if self.unidade_horizonte not in UNIDADES_HORIZONTE:
            raise ValueError(
                f"unidade_horizonte desconhecida: {self.unidade_horizonte!r} — "
                f"use uma de {UNIDADES_HORIZONTE}"
            )
        if self.passo_amostragem < 1:
            raise ValueError("passo_amostragem >= 1 (1 = todas as barras)")
        if self.horizontes and self.embargo < max(self.horizontes):
            raise ValueError(
                f"embargo ({self.embargo}) < maior horizonte ({max(self.horizontes)}): "
                "o treino veria o desfecho do teste"
            )
        if (self.venue is None) != (self.tipo_ordem is None):
            raise ValueError("venue e tipo_ordem: declare os dois ou nenhum")
        if self.unidade_horizonte == "barras":
            if not self.segundos_por_barra or self.segundos_por_barra < 1:
                raise ValueError(
                    f"{self.nome}: unidade_horizonte='barras' exige "
                    "segundos_por_barra >= 1 — sem ele, 'barra' é ambíguo entre "
                    "famílias (1 s no scalp, 3.600 s no day)"
                )
        elif self.segundos_por_barra is not None:
            raise ValueError(
                f"{self.nome}: segundos_por_barra só faz sentido com "
                f"unidade_horizonte='barras', não com {self.unidade_horizonte!r}"
            )

    # ── Conversão para a unidade AMOSTRADA ────────────────────────────────
    #
    # Declara-se na unidade do fenômeno; usa-se na unidade do dataset. Sem
    # isto o holdout vira maior que o dataset e o walk-forward devolve zero
    # folds sem reclamar.

    @property
    def embargo_amostrado(self) -> int:
        return max(1, self.embargo // self.passo_amostragem)

    @property
    def holdout_amostrado(self) -> int | None:
        if self.holdout is None:
            return None
        return max(1, self.holdout // self.passo_amostragem)

    def cfg_do_pipeline(self) -> dict[str, Any]:
        """A configuração no formato que o runner consome.

        Existe para que o runner **leia** de um lugar em vez de o script
        **escrever** no global de outro módulo."""
        return {
            "store": self.store,
            "chave": self.chave_ativo,
            "colunas_selecao": self.colunas_selecao,
            "colunas_x": self.colunas_treino or self.colunas_selecao,
            "embargo": self.embargo_amostrado,
            "unidade_embargo": self.unidade_horizonte,
            "ano_primeiro_teste": self.ano_primeiro_teste,
            # Quantas barras cruas cabem em UM passo da grade do painel. É a
            # ponte que converte `horizonte` (em barras) para passos de grade, e
            # sem ela não dá para saber quando a posição fecha: `swing_v1` tem
            # barra=dia e passo=1, então h=10 são 10 passos; `scalp_cripto_v2`
            # tem barra=1min e passo=60, então h=60 é UM passo. Os dois publicam
            # `horizonte` sem nada que os distinga —
            # `CEREBRO_EQUITY_HORIZONTE_SOBREPOSTO01`.
            "passo_amostragem": self.passo_amostragem,
        }


REGISTRO: dict[str, Familia] = {}
"""Famílias conhecidas, por nome. É daqui que o Router lê o que existe."""


def registrar(f: Familia) -> Familia:
    """Publica a família no registro. Nome duplicado **levanta**.

    Duas famílias com o mesmo nome colidiriam no Evidence Map — e duas
    hipóteses homônimas exibindo os números da primeira é o tipo de colapso de
    identidade que o mapa existe para expor, não para cometer."""
    if f.nome in REGISTRO:
        raise ValueError(f"família já registrada: {f.nome!r}")
    REGISTRO[f.nome] = f
    return f


#: Segundos num dia — a unidade comum entre `barras` e `dias_corridos`.
DIA_EM_SEGUNDOS: Final[int] = 86_400


def horizonte_em_dias(f: Familia, horizonte: int) -> float:
    """Um horizonte da família, convertido para DIAS.

    Comparar horizontes entre famílias sem converter soma unidades diferentes,
    e o erro é silencioso — não uma exceção. Usa `segundos_por_barra`
    DECLARADO: uma versão anterior presumia barra horária, e isso passou
    despercebido enquanto `day_cripto_v1` era a única família intradiária.
    Quando o Scalp entrou com barra de 1 s, os `30` segundos dele viraram
    30 horas e a guarda acusou uma invasão inexistente.
    """
    seg = f.segundos_por_barra or DIA_EM_SEGUNDOS
    return horizonte * seg / DIA_EM_SEGUNDOS


def hipoteses_duplicadas(familias: Iterable[Familia]) -> list[tuple[str, str, str]]:
    """As hipóteses declaradas DUAS vezes — `(perfil, mercado, horizonte, espaço)` igual.

    Devolve `(nome_da_nova, nome_da_anterior, descrição da colisão)`; lista
    vazia significa catálogo limpo.

    ⚠️ **Esta definição de "duplicata" foi CORRIGIDA em 13/08, e a correção
    merece desconfiança porque foi feita por quem precisava dela.** Duas
    guardas anteriores diziam coisas diferentes e mais fortes:

    * `(perfil, mercado)` não pode repetir — *"medem o mesmo território"*;
    * faixas de horizonte não podem se tocar.

    As duas proibiam o que o ADR 0032 §2 **exige**: os especialistas são
    separados por `motor × mercado × horizonte`, e `motor` é eixo. Uma regra
    de um-por-`(perfil, mercado)` torna impossível haver dois especialistas de
    swing em cripto — e o Router, por escrito, *"só existe depois de ≥2
    especialistas com zona de validade provada"*. A guarda proibia a
    arquitetura.

    O que continua proibido, e é o que ambas queriam dizer: a **mesma**
    hipótese entrando duas vezes no denominador do FDR sem trazer informação
    nova. Isso é perfil + mercado + horizonte + espaço de seleção iguais.
    Espaços diferentes no mesmo horizonte são hipóteses **concorrentes**, que
    é exatamente o que o Router precisa ter para orquestrar.

    Fica em `familia.py`, e não num arquivo de teste, porque é invariante do
    catálogo — não detalhe de uma suíte. Não levanta em `registrar()` de
    propósito: `catalogo` é importado pela API, e derrubar o processo web por
    uma invariante de pesquisa trocaria um erro de laboratório por uma queda
    de produto.
    """
    vistos: dict[tuple[str, str, float, tuple[str, ...]], str] = {}
    colisoes: list[tuple[str, str, str]] = []
    for f in familias:
        for h in f.horizontes:
            chave = (f.perfil, f.mercado, horizonte_em_dias(f, h), f.colunas_selecao)
            anterior = vistos.get(chave)
            if anterior is None:
                vistos[chave] = f.nome
            else:
                colisoes.append(
                    (f.nome, anterior, f"{f.perfil}/{f.mercado}, {chave[2]:g}d, mesmo espaço")
                )
    return colisoes


def familia(nome: str) -> Familia:
    """A família de `nome`. Desconhecida **levanta** com a lista do que existe."""
    if nome not in REGISTRO:
        raise KeyError(
            f"família desconhecida: {nome!r} — registradas: {sorted(REGISTRO)}"
        )
    return REGISTRO[nome]

"""
src/radar/cerebro/dossie.py — a FICHA de uma oportunidade.

Ordem do DEV (13/08, ADR 0040): *"o Cérebro não deveria produzir `SUI = 0.82`
sozinho"*. Deveria produzir ativo + regime + relativo + estrutura + momento +
liquidez + risco + posto no universo — porque **o mesmo ativo pode ser excelente
oportunidade de scalp e péssima de swing no mesmo instante**.

A ficha é o objeto que torna a pergunta do Cético respondível por inspeção:
*"por que este ativo e não os outros 200?"*. Sem ela, a resposta é um número, e
número sozinho não se audita.

🔴 O QUE ESTA FICHA SE RECUSA A TER
------------------------------------
O exemplo enviado pelo DEV termina com `OPORTUNIDADE: Alta` e `CONFIANÇA: 0.78`.
**Esses dois campos não existem aqui, e a ausência é deliberada.**

Nenhuma família deste projeto publica hoje `p_ganho` + `payoff_ganho` +
`payoff_perda` — a leitura R1 publica excesso líquido e persistência, que são
outra coisa. Um campo `confianca: 0.78` teria de vir de algum lugar, e o único
lugar disponível seria a invenção. Ficha que carrega juízo fabricado é pior que
ficha incompleta, porque **parece pronta**.

Então cada dimensão carrega o seu próprio estado: `MEDIDO` (com valor),
`NAO_DISPONIVEL` (a fonte não existe neste mercado) ou `NAO_MEDIDO` (existe e
ninguém mediu). Os três são informação diferente e o consumidor precisa
distingui-los — foi confundi-los que fez `NO_TRADE` e `SEM_DETECTOR` parecerem a
mesma coisa por meses.

⚠️ DESCRITIVO, NUNCA AVALIATIVO
--------------------------------
Mesma trava de `deteccao.Situacao`: a ficha REGISTRA onde o ativo está, jamais
se isso é bom. `posto_no_universo = 0,97` é fato; *"força relativa: alta"* é
julgamento, e julgamento embutido na descrição destrói a decomposição de que a
autópsia depende (*resultado da estratégia ≠ propriedade do ativo*).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

__all__ = [
    "DIMENSOES_OBRIGATORIAS",
    "Dimensao",
    "EstadoDaDimensao",
    "FichaDeOportunidade",
    "montar_ficha",
]


class EstadoDaDimensao(StrEnum):
    """Os três estados, e eles NÃO são intercambiáveis.

    Confundi-los é o defeito recorrente deste projeto: `NO_TRADE` significava
    tanto *"procurei e não achei"* quanto *"não tinha como procurar"*, e a
    diferença decide o trabalho seguinte.
    """

    MEDIDO = "medido"
    #: A fonte não existe neste mercado. Nada a fazer sem construir a fonte.
    NAO_DISPONIVEL = "nao_disponivel"
    #: A fonte existe e ninguém mediu. É trabalho, não impedimento.
    NAO_MEDIDO = "nao_medido"


@dataclass(frozen=True, slots=True)
class Dimensao:
    """Um eixo da ficha, com o estado ao lado do valor.

    `valor` e `estado` andam JUNTOS de propósito. Publicar `0.0` para o que não
    foi medido transformaria ausência em afirmação — a mesma razão de
    `ContextoDeMercado.confianca` ser `None` e não `0.0`.
    """

    nome: str
    estado: EstadoDaDimensao
    valor: float | str | None = None
    #: Por que está `NAO_DISPONIVEL`/`NAO_MEDIDO`. **Obrigatório** nesses dois
    #: casos: ausência sem motivo é ausência que ninguém investiga.
    motivo: str | None = None

    def __post_init__(self) -> None:
        if self.estado is EstadoDaDimensao.MEDIDO and self.valor is None:
            raise ValueError(f"dimensão {self.nome!r} MEDIDO sem valor")
        if self.estado is not EstadoDaDimensao.MEDIDO and not self.motivo:
            raise ValueError(
                f"dimensão {self.nome!r} está {self.estado.value} sem motivo — "
                "ausência sem motivo declarado é ausência que ninguém investiga"
            )


#: Os eixos que TODA ficha declara, mesmo quando não há dado.
#:
#: Fixos e obrigatórios porque ficha que omite o eixo faltante esconde o buraco:
#: quem lê não distingue *"o ativo não tem pares comparáveis"* de *"esqueci de
#: calcular"*. Acrescentar eixo é decisão de arquitetura, não de conveniência.
DIMENSOES_OBRIGATORIAS: Final[tuple[str, ...]] = (
    "regime",
    "posto_no_universo",
    "posto_no_setor",
    "vs_benchmark",
    "estrutura",
    "momento",
    "liquidez",
    "microestrutura",
    "risco",
    "expectativa",
)


@dataclass(frozen=True, slots=True)
class FichaDeOportunidade:
    """`ativo + contexto + relação` — a unidade que o ADR 0040 fixa.

    **Não é uma oportunidade**: é o dossiê que permite decidir se existe uma.
    `radar.cerebro.oportunidade.Oportunidade` continua sendo o objeto com
    identidade e fim declarado; a ficha é o que se olha antes de criá-lo.
    """

    ativo: str
    instante: str
    particao: str
    dimensoes: dict[str, Dimensao] = field(default_factory=dict)

    def __post_init__(self) -> None:
        faltando = set(DIMENSOES_OBRIGATORIAS) - set(self.dimensoes)
        if faltando:
            raise ValueError(
                f"ficha de {self.ativo!r} omite {sorted(faltando)} — eixo omitido "
                "esconde o buraco: quem lê não distingue 'não há dado' de "
                "'esqueci de calcular'"
            )

    @property
    def medidas(self) -> int:
        return sum(
            1 for d in self.dimensoes.values() if d.estado is EstadoDaDimensao.MEDIDO
        )

    @property
    def cobertura(self) -> float:
        """Fração das dimensões efetivamente medidas.

        Vem antes de qualquer leitura da ficha, pela mesma razão que a cobertura
        vem antes da repartição no diagnóstico de desfechos: uma ficha com 3 de
        10 eixos medidos não sustenta comparação com outra que tenha 9.
        """
        return self.medidas / len(DIMENSOES_OBRIGATORIAS)

    def por_estado(self, estado: EstadoDaDimensao) -> tuple[str, ...]:
        return tuple(
            sorted(n for n, d in self.dimensoes.items() if d.estado is estado)
        )


def _medido(nome: str, valor: float | str) -> Dimensao:
    return Dimensao(nome=nome, estado=EstadoDaDimensao.MEDIDO, valor=valor)


def _ausente(nome: str, motivo: str) -> Dimensao:
    return Dimensao(nome=nome, estado=EstadoDaDimensao.NAO_DISPONIVEL, motivo=motivo)


def _nao_medido(nome: str, motivo: str) -> Dimensao:
    return Dimensao(nome=nome, estado=EstadoDaDimensao.NAO_MEDIDO, motivo=motivo)


def montar_ficha(
    *,
    ativo: str,
    instante: str,
    particao: str,
    regime: str,
    posto_no_universo: float | None,
    valores: dict[str, Any],
    ev_medido: float | None = None,
) -> FichaDeOportunidade:
    """Monta a ficha a partir do que o ciclo já tem, declarando o que falta.

    `valores` é o que veio do store para ESTE ativo neste instante. Chave
    ausente vira `NAO_MEDIDO` com motivo, nunca `0.0` — e é aqui que a ficha
    ganha o seu valor: ela publica o buraco em vez de preenchê-lo.

    `ev_medido` é o EV LÍQUIDO da estratégia que selecionou o ativo, lido da
    leitura oficial da família. **Não é um score do ativo** — todos os
    candidatos da mesma célula recebem o mesmo valor, e é assim que deve ser:
    *resultado da estratégia ≠ propriedade do ativo*. Quem discrimina entre
    ativos é `posto_no_universo`.
    """
    d: dict[str, Dimensao] = {
        "regime": _medido("regime", regime)
        if regime != "desconhecido"
        else _nao_medido("regime", "detector devolveu DESCONHECIDO neste instante"),
    }

    d["posto_no_universo"] = (
        _medido("posto_no_universo", posto_no_universo)
        if posto_no_universo is not None
        else _nao_medido(
            "posto_no_universo", "ativo fora da seção transversal deste instante"
        )
    )

    # 🔴 O eixo que a ordem do DEV pediu e que NÃO EXISTE. Medido em 13/08: o
    # store de cripto tem ZERO colunas de setor, categoria, grupo ou cluster.
    #
    # Não é detalhe: num universo de 470 ativos heterogêneos, *"subiu mais que
    # todo mundo"* e *"subiu mais que seus pares"* são informações diferentes, e
    # a segunda é a que distingue mérito de maré. Card:
    # `CEREBRO_PARES_COMPARAVEIS_CRIPTO01`.
    d["posto_no_setor"] = _ausente(
        "posto_no_setor",
        "não há agrupamento setorial de cripto no store (0 colunas de "
        "setor/categoria/cluster) — card CEREBRO_PARES_COMPARAVEIS_CRIPTO01",
    )

    #: `ticker -> valor` para os eixos que o store já resolve.
    de_coluna = {
        "vs_benchmark": ("beta_63", "beta contra o benchmark"),
        "estrutura": ("largura_banda_20_2", "largura de banda 20/2"),
        "momento": ("razao_atr_5_20", "razão ATR 5/20"),
        "liquidez": ("faixa_liquidez", "faixa de liquidez"),
        "risco": ("atr_14_pct", "ATR 14 em % do preço"),
    }
    for eixo, (coluna, descricao) in de_coluna.items():
        valor = valores.get(coluna)
        d[eixo] = (
            _medido(eixo, valor)
            if valor is not None
            else _nao_medido(eixo, f"{descricao} (`{coluna}`) ausente para este ativo")
        )

    d["microestrutura"] = _ausente(
        "microestrutura",
        "livro e fluxo agressor existem em barras de 1s, não no store diário — "
        "a leitura diária não sustenta a pergunta de microestrutura",
    )

    # A expectativa é o EV LÍQUIDO medido da estratégia que selecionou este
    # ativo — propriedade da ESTRATÉGIA, jamais do ativo. Todos os candidatos da
    # mesma célula compartilham o mesmo valor, e isso não é simplificação: é o
    # guardrail *resultado da estratégia ≠ propriedade do ativo* em forma
    # executável. O que discrimina entre ativos é `posto_no_universo`.
    #
    # `None` continua virando NAO_MEDIDO (não `NAO_DISPONIVEL`): a fonte existe
    # — são os trades OOS —, o que falta é a família ter sido medida.
    d["expectativa"] = (
        _medido("expectativa", ev_medido)
        if ev_medido is not None
        else _nao_medido(
            "expectativa",
            "a família que selecionou este ativo não publicou expectativa "
            "medida (p_ganho + payoffs) — card "
            "CEREBRO_EXPECTATIVA_MEDIDA_POR_FAMILIA01",
        )
    )

    return FichaDeOportunidade(
        ativo=ativo, instante=instante, particao=particao, dimensoes=d
    )

"""
src/radar/cerebro/capacidade.py — quanto capital cada hipótese comporta.

O que esta lacuna era, e o que este módulo honestamente entrega
---------------------------------------------------------------
`capacidade` é um dos seis critérios de `VALIDADO`, e era o único que estava
`False` em **todas** as 51 hipóteses do Evidence Map: ninguém no projeto tinha
medido quanto capital uma estratégia absorve antes de o próprio fill comer o
resultado. Alocar capital sem isso é chutar o tamanho da posição.

**O que este módulo NÃO é**: medição de fills reais. Capacidade de verdade se
mede executando e comparando preço decidido × preço obtido. O projeto não tem
esses fills (o broker é de papel e o fill é imediato a preço de barra).

**O que ele É**: uma estimativa sob **modelo declarado**, com os parâmetros
visíveis e a sensibilidade reportada. É o teto do que se pode afirmar com dado
histórico de preço e volume — e é muito melhor do que `False`, que dizia
"desconhecido" sem dizer a ordem de grandeza.

O modelo: lei da raiz quadrada
-------------------------------
    impacto = Y · σ_diária · √(Q / ADV)

`Q` é o tamanho da ordem e `ADV` o volume financeiro diário típico, ambos na
mesma moeda. É o modelo padrão da literatura de microestrutura (Almgren–Chriss,
Kyle) e o que a prática usa: **negociar 1% do ADV custa da ordem de 10% da
volatilidade diária**. Com `Y = 1` e a B3 medida (σ ≈ 2,63%/dia), 1% do ADV sai
a ≈ 26 bps — que é a ordem de grandeza que operadores relatam.

Invertendo para o capital que cabe num orçamento `B` (em bps por lado):

    Q = ADV · ( B / (Y · σ · 10.000) )²

O orçamento não é um número solto — é uma FRAÇÃO da margem da hipótese
-----------------------------------------------------------------------
Perguntar "quanto cabe a 10 bps?" é arbitrário: 10 bps é fatal para uma
estratégia que ganha 15 bps por trade e irrelevante para uma que ganha 130. O
orçamento correto é **uma fração do slippage de equilíbrio da própria
hipótese** — quanto do edge se aceita gastar em execução.

    orçamento_bps = fração · equilíbrio_slippage_bps_lado

Com `FRACAO_DO_EDGE_PADRAO = 0,30`, a estimativa responde: *quanto capital cabe
sem gastar mais de 30% do edge medido em impacto?* Isso amarra capacidade,
custo e edge num número só, em vez de três suposições soltas.

As três premissas, ditas para poderem ser contestadas
------------------------------------------------------
1. **`ADV` do universo, não dos ativos negociados.** Os ledgers guardam
   agregados, não a lista de trades — não dá para saber QUAIS papéis cada
   hipótese tocou. Usa-se o **p25 do ADV do mercado** como "o ativo restritivo
   típico", que é conservador: assume que a estratégia negocia também a parte
   menos líquida do universo. Fechar isso de verdade exige persistir a lista de
   trades no ledger (card `CEREBRO_LEDGER_TRADES_PERSISTIDOS01`).
2. **Concorrência derivada de `trades × horizonte / dias`.** Cada trade ocupa
   capital por ~`horizonte` dias; quando o horizonte é desconhecido usa-se
   **1**, que subestima a concorrência e portanto **subestima a capacidade** —
   erra para o lado seguro.
3. **`Y = 1`**, com sensibilidade reportada em 0,5 (mercado fundo) e 2,0
   (pessimista). `Y` é o parâmetro que mais move o resultado e por isso nunca
   sai sozinho no relatório.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

COEFICIENTE_IMPACTO_PADRAO: Final[float] = 1.0
"""`Y` da lei da raiz quadrada. Ver §3 das premissas."""

COEFICIENTES_SENSIBILIDADE: Final[tuple[float, ...]] = (0.5, 1.0, 2.0)

FRACAO_DO_EDGE_PADRAO: Final[float] = 0.30
"""Quanto do edge medido se aceita gastar em impacto de execução."""

PERCENTIL_ADV_PADRAO: Final[float] = 0.25
"""O "ativo restritivo típico" do universo. Conservador de propósito."""


@dataclass(frozen=True)
class PerfilDeLiquidez:
    """O retrato de liquidez de um mercado, medido do histórico de preço/volume."""

    mercado: str
    adv_percentil: float
    """Volume financeiro diário no `PERCENTIL_ADV_PADRAO` do universo."""
    sigma_diaria: float
    n_ativos: int
    percentil: float = PERCENTIL_ADV_PADRAO
    moeda: str = "BRL"


@dataclass(frozen=True)
class CapacidadeEstimada:
    capital_por_posicao: float
    posicoes_concorrentes: float
    capital_total: float
    orcamento_bps: float
    sensibilidade: dict[str, float]
    """`capital_total` recalculado para cada `Y` de `COEFICIENTES_SENSIBILIDADE`."""
    premissas: tuple[str, ...]
    moeda: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "capital_por_posicao": self.capital_por_posicao,
            "posicoes_concorrentes": self.posicoes_concorrentes,
            "capital_total": self.capital_total,
            "orcamento_bps": self.orcamento_bps,
            "sensibilidade_por_coeficiente": self.sensibilidade,
            "premissas": list(self.premissas),
            "moeda": self.moeda,
            "natureza": "ESTIMATIVA por modelo declarado — NÃO é medição de fills reais",
        }


def impacto_bps(
    participacao: float, sigma_diaria: float, coef: float = COEFICIENTE_IMPACTO_PADRAO
) -> float:
    """
    `Y · σ · √(Q/ADV)`, em bps. `participacao = Q/ADV`.

    Levanta em participação negativa: raiz de número negativo devolveria `nan`
    e um `nan` silencioso aqui viraria capacidade "desconhecida" indistinguível
    de capacidade calculada.
    """
    if participacao < 0:
        raise ValueError(f"participacao={participacao!r} negativa")
    if sigma_diaria <= 0:
        raise ValueError(f"sigma_diaria={sigma_diaria!r} precisa ser > 0")
    return coef * sigma_diaria * math.sqrt(participacao) * 10_000.0


def capital_por_posicao(
    adv: float,
    sigma_diaria: float,
    orcamento_bps: float,
    coef: float = COEFICIENTE_IMPACTO_PADRAO,
) -> float:
    """
    Inverte a lei da raiz quadrada: o maior `Q` cujo impacto cabe em
    `orcamento_bps`. Orçamento ≤ 0 devolve 0 — sem margem para gastar em
    execução, não há capital que caiba.
    """
    if orcamento_bps <= 0:
        return 0.0
    if sigma_diaria <= 0 or adv <= 0 or coef <= 0:
        raise ValueError("adv, sigma_diaria e coef precisam ser > 0")
    participacao = (orcamento_bps / (coef * sigma_diaria * 10_000.0)) ** 2
    return adv * participacao


def posicoes_concorrentes(
    trades: int, dias_independentes: int, horizonte: int | None
) -> float:
    """
    `trades × horizonte / dias` — quantas posições coexistem em média.

    `horizonte` ausente ou zero vira **1**: subestima a concorrência e portanto
    a capacidade total. Errar para baixo aqui é a direção segura (premissa §2).
    """
    if dias_independentes <= 0:
        return 0.0
    return trades * max(1, horizonte or 0) / dias_independentes


def estimar_capacidade(
    *,
    perfil: PerfilDeLiquidez,
    equilibrio_slippage_bps: float,
    trades: int,
    dias_independentes: int,
    horizonte: int | None,
    fracao_do_edge: float = FRACAO_DO_EDGE_PADRAO,
) -> CapacidadeEstimada | None:
    """
    A estimativa completa. Devolve `None` quando não há margem para gastar em
    execução (`equilibrio_slippage_bps <= 0`) — e `None` aqui significa
    "capacidade zero por falta de edge", não "não calculei": o chamador deve
    tratar como critério REPROVADO, nunca como ausente.
    """
    orcamento = fracao_do_edge * equilibrio_slippage_bps
    if orcamento <= 0:
        return None

    concorrentes = posicoes_concorrentes(trades, dias_independentes, horizonte)
    por_posicao = capital_por_posicao(
        perfil.adv_percentil, perfil.sigma_diaria, orcamento
    )
    sensibilidade = {
        f"Y={c:g}": capital_por_posicao(
            perfil.adv_percentil, perfil.sigma_diaria, orcamento, coef=c
        ) * concorrentes
        for c in COEFICIENTES_SENSIBILIDADE
    }
    premissas = (
        f"ADV = p{perfil.percentil:.0%} do universo {perfil.mercado} "
        f"({perfil.adv_percentil:,.0f} {perfil.moeda}, {perfil.n_ativos} ativos) — "
        "o ativo restritivo típico, não os ativos de fato negociados",
        f"σ diária = {perfil.sigma_diaria:.4f} (mediana do universo)",
        f"orçamento = {fracao_do_edge:.0%} do equilíbrio de "
        f"{equilibrio_slippage_bps:.1f} bps/lado = {orcamento:.1f} bps",
        f"concorrência = {trades} trades × {max(1, horizonte or 0)} dias de holding "
        f"/ {dias_independentes} dias = {concorrentes:.1f} posições",
        f"Y = {COEFICIENTE_IMPACTO_PADRAO:g} (sensibilidade em "
        f"{', '.join(f'{c:g}' for c in COEFICIENTES_SENSIBILIDADE)})",
    )
    return CapacidadeEstimada(
        capital_por_posicao=por_posicao,
        posicoes_concorrentes=concorrentes,
        capital_total=por_posicao * concorrentes,
        orcamento_bps=orcamento,
        sensibilidade=sensibilidade,
        premissas=premissas,
        moeda=perfil.moeda,
    )


# ── capacidade pelos ativos REALMENTE negociados (ADR 0033) ──────────────────
FAIXAS_DE_CAPITAL: Final[tuple[float, ...]] = (
    1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000
)
"""
As faixas em que a curva de degradação é reportada.

Elas cobrem o caminho inteiro que o produto vai percorrer — do paper de R$ 5 mil
de hoje até a escala de fundo — porque a pergunta útil nunca foi *"suporta R$ 100
mil?"* e sim **"até onde ela suporta crescer?"**.
"""


@dataclass(frozen=True)
class PontoDaCurva:
    capital: float
    edge_liquido: float
    """Retorno médio por trade DEPOIS do impacto de execução naquele tamanho."""
    impacto_bps: float
    participacao_media: float


@dataclass(frozen=True)
class CurvaDeDegradacao:
    """
    A capacidade como **atributo contínuo**, não como booleano.

    O erro que este objeto existe para desfazer (formulação do DEV, 2026-08-04):
    colapsar um contínuo em `SIM`/`NÃO` cedo demais faz a estratégia parecer pior
    do que é. Uma que suporta R$ 40 mil não *deixa de ser boa* — ela tem um
    limite operacional diferente, e é perfeitamente utilizável enquanto o capital
    for menor que ele.
    """

    pontos: tuple[PontoDaCurva, ...]
    limite_seguro: float | None
    """Maior capital em que o edge permanece ≥ 90% do edge sem impacto."""
    limite_critico: float | None
    """Menor capital em que o edge cruza zero. `None` = não cruza dentro das
    faixas testadas — e isso é informação, não ausência dela."""
    edge_sem_impacto: float
    moeda: str

    @property
    def faixa_de_degradacao(self) -> tuple[float | None, float | None]:
        """Entre o seguro e o crítico: o edge cai mas segue positivo."""
        return (self.limite_seguro, self.limite_critico)

    def resumo_legivel(self) -> str:
        seg = f"{self.limite_seguro:,.0f}" if self.limite_seguro else "—"
        cri = f"{self.limite_critico:,.0f}" if self.limite_critico else "não cruza"
        return f"✓ até {seg} · degrada acima disso · ✗ em {cri} ({self.moeda})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_sem_impacto": self.edge_sem_impacto,
            "limite_seguro": self.limite_seguro,
            "limite_critico": self.limite_critico,
            "moeda": self.moeda,
            "resumo": self.resumo_legivel(),
            "pontos": [
                {"capital": p.capital, "edge_liquido": p.edge_liquido,
                 "impacto_bps": p.impacto_bps,
                 "participacao_media": p.participacao_media}
                for p in self.pontos
            ],
        }


def curva_de_degradacao(
    *,
    edge_por_trade: float,
    adv_dos_ativos: Sequence[float],
    sigma_diaria: float,
    posicoes_concorrentes: float,
    faixas: Sequence[float] = FAIXAS_DE_CAPITAL,
    coef: float = COEFICIENTE_IMPACTO_PADRAO,
    moeda: str = "BRL",
) -> CurvaDeDegradacao | None:
    """
    Quanto do edge sobra em cada escala de capital.

    `adv_dos_ativos` é o ADV dos tickers **de fato negociados**, um por trade (ou
    por ativo, ponderado pela frequência) — **não** o p25 do universo. A
    diferença não é cosmética: no cripto o espalhamento entre p25 e p90 do ADV é
    de **780×**, e usar um número só trataria BTC e uma altcoin do rabo da
    distribuição como o mesmo ativo.

    Devolve `None` sem ADV — e `None` é **não medida**, que o chamador reprova.
    """
    advs = [float(a) for a in adv_dos_ativos if a and a > 0]
    if not advs or sigma_diaria <= 0 or posicoes_concorrentes <= 0:
        return None

    # A média HARMÔNICA, não a aritmética: o impacto escala com √(Q/ADV), então
    # o ativo ilíquido pesa desproporcionalmente na conta. A média aritmética
    # deixaria um punhado de nomes líquidos esconder a cauda ruim — que é
    # exatamente o erro que a estimativa por p25 do universo cometia ao contrário.
    adv_efetivo = len(advs) / sum(1.0 / a for a in advs)

    pontos: list[PontoDaCurva] = []
    for capital in faixas:
        por_posicao = capital / posicoes_concorrentes
        participacao = por_posicao / adv_efetivo
        imp_bps = impacto_bps(participacao, sigma_diaria, coef=coef)
        pontos.append(PontoDaCurva(
            capital=float(capital),
            edge_liquido=edge_por_trade - imp_bps / 10_000.0,
            impacto_bps=imp_bps,
            participacao_media=participacao,
        ))

    limite_seguro = next(
        (p.capital for p in reversed(pontos)
         if p.edge_liquido >= 0.90 * edge_por_trade), None
    )
    limite_critico = next((p.capital for p in pontos if p.edge_liquido <= 0), None)

    return CurvaDeDegradacao(
        pontos=tuple(pontos), limite_seguro=limite_seguro,
        limite_critico=limite_critico, edge_sem_impacto=edge_por_trade, moeda=moeda,
    )


def adv_dos_trades(
    trades: Any, adv_por_ticker: Mapping[str, float]
) -> list[float]:
    """
    O ADV de cada trade, na frequência em que os ativos foram negociados.

    Ticker sem ADV conhecido é **descartado com o fato visível para o chamador**
    (o tamanho da lista devolvida encolhe) em vez de receber um ADV padrão —
    inventar liquidez para o ativo do qual menos se sabe é como um backtest passa
    a mentir a favor.
    """
    if trades is None or len(trades) == 0 or "ticker" not in trades:
        return []
    return [
        float(adv_por_ticker[t])
        for t in trades["ticker"]
        if t in adv_por_ticker and adv_por_ticker[t] > 0
    ]

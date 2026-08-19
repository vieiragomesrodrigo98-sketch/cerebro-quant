"""
Camada de execução — modelo de custos e slippage da B3 (ações à vista).

Hoje o P&L de paper é medido SEM custos, slippage ou liquidez (dívida A-M3 da
auditoria): sistematicamente otimista. Este modelo permite marcar cada perna
(compra/venda) com os custos reais da B3, para que a tese seja medida como seria
executada de verdade — pré-requisito para ordens reais e para um paper honesto.

Escopo: custos de TRANSAÇÃO (corretagem, emolumentos, liquidação, ISS) + slippage.
Imposto de renda (day trade 20% / swing 15%, compensável mês a mês) é uma camada
SEPARADA de apuração — não entra aqui, pois depende de agregação mensal e prejuízos
acumulados, não do trade isolado.

Valores default realistas 2024-2025 (configuráveis). Corretagem default = 0
(padrão atual de mercado); ajuste por corretora.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

# Custos B3 para ações à vista (operação normal). Percentuais sobre o financeiro.
DEFAULT_EMOLUMENTOS_PCT = Decimal("0.00005")  # 0,005% — taxa de negociação B3
DEFAULT_LIQUIDACAO_PCT = Decimal("0.00025")  # 0,025% — taxa de liquidação B3
DEFAULT_CORRETAGEM_FIXA = Decimal("0.00")  # corretagem fixa por ordem
DEFAULT_CORRETAGEM_PCT = Decimal("0.00")  # corretagem percentual sobre o financeiro
DEFAULT_ISS_PCT = Decimal("0.05")  # 5% sobre a corretagem (ISS municipal)
DEFAULT_SLIPPAGE_BPS = Decimal("5")  # 5 bps = 0,05% adverso (ações líquidas)

_CENT = Decimal("0.01")
_BPS = Decimal("10000")


def to_cents(x: Decimal) -> Decimal:
    """Arredonda para centavos (2 casas, half-up)."""
    return x.quantize(_CENT, rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class TradeCosts:
    """Custos de UMA perna (compra OU venda)."""

    corretagem: Decimal
    emolumentos: Decimal
    liquidacao: Decimal
    iss: Decimal

    @property
    def total(self) -> Decimal:
        return to_cents(self.corretagem + self.emolumentos + self.liquidacao + self.iss)


@dataclass(frozen=True)
class B3CostModel:
    """Custos e slippage da B3 para ações à vista. Todos os campos são configuráveis."""

    emolumentos_pct: Decimal = DEFAULT_EMOLUMENTOS_PCT
    liquidacao_pct: Decimal = DEFAULT_LIQUIDACAO_PCT
    corretagem_fixa: Decimal = DEFAULT_CORRETAGEM_FIXA
    corretagem_pct: Decimal = DEFAULT_CORRETAGEM_PCT
    iss_pct: Decimal = DEFAULT_ISS_PCT
    slippage_bps: Decimal = DEFAULT_SLIPPAGE_BPS

    def costs_for(self, notional: Decimal) -> TradeCosts:
        """Custos de uma perna sobre o valor financeiro (notional = qtd x preço)."""
        notional = abs(notional)
        corretagem = to_cents(self.corretagem_fixa + notional * self.corretagem_pct)
        emolumentos = to_cents(notional * self.emolumentos_pct)
        liquidacao = to_cents(notional * self.liquidacao_pct)
        iss = to_cents(corretagem * self.iss_pct)
        return TradeCosts(corretagem, emolumentos, liquidacao, iss)

    def apply_slippage(self, price: Decimal, is_buy: bool) -> Decimal:
        """Preço de execução após slippage adverso: compra paga mais, venda recebe menos."""
        adj = price * self.slippage_bps / _BPS
        return to_cents(price + adj) if is_buy else to_cents(price - adj)


# ── Cripto (Laboratório de Estratégias Técnicas, ADR 0021, Fase A4) ───────────
#
# `CryptoCostModel` é a IRMÃ de `B3CostModel` — mesma interface pública
# (`apply_slippage` + `costs_for`), para que o harness do laboratório (Fase
# A2+) trate os dois mercados de forma intercambiável sem precisar saber qual
# é qual. `B3CostModel` acima permanece INTOCADA.
#
# Taxonomia diferente de propósito: emolumento/liquidação/ISS são taxas
# específicas da B3 e não existem em corretora de cripto. O único custo de
# TRANSAÇÃO modelado aqui é a taxa TAKER da exchange (o motor entra a
# mercado, nunca maker). Fora do escopo v1, registrado e NÃO escondido:
#   - funding rate — só existe em perpétuos/futuros; v1 não opera derivativo.
#   - spread bid/ask — o slippage aproxima o efeito, não mede o livro real.
#   - taxa maker / desconto por volume ou token da exchange (ex.: BNB).
DEFAULT_TAKER_FEE_PCT = Decimal("0.0010")  # 0,10% por lado — taker padrão sem desconto
DEFAULT_CRYPTO_SLIPPAGE_BPS = Decimal("5")  # 5 bps — mesmo default conservador do B3CostModel

# ── Custo por VENUE e por tipo de ordem (CRIPTO_CUSTO_VENUE_ERRADA01) ─────────
#
# O default acima é taker no SPOT: 0,20% de round-trip. A venue que o produto
# declara operar é FUTUROS PERPÉTUOS com ordem MAKER, cujo round-trip é 0,04%
# (ADR 0031 §6.1) — **5× mais barato**.
#
# Por que isso é P0 e não um ajuste fino: o erro não esconde alpha, ele empurra
# hipótese para REFUTAÇÃO FALSA. Medido em `scalp_micro_rompimento` (1.778
# trades, 578 dias): com 0,20% o `t_NW` é −1,99; com 0,04%, −0,58 — de *quase
# refutada* para *inconclusiva*. E o denominador do FDR nunca devolve a
# tentativa que se gastou refutando por engano.
#
# Uma constante única não dá conta porque o custo depende de DUAS escolhas
# independentes (mercado à vista ou perpétuo; ordem agressora ou passiva), e
# scalping só fecha a conta na combinação mais barata.
TAXA_POR_VENUE: dict[tuple[str, str], Decimal] = {
    ("spot", "taker"): Decimal("0.0010"),
    ("spot", "maker"): Decimal("0.0008"),
    ("perpetuo", "taker"): Decimal("0.0005"),
    ("perpetuo", "maker"): Decimal("0.0002"),
}
"""Taxa por PERNA. Valores de tabela pública da Binance, nível base, sem
desconto por volume nem por token da exchange — desconto é vantagem que o
projeto não tem até prová-la."""

VENUES_VALIDAS = tuple(sorted({v for v, _ in TAXA_POR_VENUE}))
TIPOS_DE_ORDEM = tuple(sorted({t for _, t in TAXA_POR_VENUE}))


def taxa_da_venue(venue: str, tipo_ordem: str) -> Decimal:
    """
    Taxa por perna da combinação (venue, tipo de ordem).

    Combinação desconhecida **levanta** em vez de cair num default: um custo
    silenciosamente errado é exatamente o defeito que este mapa existe para
    impedir, e escolher o default mais barato seria otimismo embutido.
    """
    chave = (venue, tipo_ordem)
    if chave not in TAXA_POR_VENUE:
        raise ValueError(
            f"combinação de custo desconhecida: venue={venue!r}, "
            f"tipo_ordem={tipo_ordem!r}. Válidas: "
            f"venues {VENUES_VALIDAS}, ordens {TIPOS_DE_ORDEM}"
        )
    return TAXA_POR_VENUE[chave]


@dataclass(frozen=True)
class CryptoTradeCosts:
    """Custos de UMA perna (compra OU venda) em cripto — só a taxa taker."""

    taker_fee: Decimal

    @property
    def total(self) -> Decimal:
        # SEM quantização a centavo: cripto não tem tick de R$0,01 — DOGE em
        # 2019 negociava a US$0,002 e uma taxa de 0,10% sobre notional pequeno
        # quantizada a centavo viraria ZERO (foi exatamente o 0/0 que derrubou
        # a 1ª rodada A7 em dado real). `to_cents` é premissa de ação BRL,
        # vale só para o B3CostModel.
        return self.taker_fee


@dataclass(frozen=True)
class CryptoCostModel:
    """
    Custos e slippage de cripto (v1) — todos os campos são configuráveis.

    Mesma interface pública de `B3CostModel` (`apply_slippage`/`costs_for`),
    ver nota da seção acima para o que fica de fora do modelo v1.
    """

    taker_fee_pct: Decimal = DEFAULT_TAKER_FEE_PCT
    slippage_bps: Decimal = DEFAULT_CRYPTO_SLIPPAGE_BPS

    @classmethod
    def da_venue(
        cls,
        venue: str,
        tipo_ordem: str,
        *,
        slippage_bps: Decimal | None = None,
    ) -> CryptoCostModel:
        """
        O modelo com a taxa da venue e do tipo de ordem declarados.

        O default da classe permanece **taker spot** de propósito: mudá-lo
        alteraria em silêncio toda leitura já publicada. Quem quer o custo da
        venue real pede por ele — e fica registrado na chamada qual venue a
        leitura assume.

        Ordem MAKER não atravessa o spread, então o slippage adverso cai a
        zero: quem é passivo não paga o livro, espera nele. Passar
        `slippage_bps` explicitamente sobrepõe essa escolha.
        """
        if slippage_bps is None:
            slippage_bps = (
                Decimal("0") if tipo_ordem == "maker" else DEFAULT_CRYPTO_SLIPPAGE_BPS
            )
        return cls(
            taker_fee_pct=taxa_da_venue(venue, tipo_ordem),
            slippage_bps=slippage_bps,
        )

    @property
    def round_trip_pct(self) -> Decimal:
        """Custo das DUAS pernas, que é a unidade em que a régua econômica fala."""
        return self.taker_fee_pct * 2

    def costs_for(self, notional: Decimal) -> CryptoTradeCosts:
        """Custo de uma perna sobre o valor financeiro (notional = qtd x preço)."""
        notional = abs(notional)
        return CryptoTradeCosts(taker_fee=notional * self.taker_fee_pct)

    def apply_slippage(self, price: Decimal, is_buy: bool) -> Decimal:
        """
        Preço de execução após slippage adverso — mesma fórmula do
        `B3CostModel`, mas SEM `to_cents`: preço de cripto tem granularidade
        sub-centavo (DOGE 2019 ~US$0,002; quantizar a centavo zera o preço e
        produz 0/0 no cálculo de retorno — bug real pego pela 1ª rodada A7).
        """
        adj = price * self.slippage_bps / _BPS
        return price + adj if is_buy else price - adj

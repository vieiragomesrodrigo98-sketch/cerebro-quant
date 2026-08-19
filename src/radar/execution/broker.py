"""
Camada de execução — abstração de broker (seam paper→real) e execução simulada.

Define a interface `Broker` (Protocol) e a única implementação existente hoje,
`PaperBroker`, que simula o fill aplicando slippage e custos B3 (ou cripto —
ver `CostModel` abaixo, S151-B). Um adapter real (futuro) implementaria a MESMA
interface — é aqui que o paper→real acontece com troca de implementação, não
de chamadas.

IMPORTANTE (ADR 0001): NÃO existe, aqui nem em lugar nenhum, cliente HTTP/SDK
conectando a corretora. `enable_real_trading` continua bloqueado por design em
`config/settings.py`. Esta camada é a FUNDAÇÃO testável; habilitar ordens reais
exige um adapter novo + revisão do hard-stop + ADR de liberação (ver ADR 0013).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Protocol

from radar.execution.costs import (
    B3CostModel,
    CryptoCostModel,
    CryptoTradeCosts,
    TradeCosts,
    to_cents,
)

# S151-B: `PaperBroker`/`net_return_pct` tratam B3CostModel e CryptoCostModel
# como intercambiáveis — os dois implementam o mesmo contrato duck-typed
# (`costs_for(notional).total` + `apply_slippage(price, is_buy)`). Union
# explícita em vez de Protocol: mais simples de ler/tipar aqui, e os únicos
# dois modelos que existem hoje são exatamente esses dois.
CostModel = B3CostModel | CryptoCostModel

# Notional de referência grande: torna o arredondamento em centavos desprezível ao
# computar retorno percentual (que é quantidade-independente com custos proporcionais).
_REF_NOTIONAL = Decimal("1000000")
_REF_AT = datetime(2000, 1, 1, tzinfo=UTC)  # irrelevante ao cálculo (só carimba Fill.at)


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class Order:
    ticker: str
    side: OrderSide
    quantity: int
    reference_price: Decimal   # preço de referência (last/mid) para simular o fill


@dataclass(frozen=True)
class Fill:
    ticker: str
    side: OrderSide
    quantity: int
    fill_price: Decimal        # preço executado, após slippage
    gross_value: Decimal       # qtd x fill_price
    costs: TradeCosts | CryptoTradeCosts  # custos da perna (B3 ou cripto — S151-B)
    net_value: Decimal         # BUY: desembolso (gross + custos); SELL: recebido (gross - custos)
    at: datetime


class Broker(Protocol):
    """Interface de execução. Implementações: PaperBroker (hoje); real (futuro)."""
    def submit(self, order: Order, *, at: datetime) -> Fill: ...


class PaperBroker:
    """
    Execução SIMULADA honesta: aplica slippage adverso e custos B3 (default)
    ou cripto (`cost_model=CryptoCostModel()`, S151-B). Não toca corretora.
    """

    def __init__(self, cost_model: CostModel | None = None) -> None:
        self.costs: CostModel = cost_model or B3CostModel()

    def submit(self, order: Order, *, at: datetime) -> Fill:
        if order.quantity <= 0:
            raise ValueError("Order.quantity deve ser > 0")
        is_buy = order.side == OrderSide.BUY
        fill_price = self.costs.apply_slippage(order.reference_price, is_buy)
        gross = to_cents(fill_price * order.quantity)
        c = self.costs.costs_for(gross)
        net = to_cents(gross + c.total) if is_buy else to_cents(gross - c.total)
        return Fill(order.ticker, order.side, order.quantity, fill_price, gross, c, net, at)


def round_trip_net_pnl(entry: Fill, exit_: Fill) -> Decimal:
    """
    P&L líquido de um round-trip LONG (compra→venda), já com slippage e custos das
    duas pernas: recebido na venda - desembolsado na compra.
    """
    return to_cents(exit_.net_value - entry.net_value)


def net_return_pct(
    entry_price: Decimal,
    exit_price: Decimal,
    *,
    cost_model: CostModel | None = None,
) -> Decimal:
    """
    Retorno % LÍQUIDO de um round-trip LONG (entrada→saída) com slippage e custos B3
    (ou cripto, se `cost_model=CryptoCostModel()` — S151-B) das duas pernas, via
    PaperBroker. Torna honesto o P&L de paper — o bruto (saída-entrada)/entrada
    ignora custos e é sistematicamente otimista (dívida A-M3).

    Quantidade-independente com custos proporcionais (usa notional de referência grande
    para o arredondamento em centavos ser desprezível). Retorna 0 se entrada <= 0.
    """
    if entry_price <= 0:
        return Decimal("0")
    broker = PaperBroker(cost_model)
    qty = int(_REF_NOTIONAL / entry_price) or 1
    buy = broker.submit(Order("_", OrderSide.BUY, qty, entry_price), at=_REF_AT)
    sell = broker.submit(Order("_", OrderSide.SELL, qty, exit_price), at=_REF_AT)
    return round_trip_net_pnl(buy, sell) / (entry_price * qty) * 100

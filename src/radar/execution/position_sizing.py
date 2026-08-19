"""
Camada de execução — position sizing por risco-por-trade.

Hoje não existe sizing: a quantidade é fixa (quantity=1.0), então o P&L não é
ponderado por posição e o risco por trade não é controlado (dívida da auditoria).
Este módulo dimensiona a posição pelo risco: quantas ações comprar de modo que a
perda até o stop seja uma fração fixa do capital.

    risco_por_ação = |entrada - stop|
    qtd = (capital x risk_pct) / risco_por_ação   (limitada por teto de posição e caixa)

LONG-only por ora (os motores só geram LONG). Retorna quantidade INTEIRA (arredonda
para baixo ao lote). Tudo em Decimal.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal

from radar.execution.costs import to_cents

DEFAULT_RISK_PCT = Decimal("0.01")          # 1% do capital arriscado por trade
DEFAULT_MAX_POSITION_PCT = Decimal("0.20")  # teto de 20% do capital em uma posição


@dataclass(frozen=True)
class SizingResult:
    quantity: int          # ações a operar (múltiplo do lote; 0 = não operar)
    risk_amount: Decimal   # R$ arriscado até o stop (qtd x risco_por_ação)
    notional: Decimal      # R$ da posição (qtd x entrada)
    reason: str            # "ok" | "risco_zero" | "abaixo_do_lote" | "parametros_invalidos"


def size_position(
    equity: Decimal,
    entry_price: Decimal,
    stop_price: Decimal,
    *,
    risk_pct: Decimal = DEFAULT_RISK_PCT,
    max_position_pct: Decimal = DEFAULT_MAX_POSITION_PCT,
    lot_size: int = 1,
    available_cash: Decimal | None = None,
) -> SizingResult:
    """Dimensiona uma posição LONG por risco-por-trade. Ver docstring do módulo."""
    if equity <= 0 or entry_price <= 0 or lot_size <= 0:
        return SizingResult(0, Decimal("0"), Decimal("0"), "parametros_invalidos")

    risco_por_acao = abs(entry_price - stop_price)
    if risco_por_acao <= 0:
        return SizingResult(0, Decimal("0"), Decimal("0"), "risco_zero")

    raw_qty = (equity * risk_pct) / risco_por_acao

    # Teto por posição (não concentrar) e por caixa disponível
    qty = min(raw_qty, (equity * max_position_pct) / entry_price)
    cash = available_cash if available_cash is not None else equity
    if cash > 0:
        qty = min(qty, cash / entry_price)

    # Arredonda para baixo ao lote
    q = int((qty / lot_size).to_integral_value(rounding=ROUND_DOWN)) * lot_size
    if q <= 0:
        return SizingResult(0, Decimal("0"), Decimal("0"), "abaixo_do_lote")

    qd = Decimal(q)
    return SizingResult(q, to_cents(qd * risco_por_acao), to_cents(qd * entry_price), "ok")

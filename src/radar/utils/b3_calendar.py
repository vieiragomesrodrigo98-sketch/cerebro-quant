"""Calendário B3 — feriados nacionais e janela de pregão."""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo

BRT = ZoneInfo("America/Sao_Paulo")

# Feriados nacionais fixos (mm-dd) — não incluem Carnaval e Corpus Christi (móveis)
_FIXED_HOLIDAYS: set[tuple[int, int]] = {
    (1, 1),   # Confraternização Universal
    (4, 21),  # Tiradentes
    (5, 1),   # Dia do Trabalho
    (9, 7),   # Independência
    (10, 12), # N. Sra. Aparecida
    (11, 2),  # Finados
    (11, 15), # Proclamação da República
    (11, 20), # Dia da Consciência Negra
    (12, 24), # Véspera de Natal (meio-dia)
    (12, 25), # Natal
    (12, 31), # Véspera de Ano-Novo (meio-dia)
}

# Feriados móveis explícitos (Carnaval 2ª/3ª, Sexta Santa, Corpus Christi)
# Cobrindo 2024-2028 para garantir cobertura suficiente
_VARIABLE_HOLIDAYS: set[date] = {
    # 2024
    date(2024, 2, 12), date(2024, 2, 13),  # Carnaval
    date(2024, 3, 29),                     # Sexta-Feira Santa
    date(2024, 5, 30),                     # Corpus Christi
    # 2025
    date(2025, 3, 3), date(2025, 3, 4),    # Carnaval
    date(2025, 4, 18),                     # Sexta-Feira Santa
    date(2025, 6, 19),                     # Corpus Christi
    # 2026
    date(2026, 2, 16), date(2026, 2, 17),  # Carnaval
    date(2026, 4, 3),                      # Sexta-Feira Santa
    date(2026, 6, 4),                      # Corpus Christi
    # 2027
    date(2027, 2, 8), date(2027, 2, 9),    # Carnaval
    date(2027, 3, 26),                     # Sexta-Feira Santa
    date(2027, 5, 27),                     # Corpus Christi
    # 2028
    date(2028, 2, 28), date(2028, 2, 29),  # Carnaval
    date(2028, 4, 14),                     # Sexta-Feira Santa
    date(2028, 6, 15),                     # Corpus Christi
}

# Pregão regular B3
_MARKET_OPEN  = time(10, 0)
_MARKET_CLOSE = time(17, 0)


def is_b3_holiday(d: date) -> bool:
    if d in _VARIABLE_HOLIDAYS:
        return True
    return (d.month, d.day) in _FIXED_HOLIDAYS


def is_b3_trading_day(d: date | None = None) -> bool:
    """Retorna True se *d* é dia útil B3 (não feriado, não fim de semana)."""
    if d is None:
        d = datetime.now(tz=BRT).date()
    if d.weekday() >= 5:   # sábado=5, domingo=6
        return False
    return not is_b3_holiday(d)


def is_b3_market_open(dt: datetime | None = None) -> bool:
    """Retorna True se *dt* está dentro da janela de pregão B3 (10h–17h BRT)."""
    if dt is None:
        dt = datetime.now(tz=UTC)
    dt_brt = dt.astimezone(BRT)
    if not is_b3_trading_day(dt_brt.date()):
        return False
    return _MARKET_OPEN <= dt_brt.time() < _MARKET_CLOSE


def next_trading_day(from_date: date | None = None) -> date:
    """Retorna o próximo dia útil B3 após *from_date*."""
    from datetime import timedelta
    if from_date is None:
        from_date = datetime.now(tz=BRT).date()
    d = from_date + timedelta(days=1)
    while not is_b3_trading_day(d):
        d += timedelta(days=1)
    return d

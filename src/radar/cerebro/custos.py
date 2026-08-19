"""
src/radar/cerebro/custos.py — custo round-trip **por ativo** (ADR 0032 §7).

Por que isto existe
--------------------
O custo do motor anterior era UNIFORME: `radar.mie.metrics.custo_roundtrip_pct`
devolve a mesma fração para BTC e para uma altcoin de ponta de universo. Isso
tem duas consequências, e a segunda é a grave:

1. subestima o custo do ilíquido e superestima o do líquido;
2. **apaga a única diferença entre os dois braços de alvo da família
   `swing_v1`.** Dentro de um mesmo dia, ranquear por excesso é idêntico a
   ranquear por retorno — o benchmark é constante naquele dia. Os braços A
   (líquido) e B (excesso) só produzem ordens diferentes porque o custo varia
   por ativo. Com custo uniforme, os dois braços são o MESMO experimento.

A fórmula (pré-registro `PRE_REGISTRO_CEREBRO_SWING_V1.md` §3.1, congelado)
--------------------------------------------------------------------------
    custo_roundtrip(ativo) = 2 × ( taxa_pct + slippage_pct × mult(faixa) )

Só o **slippage** escala com liquidez: corretagem/emolumentos são fração fixa
do notional. `taxa_pct`/`slippage_pct` vêm do modelo de custo real da casa
(`radar.execution.costs`), nunca redigitados aqui — se o modelo mudar, este
módulo muda junto, automaticamente.

O multiplicador é MEDIDO, não escolhido: a lei canônica de impacto de mercado
é raiz quadrada do volume, logo `mult(faixa) = sqrt(vol_mediano(liquida) /
vol_mediano(faixa))`, com os volumes medianos na régua de nível ticker de
`radar.evidence.categorias.calcular_faixas_liquidez`. Os valores abaixo são o
resultado dessa medição em 2026-08-04, ANTES da primeira rodada da família.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Final

from radar.execution.costs import B3CostModel, CryptoCostModel

MERCADO_B3: Final[str] = "b3"
MERCADO_CRIPTO: Final[str] = "cripto"

FAIXA_LIQUIDA: Final[str] = "liquida"
FAIXA_MEDIA: Final[str] = "media"
FAIXA_ILIQUIDA: Final[str] = "iliquida"

MULTIPLICADOR_SLIPPAGE: Final[dict[str, dict[str, float]]] = {
    MERCADO_B3: {FAIXA_LIQUIDA: 1.00, FAIXA_MEDIA: 2.13, FAIXA_ILIQUIDA: 4.89},
    MERCADO_CRIPTO: {FAIXA_LIQUIDA: 1.00, FAIXA_MEDIA: 1.95, FAIXA_ILIQUIDA: 3.05},
}
"""Medidos em 2026-08-04 (pré-registro §3.1), `sqrt(vol_mediano(liquida) /
vol_mediano(faixa))`:

  B3     — razões de volume 1,00× / 4,53× / 23,96×
  Cripto — razões de volume 1,00× / 3,81× /  9,30×

Congelados pelo pré-registro: mudar um valor é família nova. `iliquida` está
aqui por completude — o filtro de universo já a exclui do treino."""

MULTIPLICADOR_UNIFORME: Final[float] = 1.00
"""Variante de SENSIBILIDADE exigida pelo pré-registro §3.1: o mesmo braço A
recalculado como se o custo não variasse por ativo. Se o contraste entre os
braços A e B desaparecer nesta variante, o achado é do modelo de CUSTO e não
do modelo de seleção — e o relatório tem de dizer isso com essas palavras."""

_NOTIONAL_REFERENCIA: Final[Decimal] = Decimal("10000")
"""Notional grande o bastante para a quantização a centavo de `costs_for` não
distorcer a fração — mesma constante e mesma razão de
`radar.mie.metrics.custo_roundtrip_pct`."""

_MODELO_PADRAO: Final[dict[str, type[B3CostModel] | type[CryptoCostModel]]] = {
    MERCADO_B3: B3CostModel,
    MERCADO_CRIPTO: CryptoCostModel,
}


def multiplicador_da_faixa(faixa: str | None, *, mercado: str, uniforme: bool = False) -> float:
    """
    Multiplicador de slippage da `faixa` no `mercado`. `uniforme=True` devolve
    sempre `MULTIPLICADOR_UNIFORME` (a variante de sensibilidade).

    **Faixa `None` ou desconhecida recebe o multiplicador de `iliquida`** —
    tratamento conservador declarado no pré-registro §3.1, mesma convenção do
    G3 do motor anterior: ausência de medida nunca vira benefício da dúvida.
    Um ticker fora da régua é, para efeito de custo, o pior caso conhecido.

    Mercado desconhecido levanta: cair num default silencioso aqui aplicaria a
    régua da B3 a cripto (ou vice-versa) sem ninguém perceber.
    """
    if uniforme:
        return MULTIPLICADOR_UNIFORME
    if mercado not in MULTIPLICADOR_SLIPPAGE:
        raise ValueError(
            f"multiplicador_da_faixa: mercado desconhecido: {mercado!r} — "
            f"use um de {sorted(MULTIPLICADOR_SLIPPAGE)}"
        )
    tabela = MULTIPLICADOR_SLIPPAGE[mercado]
    if faixa is None or faixa not in tabela:
        return tabela[FAIXA_ILIQUIDA]
    return tabela[faixa]


def custo_roundtrip_da_faixa(
    faixa: str | None,
    *,
    mercado: str,
    modelo_custo: B3CostModel | CryptoCostModel | None = None,
    uniforme: bool = False,
) -> float:
    """
    Custo round-trip (ida + volta) em fração do notional para um ativo da
    `faixa`. `modelo_custo=None` instancia o modelo default do `mercado`.

    Decompõe o custo em taxa (fixa) + slippage (escala) em vez de multiplicar
    o total: multiplicar o total inflaria também a corretagem, que não depende
    de liquidez — seria um custo inventado, e num estudo cujo veredito depende
    de sinal de excesso líquido, custo inventado é resultado inventado.
    """
    modelo = modelo_custo if modelo_custo is not None else _MODELO_PADRAO[mercado]()
    taxa_por_perna = float(modelo.costs_for(_NOTIONAL_REFERENCIA).total / _NOTIONAL_REFERENCIA)
    slippage_por_perna = float(modelo.slippage_bps) / 10_000.0
    mult = multiplicador_da_faixa(faixa, mercado=mercado, uniforme=uniforme)
    return 2.0 * (taxa_por_perna + slippage_por_perna * mult)


def custos_por_ticker(
    faixas: Mapping[str, str],
    *,
    mercado: str,
    modelo_custo: B3CostModel | CryptoCostModel | None = None,
    uniforme: bool = False,
) -> dict[str, float]:
    """
    `ticker -> custo round-trip`, a partir do mapa `ticker -> faixa` que
    `radar.evidence.categorias.calcular_faixas_liquidez` (B3) ou o próprio
    feature store (cripto) já produzem.

    Devolve só os tickers presentes em `faixas`; quem consultar um ticker
    ausente deve usar `custo_roundtrip_da_faixa(None, ...)` e receber o
    tratamento conservador, nunca um `KeyError` engolido com custo zero.
    """
    return {
        str(ticker): custo_roundtrip_da_faixa(
            faixa, mercado=mercado, modelo_custo=modelo_custo, uniforme=uniforme
        )
        for ticker, faixa in faixas.items()
    }

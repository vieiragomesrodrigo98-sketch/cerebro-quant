"""
src/radar/evidence/categorias.py — taxonomia de evidência (regex sobre o
Assunto do Fato Relevante) e classificação de liquidez por ticker.

Definição ÚNICA, usada por DOIS consumidores que precisam concordar
byte a byte:
  - `scripts/build_evidence_ledger.py` — MEDE cada célula (evidência ×
    horizonte × contexto de liquidez) contra 20 anos de preço.
  - `radar.agent.context` — CASA um candidato do motor às células já
    medidas, para ler o peso que o Ledger autorizou.

Duas cópias destas regras divergiriam sem ninguém perceber: o candidato
seria categorizado diferente de como o Ledger o mediu, e o agente leria o
peso de uma célula que não corresponde ao evento real — o mesmo erro de
classe que motivou `bar_repository.ler_barras` (definição única de "ler
barra") nesta mesma sessão.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRECOS_B3_PADRAO = _PROJECT_ROOT / "data" / "historical" / "b3" / "precos_b3.parquet"

# As evidências. Regex sobre o `Assunto` do fato relevante — cru de propósito:
# se o alfa aparece com um regex, um classificador só pode melhorá-lo. E se NÃO
# aparece, não era o classificador que estava faltando.
EVIDENCIAS: dict[str, str] = {
    "fr_recompra":    r"recompra|buyback|a[cç][aã]o em tesouraria",
    "fr_dividendo":   r"dividend|juros sobre|jcp|proventos|remunera",
    "fr_ma":          r"aquisi|incorpora|fus[aã]o|combina[cç][aã]o de neg|compra de",
    "fr_venda_ativo": r"aliena|venda de|desinvest|divest",
    "fr_guidance":    r"guidance|proje[cç]|estimativa|outlook",
    "fr_resultado":   r"resultado|balan[cç]o|lucro|ebitda|receita",
    "fr_contrato":    r"contrato|produ[cç]|opera[cç][aã]o|descoberta|po[cç]o|reserva",
    "fr_divida":      r"debentur|emiss[aã]o|capta[cç]|d[ií]vida|bond|nota promiss",
    "fr_judicial":    r"judicial|processo|multa|acordo|senten|arbitr|cade|aneel",
    "fr_governanca":  r"conselho|diretor|renun|eleiç|assemble|estatuto|ceo|cfo",
    "fr_qualquer":    r".",  # o baseline: comprar TODO fato relevante
}

CONTEXTO_NAO_OPERAVEL = "liq:iliquida"


def categorizar_assunto(assunto: str) -> list[str]:
    """
    Nomes de evidência (chaves de `EVIDENCIAS`, exceto o baseline
    `fr_qualquer`) cujo regex casa com o assunto. Pode devolver mais de um —
    "recompra e emissão de dívida" casa `fr_recompra` e `fr_divida`.
    """
    baixo = (assunto or "").lower()
    return [nome for nome, rx in EVIDENCIAS.items()
            if nome != "fr_qualquer" and re.search(rx, baixo)]


def calcular_faixas_liquidez(caminho_precos: Path = PRECOS_B3_PADRAO) -> dict[str, str]:
    """
    ticker -> "iliquida" | "media" | "liquida" — tercil de volume MEDIANO
    sobre o universo de tickers (nível ticker, não nível linha/evento).

    O tercil é calculado uma vez por TICKER, nunca ponderado por quantos
    eventos aquele ticker tem — computar o qcut sobre uma tabela já expandida
    por evento (uma linha por Fato Relevante) pesaria a distribuição pelo
    número de eventos de cada papel, e um emissor prolixo puxaria o corte de
    tercil para perto da própria liquidez. `medir()`/o Ledger têm que
    trabalhar com a MESMA régua que este candidato vai usar para se casar
    a uma célula — nível ticker, sempre.
    """
    df = pd.read_parquet(caminho_precos, columns=["ticker", "volume"])
    vol_por_ticker = df.groupby("ticker")["volume"].median()
    faixa = pd.qcut(vol_por_ticker, 3, labels=["iliquida", "media", "liquida"])
    return {str(tk): str(f) for tk, f in faixa.items()}

"""
src/radar/cerebro/operabilidade.py — operabilidade é RELAÇÃO, não atributo.

O que este módulo substitui, e por quê
--------------------------------------
Até 2026-08-04 o Cérebro herdava do motor antigo a ideia de **ativo ilíquido**:
`calcular_faixas_liquidez` (commit `c25e1e19`, 14/07, `fix(motor)`) corta o
universo em tercis de volume mediano, e `filtrar_universo_treino` **removia o
tercil de baixo do treino**. A própria docstring daquele filtro dizia que era
"a mesma doutrina de `build_evidence_ledger.py`" — ou seja, foi **herdada, não
decidida**. O ADR 0032 é explícito: nada anterior prevalece sobre o Cérebro.

Ordem do DEV (2026-08-04): *"não existe ativo ilíquido; existe não-operável a
depender do diagnóstico e da família, e operável. Categorizar qual família."*

Ele está certo, e o dado confirma. Medido no universo BRUTO, comparando o
movimento típico com o custo round-trip real de cada ativo:

| | universo | operáveis (`mov > 2× custo`) | dos "ilíquidos" do tercil |
|---|---|---|---|
| B3 | 162 | **162 (100%)** em h5/h10/h21 | **56 de 56 (100%)** |
| Cripto | 470 | 430–451 (91–96%) | 126–144 (80–92%) |

Na B3 o movimento típico em h5 é 3,24% contra custo mediano de 0,27% — **doze
vezes**. O tercil removia 35% do universo sem justificativa econômica, e
removia justamente onde a ineficiência de preço é mais provável.

O critério que fica no lugar
-----------------------------
    operavel(ativo, horizonte) ⟺ movimento_tipico(ativo, h) > margem × custo(ativo)

`movimento_tipico` é a **mediana do |retorno| futuro** daquele ativo naquele
horizonte — do próprio ativo, contra ele mesmo, nunca contra o universo.
Mediana e não média porque a média em cripto é dominada pela cauda: um ativo
que fez +900% uma vez teria "movimento típico" enorme e viraria operável por
um evento único.

**Isto é classificação, nunca filtro de existência.** O ativo não sai do
universo — ele recebe a lista de famílias que consegue sustentar. Um ativo caro
de negociar é inoperável em scalp (o custo come o movimento de minutos) e
perfeitamente operável em position (onde o movimento é ordens de grandeza
maior). Excluí-lo do APRENDIZADO por causa do scalp seria jogar fora o que ele
ensina sobre position.

O que este módulo NÃO faz
--------------------------
Não diz que o ativo é lucrativo. `mov > 2 × custo` diz que **existe movimento
suficiente para pagar o pedágio** — não que se consiga prever a direção dele.
Operabilidade é condição necessária, nunca suficiente, e confundir as duas
seria transformar "dá para operar" em "vale a pena operar".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

MARGEM_PADRAO: Final[float] = 2.0
"""O movimento típico precisa ser pelo menos 2× o custo round-trip.

Declarado, não otimizado. A razão de ser 2 e não 1: em `1×` o ativo empata com
o custo na mediana, ou seja, metade dos trades perde por construção antes de
qualquer questão de previsão. `2×` dá uma margem de um custo inteiro para o
erro de timing — que é o que sobra depois de o modelo escolher. Mudar este
valor é decisão de pré-registro, nunca ajuste de rodada."""


@dataclass(frozen=True)
class Operabilidade:
    """Por ativo: quanto ele se move, quanto custa, e em que famílias cabe."""

    ticker: str
    custo_roundtrip: float
    movimento_por_horizonte: dict[int, float]
    horizontes_operaveis: tuple[int, ...]

    @property
    def operavel_em_algum(self) -> bool:
        return bool(self.horizontes_operaveis)


def classificar_operabilidade(
    store: pd.DataFrame,
    custos_por_ticker: dict[str, float],
    *,
    horizontes: tuple[int, ...],
    margem: float = MARGEM_PADRAO,
    coluna_ticker: str = "ticker",
    prefixo_retorno: str = "ret_fwd_",
) -> dict[str, Operabilidade]:
    """
    `ticker -> Operabilidade`, para todo ativo do store.

    **Ninguém é excluído.** Um ativo sem nenhum horizonte operável aparece com
    `horizontes_operaveis=()` — visível e contável, em vez de sumir do
    universo sem deixar rastro. Essa diferença é o ponto: o filtro antigo
    apagava 56 tickers da B3 e nada no relatório dizia quais nem por quê.

    Ativo sem custo conhecido (`custos_por_ticker` não o tem) recebe
    `custo = NaN` e nenhum horizonte operável — tratamento conservador, mesma
    doutrina do custo: ausência de medida nunca vira benefício da dúvida.
    """
    if coluna_ticker not in store.columns:
        raise KeyError(f"classificar_operabilidade: coluna {coluna_ticker!r} ausente do store")

    faltantes = [h for h in horizontes if f"{prefixo_retorno}{h}" not in store.columns]
    if faltantes:
        raise KeyError(
            f"classificar_operabilidade: coluna(s) de retorno ausente(s) para "
            f"horizonte(s) {faltantes}"
        )

    grupos = store.groupby(coluna_ticker)
    movimentos = {
        h: grupos[f"{prefixo_retorno}{h}"].apply(lambda s: float(s.abs().median()))
        for h in horizontes
    }

    saida: dict[str, Operabilidade] = {}
    for ticker in grupos.groups:
        nome = str(ticker)
        custo = custos_por_ticker.get(nome, float("nan"))
        por_h = {h: float(movimentos[h].get(ticker, float("nan"))) for h in horizontes}
        operaveis = tuple(
            h
            for h in horizontes
            if np.isfinite(custo) and np.isfinite(por_h[h]) and por_h[h] > margem * custo
        )
        saida[nome] = Operabilidade(nome, custo, por_h, operaveis)
    return saida


def resumo_operabilidade(classificacao: dict[str, Operabilidade]) -> dict[str, object]:
    """Painel para o relatório: quantos ativos cabem em cada família, e quantos
    não cabem em nenhuma. O segundo número é o que o filtro antigo escondia."""
    if not classificacao:
        return {"n_ativos": 0, "por_horizonte": {}, "sem_nenhum_horizonte": 0}
    horizontes = sorted({h for o in classificacao.values() for h in o.movimento_por_horizonte})
    return {
        "n_ativos": len(classificacao),
        "por_horizonte": {
            h: sum(1 for o in classificacao.values() if h in o.horizontes_operaveis)
            for h in horizontes
        },
        "sem_nenhum_horizonte": sum(
            1 for o in classificacao.values() if not o.operavel_em_algum
        ),
        "tickers_sem_nenhum_horizonte": sorted(
            o.ticker for o in classificacao.values() if not o.operavel_em_algum
        )[:50],
    }

"""
src/radar/cerebro/ranking.py — o estágio RANKING do Cérebro v2.

Ranquear candidatos DENTRO do instante e operar uma fração do topo (ou do
fundo) é a pergunta que o produto de fato faz — "quais são os melhores agora?"
— e é a única forma de seleção que já produziu resultado verde neste projeto:
`short_bottom_10%_bear` e `short_bottom_20%_bear` (t_NW +2,83 e +2,63,
persistência 100%, sobrevivendo ao funding real medido).

Por que isto virou núcleo (e por que agora)
--------------------------------------------
Até 2026-08-04 esta função existia DUAS vezes, copiada literalmente entre
`scripts/mie_training/continuacao_short_bear_r1.py` e `scripts/mie_training/
replicacao_short_bear_h5_h10_r1.py` (e antes disso em `h_reversao_cripto_r1.
py`), sempre com o comentário "reexecução LITERAL de". O mecanismo por trás
das duas únicas células verdes do projeto morava em script de pesquisa, fora
do núcleo, em duplicata — exatamente o tipo de coisa que um redesenho perde
sem perceber. Promover é preservar.

Comportamento CONGELADO, byte a byte
-------------------------------------
A implementação abaixo é a mesma, sem "melhorias": mesmo `k = min(n, max(1,
round(n * frac)))`, mesmo `nsmallest`/`nlargest` (que resolvem empate pela
ordem de aparição), mesmo sorteio por `rng.choice(..., replace=False)`.
Inclusive o `round()` do Python — arredondamento bancário, que difere de
`int(n*frac + 0.5)` em `.5` exato. Nada disso é ideal em abstrato; tudo isso
é o que as células verdes mediram, e mudar qualquer um invalida a
reprodutibilidade delas. Uma variante melhor é uma FAMÍLIA NOVA, com
pré-registro próprio — nunca uma emenda silenciosa aqui.

**A única diferença, declarada para não inflar o "byte a byte":** a validação
de `modo` subiu para antes do laço. No original, um `modo` inválido só
levantava ao processar o primeiro instante — logo um pool VAZIO com um typo
(`"botom"`) devolvia vazio em silêncio em vez de levantar. Muda o
comportamento apenas para ENTRADA INVÁLIDA, nunca para uma rodada válida, e
portanto não toca nenhum número já medido.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

MODO_TOP: Final[str] = "top"
MODO_BOTTOM: Final[str] = "bottom"
MODO_ALEATORIO: Final[str] = "aleatorio"

MODOS_VALIDOS: Final[tuple[str, ...]] = (MODO_TOP, MODO_BOTTOM, MODO_ALEATORIO)
"""`"top"`/`"bottom"` são as duas pontas da mesma régua — e o `"aleatorio"`
NÃO é conveniência: é o CONTROLE da medição (mesma frequência de sinal, mesma
seção transversal, escolha cega). Uma leitura de `"top"` sem o `"aleatorio"`
ao lado não distingue seleção de sorte, e por isso os três moram juntos."""


def selecionar_percentil_instante(
    pool: pd.DataFrame,
    frac: float,
    modo: str,
    *,
    col_score: str = "prob",
    col_instante: str = "date",
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Seleciona, DENTRO de cada instante (`col_instante` — o dia no swing, a
    barra no scalp), a fração `frac` dos candidatos com maior (`"top"`), menor
    (`"bottom"`) ou aleatório (`"aleatorio"`, exige `rng`) valor de
    `col_score`.

    `k = min(n, max(1, round(n * frac)))` por instante: **pelo menos 1** (um
    instante nunca fica sem candidato só porque a seção transversal é pequena)
    e **nunca mais que n**. Este piso de 1 é uma decisão com efeito real e
    documentada aqui porque ela aparece na leitura: em instantes magros,
    `frac=10%` opera 1 de 3, que é 33% — a frequência efetiva de sinal não é a
    nominal, e quem lê a célula precisa saber disso.

    Devolve um frame NOVO (`ignore_index=True`), com as mesmas colunas de
    `pool`; `pool` vazio devolve um frame vazio com o mesmo schema (nunca
    levanta, nunca devolve `None` — um pool vazio é um estado legítimo do
    mercado, não um erro).

    `modo` inválido levanta `ValueError` em vez de cair num default: um typo
    em `"botom"` que silenciosamente virasse `"top"` inverteria o sinal da
    estratégia inteira.
    """
    if modo not in MODOS_VALIDOS:
        raise ValueError(
            f"selecionar_percentil_instante: modo desconhecido: {modo!r} — use um de {MODOS_VALIDOS}"
        )
    if modo == MODO_ALEATORIO and rng is None:
        raise ValueError(
            "selecionar_percentil_instante: modo='aleatorio' exige rng (semente explícita — "
            "controle sem semente não é reproduzível, logo não é controle)"
        )

    partes: list[pd.DataFrame] = []
    for _, grupo in pool.groupby(col_instante, sort=True):
        n = len(grupo)
        k = min(n, max(1, round(n * frac)))
        if modo == MODO_BOTTOM:
            escolhidos = grupo.nsmallest(k, col_score)
        elif modo == MODO_TOP:
            escolhidos = grupo.nlargest(k, col_score)
        else:  # MODO_ALEATORIO — `rng` já validado acima
            assert rng is not None
            posicoes = rng.choice(n, size=k, replace=False)
            escolhidos = grupo.iloc[posicoes]
        partes.append(escolhidos)

    vazio = pool.iloc[0:0]
    return pd.concat(partes, ignore_index=True) if partes else vazio


def construir_trades_topo(
    meta: pd.DataFrame,
    score: np.ndarray | pd.Series,
    *,
    frac: float,
    custos_por_ticker: dict[str, float],
    col_retorno: str,
    col_ticker: str = "ticker",
    col_instante: str = "date",
    modo: str = MODO_TOP,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Os trades que a régua Top-N do Cérebro v2 dispara, já com **excesso
    líquido por ativo**.

    Substitui, para o v2, o `radar.mie.metrics.construir_trades` — que dispara
    por limiar ABSOLUTO de probabilidade e desconta um custo UNIFORME. As duas
    coisas são justamente o que o ADR 0032 corrigiu: o limiar absoluto tem
    faixa alcançável que é artefato do fold, e o custo uniforme apaga a única
    diferença entre os dois braços de alvo do pré-registro.

    `custos_por_ticker` vem de `radar.cerebro.custos.custos_por_ticker`.
    **Ticker ausente do mapa vira `NaN` no excesso líquido, nunca custo zero**
    — e a linha permanece na saída, com o `NaN` visível, para que a leitura
    reporte quantos trades ficaram sem custo em vez de descartá-los em
    silêncio (descartar mudaria a frequência de sinal, que é justamente o que
    o controle aleatório precisa reproduzir).

    `modo="aleatorio"` (com `rng`) produz o **controle de mesma frequência**
    exigido pelo pré-registro §7: mesmo número de trades por instante, escolha
    cega. Sem ele, uma leitura de Top-N não distingue seleção de sorte.

    Devolve as linhas selecionadas de `meta` com duas colunas novas: `score` e
    `excesso_liquido`.
    """
    if len(meta) != len(score):
        raise ValueError(
            f"construir_trades_topo: meta ({len(meta)}) e score ({len(score)}) "
            "têm tamanhos diferentes"
        )
    duplicadas = meta.duplicated(subset=[col_ticker, col_instante])
    if duplicadas.any():
        raise ValueError(
            f"construir_trades_topo: {int(duplicadas.sum())} linha(s) duplicada(s) em "
            f"({col_ticker}, {col_instante}) — 1 trade por (ativo, instante) é o contrato"
        )

    pool = meta.copy()
    pool["score"] = np.asarray(score, dtype=float)

    escolhidos = selecionar_percentil_instante(
        pool, frac, modo, col_score="score", col_instante=col_instante, rng=rng
    )
    if escolhidos.empty:
        vazio = pool.iloc[0:0].copy()
        vazio["excesso_liquido"] = pd.Series(dtype=float)
        return vazio

    custo = pd.to_numeric(
        escolhidos[col_ticker].astype(object).map(custos_por_ticker), errors="coerce"
    )
    escolhidos = escolhidos.copy()
    escolhidos["custo_roundtrip"] = custo.to_numpy()
    escolhidos["excesso_liquido"] = (
        pd.to_numeric(escolhidos[col_retorno], errors="coerce").to_numpy() - custo.to_numpy()
    )
    return escolhidos.reset_index(drop=True)

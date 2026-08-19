"""
src/radar/cerebro/alvo.py — o alvo CROSS-SECTIONAL do Cérebro v2.

A mudança de pergunta, e por que ela é mecânica e não filosófica
-----------------------------------------------------------------
O motor anterior perguntava `1[exc_fwd_h > limiar]` — um limiar ABSOLUTO,
igual para todos os dias. A prevalência desse alvo varia enormemente entre
dias: num dia de enchente muitos ativos batem o limiar, num dia parado
nenhum. Resultado: **o eixo dominante de variância do alvo é o DIA**, e o
modelo aprende o dia (regime, breadth, calendário) em vez do ativo. Foi
exatamente o que aconteceu — a feature de maior *gain* do modelo em produção
era constante entre ativos.

Aqui a pergunta é: **este ativo está no topo do SEU dia?**

    y = 1[ rank_pct_dentro_do_instante(alvo) > fracao_topo ]

O operador é `>` **estrito**, e não `>=`. Não é preciosismo: `rank(pct=True)`
do pandas devolve `rank / n`, então num instante de 100 candidatos os postos
são `0,01 … 1,00` e `>= 0,90` marcaria **11** ativos, não 10 — a prevalência
sairia 11% e o alvo deixaria de ser o decil superior que o pré-registro
declara. Com `>`, sobram exatamente `0,91 … 1,00`. O pré-registro foi emendado
(antes da primeira rodada) para fixar o operador junto com a intenção.

A prevalência fica **fixa por construção** (10% com `fracao_topo=0,90`),
igual em todo instante. Isso remove a variação entre dias do alvo — não como
efeito colateral, mas como o ponto: o que sobra para o modelo explicar é
exclusivamente quem, DENTRO do dia, está acima de quem. Features constantes
no instante passam a ser inúteis por construção, que é o G-P1 do Contrato de
Pensamento realizado no alvo em vez de imposto por fora.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import pandas as pd

FRACAO_TOPO_PADRAO: Final[float] = 0.90
"""Decil superior — pré-registro `PRE_REGISTRO_CEREBRO_SWING_V1.md` §3."""

MINIMO_POR_INSTANTE_PADRAO: Final[int] = 20
"""Instantes com menos candidatos que isto ficam SEM rótulo (`NaN`), e as
linhas caem no `montar_dataset`. Razão: com 5 candidatos, "decil superior"
é 1 ativo — o rótulo passa a significar "o melhor de 5", que é outra
pergunta, com outra dificuldade e outra taxa-base efetiva. Misturar as duas
no mesmo treino contamina a amostra com um alvo que não é o declarado."""


def rotular_topo_do_instante(
    valor: pd.Series,
    chave_instante: pd.Series,
    *,
    fracao_topo: float = FRACAO_TOPO_PADRAO,
    minimo_por_instante: int = MINIMO_POR_INSTANTE_PADRAO,
) -> pd.Series:
    """
    `y = 1[rank_pct_dentro_do_instante(valor) > fracao_topo]`, `NaN` onde não
    é computável. Operador ESTRITO — ver docstring do módulo para o porquê
    (com `>=`, um instante de 100 candidatos marcaria 11, não 10).

    Devolve `NaN` (nunca 0) em três situações — e a distinção importa, porque
    `0` significaria "medi e este ativo NÃO está no topo", uma afirmação que
    nestes casos não se pode fazer:

      * `valor` ausente (`NaN`) — alvo não resolvido (horizonte além do fim do
        histórico, benchmark faltando naquela data);
      * instante com menos de `minimo_por_instante` candidatos VÁLIDOS;
      * instante em que todos os valores válidos são iguais — não há topo.

    O rank é `pct=True` com `method="average"` (default do pandas): empates
    recebem a média das posições, então um bloco de empatados nunca é partido
    arbitrariamente pela ordem das linhas. Consequência declarada: num
    instante com muitos empates, a fração rotulada `1` pode ficar ABAIXO de
    `1 - fracao_topo` — preferível a inventar uma ordem que o dado não tem.

    `valor` e `chave_instante` precisam compartilhar o índice; o resultado sai
    no mesmo índice de `valor`.
    """
    if len(valor) != len(chave_instante):
        raise ValueError(
            f"rotular_topo_do_instante: valor ({len(valor)}) e chave_instante "
            f"({len(chave_instante)}) têm tamanhos diferentes"
        )
    if not 0.0 < fracao_topo < 1.0:
        raise ValueError(
            f"rotular_topo_do_instante: fracao_topo={fracao_topo!r} fora de (0,1) — "
            "0 rotularia tudo como topo e 1 não rotularia nada"
        )

    df = pd.DataFrame({"valor": np.asarray(valor, dtype=float)})
    df["_instante"] = np.asarray(chave_instante)

    validos = df["valor"].notna()
    n_validos = validos.groupby(df["_instante"]).transform("sum")
    # nunique sobre os VÁLIDOS: um instante todo NaN mais um valor não conta
    # como "tem variação"
    distintos = (
        df.loc[validos].groupby("_instante")["valor"].transform("nunique")
        .reindex(df.index)
    )

    computavel = validos & (n_validos >= minimo_por_instante) & (distintos > 1)

    y = pd.Series(np.nan, index=range(len(df)), dtype=float)
    if computavel.any():
        rank = (
            df.loc[computavel]
            .groupby("_instante")["valor"]
            .rank(pct=True, method="average")
        )
        y.loc[computavel.to_numpy().nonzero()[0]] = (rank > fracao_topo).astype(float).to_numpy()

    y.index = pd.Index(valor.index)
    return y


def montar_valor_alvo_liquido(
    retorno_futuro: pd.Series, ticker: pd.Series, custos_por_ticker: dict[str, float]
) -> pd.Series:
    """
    **Braço A** do pré-registro §3: `retorno futuro − custo round-trip DO
    ATIVO`. É o único ponto em que os dois braços podem divergir — dentro de
    um mesmo instante, ranquear por excesso é idêntico a ranquear por retorno
    (o benchmark é constante ali), então é o custo por ativo que muda a ordem.

    Ticker ausente de `custos_por_ticker` recebe `NaN` — nunca custo zero.
    Custo zero seria a hipótese mais favorável possível para o ativo do qual
    menos se sabe, e é assim que um backtest passa a mentir a favor.
    """
    custo = pd.Series(np.asarray(ticker, dtype=object)).map(custos_por_ticker)
    custo = pd.to_numeric(custo, errors="coerce").to_numpy()
    return pd.Series(np.asarray(retorno_futuro, dtype=float) - custo, index=retorno_futuro.index)


def rotular_retorno_positivo(
    retorno_futuro: pd.Series, ticker: pd.Series, custos_por_ticker: dict[str, float]
) -> pd.Series:
    """
    **O ativo contra ELE MESMO** (ordem do DEV, 2026-08-04):

        y = 1[ ret_fwd(ativo) − custo(ativo) > 0 ]

    Nem contra um benchmark (`exc_fwd`), nem contra os outros ativos do dia
    (`rotular_topo_do_instante`). A pergunta é: **este ativo subiu o bastante
    para pagar o próprio pedágio?**

    Por que isto é mecanicamente melhor, medido e não opinado
    ---------------------------------------------------------
    O rótulo relativo (decil do dia) tem viés de volatilidade: um ativo
    volátil tem mais chance de cair no topo do dia sem ter retorno esperado
    melhor. Medido no cripto h5, por quintil de volatilidade:

    | | `P(topo do dia)` | `P(líquido > 0)` |
    |---|---|---|
    | vol baixa | 8,8% | 44,5% |
    | vol alta | **14,3%** | 42,0% |
    | **amplitude** | **0,0643** | **0,0304** |

    O alvo auto-referente **corta o viés pela metade**. E traz um segundo
    ganho independente: a prevalência sai de 10% para ~44%, eliminando o
    desequilíbrio de classe que exigia `scale_pos_weight ≈ 3,7` — que foi a
    causa mecânica do `num_trees=1` dos 9 modelos do v1.

    `NaN` onde o retorno ou o custo faltam — nunca `0`. `0` afirmaria "medi e
    este ativo NÃO pagou o pedágio", afirmação impossível sem os dois números.
    """
    liquido = montar_valor_alvo_liquido(retorno_futuro, ticker, custos_por_ticker)
    y = pd.Series(np.nan, index=liquido.index, dtype=float)
    valido = liquido.notna()
    y.loc[valido] = (liquido.loc[valido] > 0).astype(float)
    return y


def resumo_prevalencia(y: pd.Series, chave_instante: pd.Series) -> dict[str, Any]:
    """
    Retrato da prevalência do rótulo — a prova de que o alvo cross-sectional
    fez o que promete. `prevalencia_por_instante_desvio` perto de zero é a
    assinatura: o alvo deixou de variar entre instantes, que era a causa
    mecânica de o modelo aprender o dia.
    """
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "_instante": np.asarray(chave_instante)})
    validos = df.dropna(subset=["y"])
    if validos.empty:
        return {"n": 0, "prevalencia": float("nan"),
                "prevalencia_por_instante_media": float("nan"),
                "prevalencia_por_instante_desvio": float("nan"), "n_instantes": 0}
    por_instante = validos.groupby("_instante")["y"].mean()
    return {
        "n": len(validos),
        "prevalencia": float(validos["y"].mean()),
        "prevalencia_por_instante_media": float(por_instante.mean()),
        "prevalencia_por_instante_desvio": float(por_instante.std(ddof=0)),
        "n_instantes": len(por_instante),
    }

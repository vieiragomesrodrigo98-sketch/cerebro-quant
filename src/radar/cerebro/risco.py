"""
src/radar/cerebro/risco.py — curva de equity, drawdown e o que ele decide.

A lacuna que este módulo fecha
-------------------------------
`drawdown` era a única das quatro lacunas do Evidence Map que ficou **aberta e
declarada como tal**: os ledgers publicavam agregados por período, nunca a série
temporal, e sem a curva não há drawdown. Com os trades OOS persistidos
(`CEREBRO_LEDGER_TRADES_PERSISTIDOS01`) a curva passa a existir.

Por que `+1,364%/trade` não basta — a formulação do DEV
--------------------------------------------------------
    "Uma estratégia que ganha 1,3% por trade mas passa 18 meses em drawdown é
    completamente diferente de uma que entrega o mesmo retorno com drawdown
    controlado."

O retorno médio por trade é insensível à ORDEM em que os trades acontecem, e a
ordem é justamente o que determina se dá para operar aquilo com dinheiro real.
Duas séries com a mesma média podem ter drawdowns de 3% e de 40%.

⚠️ Esta métrica só REBAIXA, nunca promove
------------------------------------------
Declarado antes de medir, para não ser lido como decepção depois: drawdown é um
critério que hoje **não é avaliado**. Ao passar a ser, ele **adiciona uma
porta** — nenhuma hipótese vai de `CANDIDATA` para `VALIDADA` porque o drawdown
foi medido; algumas vão de `CANDIDATA` para `CANDIDATA` com um critério a mais
faltando.

Ele continua obrigatório: é portão de segurança antes de qualquer capital. Só
não é caminho de validação, e confundir as duas coisas gera expectativa errada
sobre o que esta medição entrega.

Ausência é `None`, e `None` REPROVA
------------------------------------
Quarta aplicação do mesmo princípio nesta linhagem de módulos: sem trades, sem
curva, sem drawdown ⇒ `None`. E o chamador trata `None` como **NÃO MEDIDA**, que
bloqueia — jamais como "passou". Esta sessão já corrigiu três vezes o mesmo bug
de ausência-imitando-aprovação (`bloqueio_dominante` devolvendo `NENHUM`,
`persistencia_de` pulando o critério, `avaliar_necessidade` devolvendo
`NENHUMA`); aqui ele nasce fechado.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd

DRAWDOWN_MAXIMO_TOLERADO: Final[float] = 0.25
"""
Queda máxima aceitável do pico da curva de equity, em fração do capital.

25% é o limite prático em que um operador individual costuma abandonar a
estratégia — e estratégia abandonada no fundo do poço realiza a perda sem
capturar a recuperação, o que torna o drawdown teórico irrelevante. O número é
uma **declaração de tolerância**, não uma medição, e por isso mora aqui visível
em vez de embutido numa comparação.
"""

CAPITAL_INICIAL: Final[float] = 1.0
"""Patrimônio no instante zero, antes do primeiro passo.

Existe nomeado porque aparece em dois lugares com papéis diferentes e é fácil
confundi-los: é o multiplicando do `cumprod` **e** o piso do pico. O piso é o que
impede uma estratégia que só perde de ter o próprio primeiro ponto (já abaixo de 1)
tomado como pico, o que apagaria a primeira perda do drawdown."""

MINIMO_PONTOS_CURVA: Final[int] = 30
"""Abaixo disto a curva não tem forma: o drawdown vira o pior de meia dúzia de
pontos, que é ruído. Devolve `None` (não medida), nunca um número frágil."""


@dataclass(frozen=True)
class Risco:
    n_pontos: int
    max_drawdown: float
    """Fração POSITIVA do PICO (0,18 = queda de 18% do patrimônio no pico).

    Calculada sobre a curva de PATRIMÔNIO composta, e o pico nunca é menor que
    `CAPITAL_INICIAL` — comparar retorno acumulado cru com uma tolerância
    percentual mistura grandezas e reprova todo mundo."""
    duracao_drawdown_dias: int
    """Do pico até o fundo."""
    recovery_dias: int | None
    """Do fundo de volta ao pico. `None` = **nunca recuperou** dentro da
    amostra — e `None` aqui é pior que um número grande, não melhor."""
    retorno_total: float
    calmar: float | None
    """`retorno_total / max_drawdown`. `None` quando o drawdown é zero (não há
    o que dividir) — não é "infinitamente bom", é "não computável"."""
    dentro_da_tolerancia: bool
    motivos: tuple[str, ...]

    patrimonio_valido: bool = True
    """
    **BLOQUEIO P0-B (ADR 0034).** `False` quando o patrimônio chegou a zero.

    Este campo MUDOU de papel quando `CEREBRO_EQUITY_HORIZONTE_SOBREPOSTO01`
    foi corrigido, e a mudança é o registro do conserto.

    **Antes:** a curva era `1 + cumsum(retorno)` e ficava negativa, produzindo
    drawdowns de 583%. Um drawdown de 550% não é estratégia catastrófica, é
    prova de que a curva não era de patrimônio — e o campo existia para pegar
    isso. Pegava 20 das 42 hipóteses do mapa em 2026-08-05.

    **Agora:** a curva compõe e o clamp de ruína vai no fator de cada passo,
    então ela não pode cruzar zero por construção. O campo virou uma
    **invariante**: só é `False` em RUÍNA de verdade (retorno de passo ≤ −100%)
    ou se um defeito novo aparecer. Continua aqui de propósito — retirar uma
    trava porque "agora não acontece" é como o próximo defeito passa calado.

    Quando é `False`, **`dentro_da_tolerancia` não tem significado econômico** e
    o consumidor trata o critério como NÃO MEDIDO, nunca como reprovação.
    """

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_pontos": self.n_pontos,
            "max_drawdown": self.max_drawdown,
            "duracao_drawdown_dias": self.duracao_drawdown_dias,
            "recovery_dias": self.recovery_dias,
            "retorno_total": self.retorno_total,
            "calmar": self.calmar,
            "dentro_da_tolerancia": self.dentro_da_tolerancia,
            "patrimonio_valido": self.patrimonio_valido,
            "motivos": list(self.motivos),
        }


def curva_de_equity(trades: pd.DataFrame) -> pd.Series | None:
    """
    Curva de PATRIMÔNIO de uma carteira alavancagem 1, capital inicial 1.

    Os dois defeitos que esta função teve, e por que corrigir um só era pior
    ---------------------------------------------------------------------------
    A versão anterior fazia `1 + cumsum(média do líquido por instante)`. Errava
    duas coisas ao mesmo tempo, e elas **se cancelavam parcialmente** — medido
    em `swing_b3_h10_A#base`: com os dois defeitos o equity final dava +19,9;
    corrigindo só a acumulação dava **+4.767.225**; corrigindo os dois, +6,4.
    Consertar metade era pior que não consertar.

    1. **Sobreposição de horizonte.** Com horizonte `h` e entrada a cada passo,
       `h` posições ficam abertas ao mesmo tempo. Somar o retorno TOTAL de cada
       uma no instante de entrada equivale a operar com alavancagem `h` — e
       infla a curva por um fator `h`. É o card
       `CEREBRO_EQUITY_HORIZONTE_SOBREPOSTO01`.
    2. **Acumulação aritmética.** `1 + cumsum(r)` fica NEGATIVO quando a soma
       passa de −1, e patrimônio não fica negativo: quem opera perde no máximo
       o que tem. Um patrimônio composto (`cumprod(1+r)`) tende a zero e nunca
       o cruza. Era este o defeito que produzia os drawdowns de 583%.

    Como a curva é montada agora
    -----------------------------
    A grade é o conjunto de instantes em que a estratégia AGE (as entradas
    distintas). Cada trade fica aberto de `date` até `exit_date`, e contribui
    com `liquido / L` por passo, onde `L` é quantos passos ele atravessa —
    espalhar o resultado total pela janela é o que impede contá-lo `L` vezes.

    O retorno da carteira no passo `t` é a **média** entre as posições abertas
    naquele passo, não a soma: capital 1 dividido igualmente entre o que está
    aberto é exatamente alavancagem 1. Daí `cumprod(1 + R_t)`.

    `exit_date` é OBRIGATÓRIO — e a ausência não é chutada
    ------------------------------------------------------
    Sem saber quando a posição fecha não há como saber quantas ficam abertas, e
    portanto não há curva de patrimônio. Inferir a janela do `horizonte` exigiria
    adivinhar a duração da barra, que **não é derivável do parquet**: no
    `swing_v1` a barra é o dia e a entrada é diária (`h` = `h` passos de grade),
    no `scalp_cripto_v2` a barra é de 1 minuto, o painel é amostrado de 60 em 60
    e `h=60` barras vale **1** passo de grade. Os dois trazem `horizonte == 60`
    e `horizonte == 10` sem nada que os distinga.

    Por isso quem produz o parquet declara `exit_date` (`trades_para_parquet`,
    que resolve a conversão com o `passo_amostragem` da `Familia`), e aqui a
    ausência devolve `None` = **NÃO MEDIDA**. É a mesma escolha do resto do
    módulo: ausência reprova, nunca aprova — e é preferível a um número que
    parece medido e está errado no sentido perigoso (a curva velha
    **subestimava** o drawdown de quem sobrevivia à validação: 34,4% publicado
    contra 86,7% real em `swing_cripto_h10_A#topo`).
    """
    if trades is None or trades.empty:
        return None
    if not {"date", "exit_date", "liquido"}.issubset(trades.columns):
        return None

    d = trades.loc[:, ["date", "exit_date", "liquido"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    d["exit_date"] = pd.to_datetime(d["exit_date"])
    d = d.dropna(subset=["date", "exit_date", "liquido"])
    if d.empty:
        return None

    grade = np.sort(d["date"].unique())
    if len(grade) < MINIMO_PONTOS_CURVA:
        return None

    # Índice de abertura e de fechamento de cada trade NA GRADE. `searchsorted`
    # à esquerda no fechamento: a posição rende ATÉ o passo anterior ao de saída.
    abre = np.searchsorted(grade, d["date"].to_numpy(), side="left")
    fecha = np.searchsorted(grade, d["exit_date"].to_numpy(), side="left")
    fecha = np.clip(fecha, abre + 1, len(grade))  # toda posição vive ≥ 1 passo
    passos = (fecha - abre).astype(float)

    # Diferenças acumuladas: O(n) em vez de varrer a janela de cada trade.
    # `soma` acumula a contribuição por passo; `abertas` conta quantas posições
    # estão vivas, que é o divisor do peso igual.
    soma = np.zeros(len(grade) + 1)
    abertas = np.zeros(len(grade) + 1)
    por_passo = d["liquido"].to_numpy(dtype=float) / passos
    np.add.at(soma, abre, por_passo)
    np.add.at(soma, fecha, -por_passo)
    np.add.at(abertas, abre, 1.0)
    np.add.at(abertas, fecha, -1.0)
    soma = np.cumsum(soma)[: len(grade)]
    abertas = np.cumsum(abertas)[: len(grade)]

    with np.errstate(divide="ignore", invalid="ignore"):
        retorno = np.where(abertas > 0, soma / abertas, 0.0)

    # Patrimônio composto. O clamp vai no FATOR de cada passo, não no produto
    # acumulado: um passo com retorno ≤ −100% zera a conta e ela fica zerada
    # para sempre, que é o que ruína significa. Clampar depois do `cumprod`
    # pareceria equivalente e não é — um fator negativo sobrevive multiplicando
    # e volta a inverter o sinal no passo seguinte, ressuscitando a carteira.
    fator = np.maximum(1.0 + retorno, 0.0)
    return pd.Series(CAPITAL_INICIAL * np.cumprod(fator), index=pd.DatetimeIndex(grade))


def avaliar(
    trades: pd.DataFrame,
    *,
    drawdown_maximo: float = DRAWDOWN_MAXIMO_TOLERADO,
) -> Risco | None:
    """
    Métricas de risco da curva. `None` = **não medida** (o chamador reprova).

    O `max_drawdown` é o de uma carteira de UMA estratégia, alavancagem 1,
    capital dividido igualmente entre as posições abertas. Não é o da carteira
    real do usuário, que combina estratégias e dimensiona posição — essa terá
    outro, e provavelmente menor, por diversificação.
    """
    curva = curva_de_equity(trades)
    if curva is None:
        return None

    # `curva_de_equity` já devolve PATRIMÔNIO (capital inicial 1), não retorno
    # acumulado. Antes ela devolvia `cumsum` de retorno e era aqui que o `1 +`
    # convertia — somar 1 de novo contaria o capital inicial duas vezes e
    # amassaria o drawdown para perto da metade.
    valores = curva.to_numpy(dtype=float)
    # O pico nunca é menor que o CAPITAL INICIAL. A curva começa depois do
    # primeiro passo, então uma estratégia que só perde tem seu primeiro ponto
    # já abaixo de 1 — e tomar esse ponto como pico apaga a primeira perda do
    # drawdown. Medido: 40 perdas de 5% saíam com 86,47% em vez de 87,15%,
    # porque o pico virava 0,95 em vez de 1,00.
    pico = np.maximum(np.maximum.accumulate(valores), CAPITAL_INICIAL)
    with np.errstate(divide="ignore", invalid="ignore"):
        queda = np.where(pico > 0, (pico - valores) / pico, 0.0)
    i_fundo = int(np.argmax(queda))
    max_dd = float(queda[i_fundo])

    # O pico que originou o fundo: o último índice, até o fundo, em que a curva
    # tocou o máximo acumulado.
    i_pico = int(np.argmax(valores[: i_fundo + 1])) if i_fundo > 0 else 0

    datas = curva.index
    duracao = int((datas[i_fundo] - datas[i_pico]).days) if i_fundo > i_pico else 0

    # Recuperação: primeiro ponto APÓS o fundo que volta ao nível do pico.
    recovery: int | None = None
    depois = valores[i_fundo:]
    alcanca = np.flatnonzero(depois >= valores[i_pico])
    if alcanca.size:
        recovery = int((datas[i_fundo + int(alcanca[0])] - datas[i_fundo]).days)

    retorno_total = float(valores[-1] - CAPITAL_INICIAL)
    calmar = None if max_dd <= 0 else float(retorno_total / max_dd)

    # BLOQUEIO P0-B (ADR 0034): patrimônio não fica negativo. Se ficou, a curva
    # não é de patrimônio, e nenhum critério derivado dela vale.
    #
    # A trava CONTINUA depois de `CEREBRO_EQUITY_HORIZONTE_SOBREPOSTO01`, e não
    # é decoração: a curva composta não pode mais ficar negativa por construção,
    # então esta condição virou uma INVARIANTE. Se ela disparar de novo, o
    # defeito é novo — e a alternativa (remover a trava porque "agora não
    # acontece") é a forma exata de deixar o próximo passar em silêncio.
    patrimonio_valido = bool(valores.min() > 0.0)

    motivos: list[str] = []
    if not patrimonio_valido:
        motivos.append(
            f"curva NÃO representa patrimônio — equity chegou a {valores.min():+.4f}. "
            "A curva é composta e não pode cruzar zero por construção, então isto é "
            "RUÍNA medida (retorno de passo ≤ −100%) ou defeito novo. Critério de "
            "risco NÃO MEDIDO"
        )
    if max_dd > drawdown_maximo:
        motivos.append(
            f"drawdown máximo {max_dd:.1%} > tolerância {drawdown_maximo:.0%}"
        )
    if recovery is None and max_dd > 0:
        motivos.append(
            f"nunca recuperou o pico dentro da amostra ({duracao} dias em queda)"
        )

    return Risco(
        n_pontos=len(valores),
        max_drawdown=max_dd,
        duracao_drawdown_dias=duracao,
        recovery_dias=recovery,
        retorno_total=retorno_total,
        calmar=calmar,
        dentro_da_tolerancia=not motivos,
        motivos=tuple(motivos),
        patrimonio_valido=patrimonio_valido,
    )

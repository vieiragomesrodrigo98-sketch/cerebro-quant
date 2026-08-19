"""
src/radar/cerebro/alocacao.py — quantas posições, e quanto em cada.

O que este módulo resolve, e o número que o obriga a existir
-------------------------------------------------------------
`RiskManager` limita **contagem** de posições (5) e exposição (80%). Medido nos
20 pares de cripto sobre 835 dias:

```
correlação média entre pares    0,627
correlação média com o BTC      0,665
1º componente principal         66% da variância
os 20 ativos equivalem a        ~2,2 apostas independentes
```

Com `ρ = 0,627`, **"máximo 4 posições" entrega 1,4 aposta independente**. Acha-se
que diversificou em 4 e tem-se 1,4× a mesma aposta em BTC, alavancada. O teto por
CONTAGEM mede a coisa errada — e foi assim que uma posição consumiu a carteira
inteira na S165 enquanto "só" havia uma.

Este módulo é puro: não lê banco, não emite ordem, não escolhe ativo. Recebe
correlação e probabilidades, devolve quantas posições cabem e o peso de cada uma.
"""

from __future__ import annotations

from typing import Final

import numpy as np

__all__ = [
    "CONFIANCA_MAXIMA",
    "CONFIANCA_MINIMA",
    "multiplicador_por_confianca",
    "numero_efetivo_de_apostas",
    "posicoes_para_apostas_efetivas",
]


def numero_efetivo_de_apostas(correlacao: np.ndarray | float, n: int | None = None) -> float:
    """
    Quantas apostas **independentes** existem em `n` posições correlacionadas.

    Duas formas, e as duas medem a mesma coisa:

    * `correlacao` escalar + `n` → fórmula fechada `n / (1 + (n-1)ρ)`, a
      aproximação de correlação uniforme;
    * `correlacao` como matriz → **razão de participação** dos autovalores,
      `(Σλ)² / Σλ²`, que não assume uniformidade.

    Com `ρ = 0` devolve `n` (tudo independente); com `ρ = 1` devolve 1 (é tudo a
    mesma aposta). É a métrica sobre a qual o teto de carteira deve ser
    declarado — não a contagem.
    """
    if isinstance(correlacao, np.ndarray) and correlacao.ndim == 2:
        autovalores = np.linalg.eigvalsh(correlacao)
        autovalores = autovalores[autovalores > 0]
        if autovalores.size == 0:
            return 0.0
        return float(autovalores.sum() ** 2 / (autovalores**2).sum())

    if n is None:
        raise ValueError("com correlação escalar, `n` é obrigatório")
    rho = float(correlacao)
    if n <= 0:
        return 0.0
    denominador = 1.0 + (n - 1) * rho
    if denominador <= 0:
        return float(n)
    return float(n / denominador)


def posicoes_para_apostas_efetivas(
    apostas_desejadas: float, correlacao: float, *, teto: int = 50
) -> int:
    """
    Quantas posições são necessárias para chegar a `apostas_desejadas`.

    O inverso da conta acima, e o que o Portfolio Engine de fato pergunta:
    *"quero 3 apostas independentes; com correlação de 0,63, quantas posições
    isso exige?"*

    **Pode ser impossível** — com `ρ` alto existe um limite assintótico
    (`1/ρ`), e pedir mais que isso devolve o `teto` em vez de crescer para
    sempre. Devolver o teto em silêncio seria pior que devolver um número
    grande: o chamador precisa perceber que atingiu o limite estrutural do
    universo, não do parâmetro.
    """
    if apostas_desejadas <= 1:
        return 1
    for n in range(1, teto + 1):
        if numero_efetivo_de_apostas(correlacao, n) >= apostas_desejadas:
            return n
    return teto


# ── Sizing por confiança ───────────────────────────────────────────────────
#
# MEDIDO em 1.468.114 predições OOS de cripto, por quintil de probabilidade
# calibrada:
#
#   quintil   prob média   P(y=1) real   excesso (sem o 1% extremo)
#   Q1          0,1380        16,5%           −2,398%
#   Q5          0,3232        27,4%           −0,290%
#
# A amplitude de acerto é **+10,9 pontos**, monotônica, e é uma proporção —
# imune aos outliers que corrompem a média. O excesso winsorizado melhora
# monotonicamente do Q2 ao Q5. **A probabilidade discrimina**, e é isso que
# autoriza este bloco a existir.
#
# E o que ele NÃO faz: criar edge. O Q5 ainda perde (−0,290%). Sizing melhora a
# MISTURA — concentra capital onde o modelo acerta mais — e não transforma
# negativo em positivo.

CONFIANCA_MINIMA: Final[float] = 0.5
CONFIANCA_MAXIMA: Final[float] = 1.0
"""Faixa do multiplicador. O piso não é zero de propósito: **posição pequena
demais é dominada pelo custo fixo**, e a decisão de não operar é do gate, não do
sizing — um multiplicador que zera esconderia uma recusa dentro de um cálculo de
tamanho."""


def multiplicador_por_confianca(
    prob: float, *, piso: float, teto: float,
) -> float:
    """
    Multiplicador de tamanho a partir da probabilidade calibrada.

    `piso` e `teto` são a faixa **observada no fold** — nunca 0 e 1. A faixa
    alcançável de uma probabilidade calibrada é **artefato do fold** (foi assim
    que o emissor v1 ficou mudo um mês, comparando 0,60 com um calibrador que
    saturava em 0,187): normalizar por 0-1 daria multiplicador quase constante
    quando o calibrador satura, e o sizing perderia a informação que existe.

    Linear entre `CONFIANCA_MINIMA` e `CONFIANCA_MAXIMA` porque a relação medida
    entre quintil e acerto é aproximadamente linear (16,5% → 27,4% em cinco
    passos regulares). Curva mais agressiva concentraria em cima de uma forma
    que os dados não sustentam.
    """
    if teto <= piso:
        return CONFIANCA_MAXIMA
    pos = (float(prob) - piso) / (teto - piso)
    pos = min(1.0, max(0.0, pos))
    return CONFIANCA_MINIMA + pos * (CONFIANCA_MAXIMA - CONFIANCA_MINIMA)

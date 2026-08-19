"""
src/radar/cerebro/contrato.py — o **Contrato de Pensamento** (ADR 0032).

Por que este módulo existe (a autópsia que o originou)
------------------------------------------------------
O MIE v1 esteve em produção como emissor de 2026-07-29 a 2026-08-04 e **nunca
emitiu um sinal** — não por seletividade, por impossibilidade aritmética. Três
defeitos encadeados, todos medidos:

1. **Não discriminava ativo.** Das 45 colunas de entrada, 7 eram CONSTANTES
   dentro do dia (`breadth_sma50`, `ibov_dist_ema50`, os 3 de calendário e as
   2 categóricas de regime) — e a de maior *gain* no modelo em produção era
   uma delas. O modelo respondia o mesmo número para todos os ativos do dia:
   12 valores distintos em 283 pares avaliados. Ele enxergava o *clima*, não
   os *jogadores*.
2. **A saída era quase constante e ninguém mediu isso.** Nenhuma etapa do
   pipeline perguntava "a saída varia?"; a mudez atravessou o go-live, uma
   janela de Marco 0 e três leituras oficiais.
3. **O limiar era inalcançável.** O gate de emissão (0,55/0,60) era comparado
   a uma probabilidade calibrada cujo teto — determinado pelo bloco mais alto
   da isotônica — era 0,187 a 0,365. Gate acima do teto ⇒ abstenção eterna,
   reportada como saúde.

E o quarto, que só aparece quando se olha o sistema inteiro: sem vocabulário
de contexto comum, cada motor descobre sua zona de validade num formato
próprio e o **Strategy Router** (Fase 6 do Plano de Reconstrução) não tem o
que orquestrar.

Os 4 portões
-------------
Cada portão é uma função PURA (zero I/O, zero `settings`, zero rede) que
levanta uma exceção própria ou devolve um diagnóstico. São mecanicamente
verificáveis por `pytest` — não são revisão humana, não são convenção de
nome. Juntos, teriam pego 100% do desastre acima.

  **G-P1 · Separação seleção × condição** (`validar_espaco_de_selecao`)
      Nenhuma coluna constante dentro do INSTANTE entra no espaço de features
      de *seleção*. Contexto (regime, breadth, calendário) tem outro papel:
      entra como **partição pré-registrada** (ADR 0031 §4), nunca como
      feature competindo com features de ativo. "Instante" é o dia no swing,
      a barra no scalp — por isso a chave é sempre um argumento.

  **G-P2 · A saída tem que discriminar** (`diagnosticar_saida`)
      Saída quase constante é **emissor mudo**, e mudez é FALHA, nunca
      abstenção. Este portão roda tanto no treino quanto em produção, a cada
      rodada, porque é barato e porque foi exatamente o que faltou.

  **G-P3 · O limiar tem que ser alcançável** (`validar_limiar_alcancavel`)
      Todo limiar de emissão é comparado ao TETO alcançável da escala em que
      ele se expressa, no carregamento — antes de decidir qualquer coisa.
      Limiar acima do teto levanta; nunca vira abstenção silenciosa.

  **G-P4 · Vocabulário de contexto comum** (`validar_contextos`)
      Toda partição condicional usa `DIMENSOES_CONTEXTO`. É o que permite
      comparar o mapa de validade de um especialista com o de outro — e,
      depois, construir o Router.

O que este módulo deliberadamente NÃO faz
------------------------------------------
Não computa Rank IC: recebe o número pronto (`rank_ic_medio`). A estatística
vive em quem já a tem (`radar.mie.metrics.rank_ic_diario`, sobre
`radar.evidence.ledger.t_newey_west`) e o contrato não pode depender de um
motor. Também não define formato novo de mapa de validade: o mapa É uma lista
de `radar.evidence.ledger.Celula` variando `contexto` — a máquina de medição
(7 facas, FDR, walk-forward) fica intocada, como manda o ADR 0031.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd

# ── Vocabulário de contexto (G-P4) ──────────────────────────────────────────

DIMENSOES_CONTEXTO: Final[tuple[str, ...]] = (
    "regime",
    "volatilidade",
    "liquidez",
    "horario",
    "ativo",
    "horizonte",
    "mercado",
)
"""As dimensões pelas quais um especialista pode declarar sua zona de validade
(Plano de Reconstrução §3/§7 — `Modelo × Contexto × Mercado × Horizonte ×
Ativo × Execução`). Fechada de propósito: cada dimensão nova multiplica a
grade condicional e, com ela, o risco de alpha falso — acrescentar uma é
decisão de ADR, com o denominador do FDR recalculado (ADR 0031 §4, controle
2). O `contexto` de uma `radar.evidence.ledger.Celula` deve nomear uma destas
(ex.: `"regime:bear"`, `"liquidez:alta"`) para que o mapa de validade de um
motor seja comparável ao de outro."""

SEPARADOR_CONTEXTO: Final[str] = ":"
"""`"<dimensao>:<valor>"` — a forma canônica de um rótulo de contexto. O
`"global"` (sem separador) é o único rótulo aceito fora dessa forma: nomeia a
célula NÃO condicionada, que precisa existir em toda leitura para o alpha
condicional ser lido contra ela, nunca no lugar dela."""

CONTEXTO_GLOBAL: Final[str] = "global"

MINIMO_VALORES_DISTINTOS: Final[int] = 20
"""Piso de valores distintos na saída, abaixo do qual ela é declarada MUDA
(G-P2). Referência medida, não arbitrada: o MIE v1 produzia 12 valores
distintos no universo INTEIRO (283 pares × 5 dias) porque uma árvore com 30
splits tem no máximo 31 folhas e a isotônica colapsava isso em 2 a 6 blocos.
Qualquer motor que discrimine ativo de verdade passa disto com folga; quem
não passa não está sendo seletivo, está mudo."""

MINIMA_FRACAO_INSTANTES_QUE_VARIAM: Final[float] = 0.90
"""Fração mínima de instantes (dias/barras) em que a saída precisa variar
ENTRE os candidatos daquele instante (G-P2). Um motor pode legitimamente não
ter opinião num dia; não pode deixar de ter opinião em quase todos. Instantes
com um único candidato não entram na conta — não há o que variar ali."""


class ContratoDePensamentoError(Exception):
    """Base de todas as violações — permite `except ContratoDePensamentoError`
    num orquestrador sem enumerar os 4 portões."""


class InstrumentoNaoAprendeuError(ContratoDePensamentoError):
    """G-P0: o modelo treinado é degenerado — não tem complexidade suficiente
    para ter aprendido coisa alguma."""


class EspacoDeSelecaoInvalidoError(ContratoDePensamentoError):
    """G-P1: há coluna constante no instante no espaço de features de seleção."""


class EmissorMudoError(ContratoDePensamentoError):
    """G-P2: a saída não discrimina — é (quase) a mesma para todo candidato."""


class LimiarInalcancavelError(ContratoDePensamentoError):
    """G-P3: o limiar de emissão está acima do teto alcançável da escala."""


class ContextoForaDoVocabularioError(ContratoDePensamentoError):
    """G-P4: rótulo de contexto que não nomeia uma `DIMENSOES_CONTEXTO`."""


# ── G-P0 · O instrumento tem que aprender ───────────────────────────────────

MINIMO_UNIDADES_APRENDIDAS: Final[int] = 10
"""Piso de complexidade de um modelo treinado — árvores num GBM, regras num
catálogo. Medido, não arbitrado: os 9 modelos do MIE v1 em produção tinham
`num_trees=1` com `n_estimators=400` congelado no pré-registro. Uma árvore não
é boosting; o `save_model()` do LightGBM persiste até `best_iteration`, e o
early stopping elegia a iteração 1 porque a `binary_logloss` do holdout
(não-ponderado) piorava já na primeira, com `scale_pos_weight≈3,7`. O piso é
baixo de propósito — não mede qualidade, mede se o treino DEGENEROU. Um modelo
que não passa daqui não é um modelo ruim: é um modelo que não existe."""


def validar_instrumento_aprendeu(
    n_unidades: int,
    *,
    motor: str,
    unidade: str = "árvore(s)",
    minimo: int = MINIMO_UNIDADES_APRENDIDAS,
) -> None:
    """
    G-P0 — o primeiro portão, e o único que roda ANTES de qualquer medição.
    Levanta `InstrumentoNaoAprendeuError` se o modelo treinado for degenerado.

    Genérico por número de propósito: quem conta as unidades é o motor (o
    LightGBM tem `radar.mie.model.contar_arvores`; um catálogo de regras conta
    regras ativas), este contrato só sabe que abaixo de um piso não há
    instrumento. É o critério 1 dos 5 do Go/No-Go da Fase 1 do Plano de
    Reconstrução ("modelo aprende"), na forma que ABORTA.

    Não confundir com qualidade: passar aqui não diz nada sobre haver edge —
    diz apenas que a medição seguinte vai medir um modelo, e não um toco.
    Foi por não ter este portão que três leituras oficiais, uma janela de
    Marco 0 e uma emenda de ADR foram construídas sobre uma árvore só.
    """
    if n_unidades < minimo:
        raise InstrumentoNaoAprendeuError(
            f"{motor}: instrumento DEGENERADO — {n_unidades} {unidade} (mínimo {minimo}). "
            f"Um modelo com esta complexidade não aprendeu nada, e qualquer leitura feita "
            f"sobre ele mede o defeito, não o mercado (G-P0 do Contrato de Pensamento, "
            f"ADR 0032). Foi assim que os 9 modelos do MIE v1 foram para produção com "
            f"num_trees=1 e n_estimators=400 no pré-registro."
        )


# ── G-P1 · Separação seleção × condição ─────────────────────────────────────


def colunas_constantes_no_instante(
    df: pd.DataFrame, colunas: Sequence[str], *, chave_instante: str
) -> tuple[str, ...]:
    """
    Quais de `colunas` são constantes DENTRO de cada instante (`chave_instante`
    — `date` no swing, o timestamp da barra no scalp) — isto é, não carregam
    nenhuma informação capaz de separar um ativo de outro no mesmo momento.

    Uma coluna é constante no instante quando, em TODO instante com 2+ linhas,
    ela tem no máximo 1 valor distinto — contando `NaN` COMO um valor
    (`nunique(dropna=False)`, não o default do pandas). A escolha é
    deliberada e a razão é operacional: o LightGBM roteia o ausente para um
    lado da árvore, então uma coluna que é `NaN` para metade dos ativos e
    `0,3` para a outra metade **de fato separa** os dois grupos naquele
    instante — absolvê-la como "constante" esconderia exatamente o tipo de
    separação silenciosa que este portão existe para revelar. Uma coluna
    inteiramente `NaN` continua constante (um único valor: a ausência).

    Instantes com uma única linha são IGNORADOS: ali toda coluna é
    trivialmente constante, e contá-los faria uma feature legítima ser
    reprovada só porque o universo teve um dia magro. Devolve `()` se `df`
    estiver vazio ou não sobrar nenhum instante com 2+ linhas — nesse caso não
    há evidência para acusar ninguém, e acusar sem evidência seria o mesmo
    erro que este módulo existe para impedir.

    Coluna ausente de `df` levanta `KeyError` (nunca é silenciosamente
    ignorada: um typo no nome da feature reprovaria o portão sem que ninguém
    percebesse — foi assim que features morreram por typo neste projeto).
    """
    faltantes = [c for c in colunas if c not in df.columns]
    if faltantes:
        raise KeyError(
            f"colunas_constantes_no_instante: coluna(s) ausente(s) do DataFrame: {faltantes}"
        )
    if chave_instante not in df.columns:
        raise KeyError(
            f"colunas_constantes_no_instante: chave de instante ausente: {chave_instante!r}"
        )
    if df.empty or not colunas:
        return ()

    tamanho = df.groupby(chave_instante)[chave_instante].transform("size")
    com_secao = df.loc[tamanho >= 2]
    if com_secao.empty:
        return ()

    grupos = com_secao.groupby(chave_instante)
    constantes = [c for c in colunas if int(grupos[c].nunique(dropna=False).max() or 0) <= 1]
    return tuple(constantes)


def validar_espaco_de_selecao(
    df: pd.DataFrame, colunas_selecao: Sequence[str], *, chave_instante: str, motor: str
) -> None:
    """
    G-P1. Levanta `EspacoDeSelecaoInvalidoError` se qualquer coluna de
    `colunas_selecao` for constante no instante — com os nomes na mensagem,
    porque o conserto é sempre "mova estas colunas para a partição de
    contexto", nunca "investigue".

    `motor` entra na mensagem (ex.: `"mie_b3_h21"`, `"scalp_cripto_v1"`) para
    que o erro diga de quem é o espaço reprovado sem quem lê precisar
    reconstruir o stack.
    """
    constantes = colunas_constantes_no_instante(df, colunas_selecao, chave_instante=chave_instante)
    if constantes:
        raise EspacoDeSelecaoInvalidoError(
            f"{motor}: {len(constantes)} coluna(s) do espaço de SELEÇÃO são constantes dentro do "
            f"instante ({chave_instante}) e não podem separar um ativo de outro: "
            f"{list(constantes)}. Contexto entra como PARTIÇÃO pré-registrada "
            f"(DIMENSOES_CONTEXTO), nunca como feature de seleção — ver G-P1 do Contrato de "
            f"Pensamento (ADR 0032). Foi este defeito que fez o MIE v1 responder o mesmo número "
            f"para todos os ativos do dia."
        )


# ── G-P2 · A saída tem que discriminar ──────────────────────────────────────


@dataclass(frozen=True)
class DiagnosticoSaida:
    """Retrato da saída de um motor num conjunto de candidatos.

    `mudo` é a decisão do portão; os demais campos existem para que a decisão
    seja auditável sem re-rodar nada. `rank_ic_medio` é `None` quando o alvo
    ainda não é conhecido (produção, dia 0) — a ausência é um estado legítimo
    e declarado, nunca um zero fabricado.
    """

    n_total: int
    n_valores_distintos: int
    fracao_instantes_que_variam: float
    n_instantes_avaliados: int
    rank_ic_medio: float | None
    mudo: bool
    motivo: str


MOTIVO_SAIDA_OK: Final[str] = "discrimina"
MOTIVO_SAIDA_VAZIA: Final[str] = "sem_candidatos"
MOTIVO_POUCOS_VALORES: Final[str] = "poucos_valores_distintos"
MOTIVO_CONSTANTE_NO_INSTANTE: Final[str] = "constante_dentro_do_instante"
MOTIVO_RANK_IC_INVALIDO: Final[str] = "rank_ic_nao_computavel"


def diagnosticar_saida(
    predicao: Iterable[float],
    *,
    chave_instante: Iterable[Any],
    rank_ic_medio: float | None = None,
    minimo_valores_distintos: int = MINIMO_VALORES_DISTINTOS,
    minima_fracao_instantes_que_variam: float = MINIMA_FRACAO_INSTANTES_QUE_VARIAM,
) -> DiagnosticoSaida:
    """
    G-P2. Retrato + veredito da saída de um motor. NÃO levanta — devolve o
    diagnóstico; quem decide o que fazer com `mudo=True` é o chamador (treino
    aborta, scanner alerta e não emite). Separar medir de agir é o que permite
    o mesmo portão servir a produção e a pesquisa.

    Dois critérios independentes, ambos necessários:
      * **variedade global** — `n_valores_distintos >= minimo_valores_distintos`
        (o MIE v1 tinha 12 no universo inteiro);
      * **variedade no instante** — a saída varia entre candidatos do mesmo
        instante em pelo menos `minima_fracao_instantes_que_variam` deles.
        É o critério que pega o caso patológico exato do v1: um número que
        muda ao longo do tempo (o regime) e é idêntico entre ativos do dia.

    `NaN` na predição é tratado como valor inválido: entra em `n_total` mas
    nunca conta como um valor distinto nem como variação dentro do instante —
    uma saída toda `NaN` é muda, não "variada".

    `rank_ic_medio` é apenas TRANSPORTADO para o diagnóstico (ver docstring do
    módulo: este contrato não computa estatística). Quando informado, um Rank
    IC não-finito também derruba o portão — se a correlação nem é computável,
    a saída não está ordenando nada.

    O piso de variedade global é `min(minimo_valores_distintos, n_total)`:
    nenhuma saída pode ter mais valores distintos do que candidatos, e exigir
    o impossível reprovaria por aritmética um universo pequeno (uma rodada com
    5 candidatos, um teste com 2) — que é o MESMO erro de comparar um número
    fora da escala dele que o G-P3 existe para impedir. Com o universo real do
    MIE (283 pares cripto, 163 tickers B3) o piso é o valor cheio.
    """
    pred = np.asarray(list(predicao), dtype=float)
    instantes = pd.Series(list(chave_instante))
    if len(pred) != len(instantes):
        raise ValueError(
            f"diagnosticar_saida: predicao ({len(pred)}) e chave_instante "
            f"({len(instantes)}) têm tamanhos diferentes"
        )

    n_total = len(pred)
    if n_total == 0:
        return DiagnosticoSaida(0, 0, 0.0, 0, rank_ic_medio, True, MOTIVO_SAIDA_VAZIA)

    finitos = pred[np.isfinite(pred)]
    n_distintos = len(np.unique(finitos))

    df = pd.DataFrame({"pred": pred, "_instante": instantes.to_numpy()})
    tamanho = df.groupby("_instante")["pred"].transform("size")
    com_secao = df.loc[tamanho >= 2]
    if com_secao.empty:
        n_instantes = 0
        fracao_varia = 0.0
    else:
        distintos_por_instante = com_secao.groupby("_instante")["pred"].nunique()
        n_instantes = len(distintos_por_instante)
        fracao_varia = float((distintos_por_instante >= 2).mean())

    piso_variedade = min(minimo_valores_distintos, n_total)
    if n_distintos < piso_variedade:
        motivo = MOTIVO_POUCOS_VALORES
    elif n_instantes > 0 and fracao_varia < minima_fracao_instantes_que_variam:
        motivo = MOTIVO_CONSTANTE_NO_INSTANTE
    elif rank_ic_medio is not None and not math.isfinite(rank_ic_medio):
        motivo = MOTIVO_RANK_IC_INVALIDO
    else:
        motivo = MOTIVO_SAIDA_OK

    return DiagnosticoSaida(
        n_total=n_total,
        n_valores_distintos=n_distintos,
        fracao_instantes_que_variam=fracao_varia,
        n_instantes_avaliados=n_instantes,
        rank_ic_medio=rank_ic_medio,
        mudo=motivo != MOTIVO_SAIDA_OK,
        motivo=motivo,
    )


def exigir_saida_que_discrimina(diagnostico: DiagnosticoSaida, *, motor: str) -> None:
    """G-P2 na forma que ABORTA (treino, promoção): levanta `EmissorMudoError` se
    `diagnostico.mudo`. Existe separada de `diagnosticar_saida` porque
    produção precisa medir e alertar sem derrubar a rodada, e treino precisa
    derrubar — mesmo diagnóstico, decisões diferentes."""
    if not diagnostico.mudo:
        return
    raise EmissorMudoError(
        f"{motor}: saída MUDA ({diagnostico.motivo}) — {diagnostico.n_valores_distintos} valor(es) "
        f"distinto(s) em {diagnostico.n_total} candidato(s); varia em "
        f"{diagnostico.fracao_instantes_que_variam:.0%} dos {diagnostico.n_instantes_avaliados} "
        f"instante(s) com seção transversal. Mudez é FALHA, nunca abstenção (G-P2 do Contrato de "
        f"Pensamento, ADR 0032): um emissor que não discrimina não está sendo seletivo."
    )


# ── G-P3 · O limiar tem que ser alcançável ──────────────────────────────────


def validar_limiar_alcancavel(
    limiar: float, teto: float, *, motor: str, escala: str, piso: float | None = None
) -> None:
    """
    G-P3. Levanta `LimiarInalcancavelError` se `limiar > teto` — a checagem que
    faltava no carregamento do scanner do MIE e que teria transformado um mês
    de abstenção silenciosa num erro no primeiro minuto.

    `teto` é o maior valor que a escala PODE produzir (para a calibração
    isotônica, `radar.mie.calibration.teto_calibrador`); `escala` nomeia essa
    escala na mensagem (`"probabilidade calibrada"`, `"percentil do dia"`),
    porque o defeito é sempre o mesmo — comparar um número fora da escala dele
    — e a mensagem precisa dizer QUAL escala.

    `piso` (opcional) cobre o espelho: um limiar abaixo do menor valor
    possível aprova TUDO, que é o modo de falha oposto e igualmente silencioso.
    `NaN` em qualquer dos três levanta — comparação com `NaN` é sempre `False`
    e deixaria passar exatamente o caso inválido.
    """
    if math.isnan(limiar) or math.isnan(teto):
        raise LimiarInalcancavelError(
            f"{motor}: limiar ou teto NaN (limiar={limiar}, teto={teto}) na escala {escala!r} — "
            f"comparação com NaN é sempre False e aprovaria o inválido em silêncio (G-P3)."
        )
    if limiar > teto:
        raise LimiarInalcancavelError(
            f"{motor}: limiar de emissão {limiar:.4f} é INALCANÇÁVEL — o teto da escala "
            f"{escala!r} é {teto:.4f}. Nenhum candidato jamais passa: isto é abstenção 100% por "
            f"aritmética, não seletividade (G-P3 do Contrato de Pensamento, ADR 0032). Foi este "
            f"defeito que manteve o MIE v1 mudo por um mês em produção, reportado como saúde."
        )
    if piso is not None:
        if math.isnan(piso):
            raise LimiarInalcancavelError(
                f"{motor}: piso NaN na escala {escala!r} — ver acima, mesmo motivo (G-P3)."
            )
        if limiar < piso:
            raise LimiarInalcancavelError(
                f"{motor}: limiar de emissão {limiar:.4f} está ABAIXO do piso {piso:.4f} da "
                f"escala {escala!r} — aprova todo candidato, que é o espelho do gate "
                f"inalcançável e igualmente silencioso (G-P3)."
            )


# ── G-P4 · Vocabulário de contexto ──────────────────────────────────────────


def validar_contextos(contextos: Iterable[str], *, motor: str) -> None:
    """
    G-P4. Todo rótulo precisa ser `CONTEXTO_GLOBAL` ou
    `"<dimensao>:<valor>"` com `dimensao` em `DIMENSOES_CONTEXTO`.

    Rótulo livre é o que impede o Strategy Router de existir: se um motor
    escreve `"bear"` e outro `"regime_bear"`, o mapa de validade de um não
    é comparável ao do outro, e a Fase 6 do Plano de Reconstrução não tem o
    que orquestrar. Valor vazio (`"regime:"`) também levanta — nomeia uma
    dimensão sem dizer qual recorte dela, o que reabre a porta do subgrupo
    conveniente que o ADR 0031 §4 fecha.
    """
    invalidos: list[str] = []
    for rotulo in contextos:
        if rotulo == CONTEXTO_GLOBAL:
            continue
        dimensao, sep, valor = str(rotulo).partition(SEPARADOR_CONTEXTO)
        if not sep or dimensao not in DIMENSOES_CONTEXTO or not valor:
            invalidos.append(str(rotulo))
    if invalidos:
        raise ContextoForaDoVocabularioError(
            f"{motor}: rótulo(s) de contexto fora do vocabulário: {invalidos}. Use "
            f"{CONTEXTO_GLOBAL!r} ou '<dimensao>{SEPARADOR_CONTEXTO}<valor>' com dimensão em "
            f"{list(DIMENSOES_CONTEXTO)} (G-P4 do Contrato de Pensamento, ADR 0032) — sem vocabulário "
            f"comum, os mapas de validade dos especialistas não são comparáveis e o Strategy "
            f"Router não é construível."
        )

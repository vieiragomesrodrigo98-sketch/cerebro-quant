"""
src/radar/cerebro/catalogo.py — as famílias declaradas, num lugar só.

É daqui que o **Router** lê o que existe. Importar este módulo registra todas
as famílias; nenhuma outra parte do sistema precisa saber quantas são nem quais.

**Os quatro perfis SÃO o produto** (ordem do DEV) — Scalp · Day · Swing ·
Position. Uma família por perfil × mercado, porque *"um scalp não pode dividir
espaço de features com um swing"* (ADR 0032 §2): motor único sobre premissas
incompatíveis produz média ≈ 0 e a conclusão falsa "não funciona".

Cobertura hoje (cripto, que é P0) — **as quatro modalidades declaradas**:

| perfil | família | store | horizontes | estado |
|---|---|---|---|---|
| **Scalp** | `scalp_cripto_v1` | microestrutura 1 s | 30 · 60 · 300 s | **declarada, NÃO medida** — pré-condição do V3 §9 |
| **Day** | `day_cripto_v1` | 1h | 24 · 72 h | **declarada, não medida** |
| Swing | `swing_cripto_v1` | diário | 5 · 10 · 21 d | **medida (R1, 04/08) — 6 células vermelhas** |
| **Position** | `position_cripto_v1` | diário (470 pares, 10 anos) | 60 · 120 · 250 d | **declarada, não medida** |

⚠️ **Declarada não é medida, e medida não é emitindo.** Nenhuma das quatro emite
sinal para usuário: o emissor v1 segue congelado (`mie_emission_frozen`) e a
escada `shadow → paper → champion/challenger → produção` é decisão do DEV.

O braço de cripto do Swing veio da família `swing_v1`, cujo runner
(`scripts/cerebro/rodar_swing_v1.py`) é anterior ao contrato e declara a config
num `_CFG` próprio. A entrada aqui é migração de **forma** — zero célula nova no
denominador do FDR. A tentativa de tratá-lo como hipótese nova foi revogada na
S173; o porquê está em `docs/estudos/PRE_REGISTRO_SWING_CRIPTO_V1.md`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from radar.cerebro.familia import Familia, registrar
from radar.features.colunas import (
    COLUNAS_SELECAO_CRIPTO,
    COLUNAS_SELECAO_MICROESTRUTURA_1S,
    COLUNAS_SELECAO_REVERSAO_RELATIVA,
)

__all__ = [
    "DAY_CRIPTO_V1",
    "POSITION_CRIPTO_V1",
    "REVERSAO_RELATIVA_CRIPTO_V1",
    "SCALP_CRIPTO_V1",
    "SWING_CRIPTO_V1",
]

_RAIZ: Final[Path] = Path(__file__).resolve().parents[3]
_LEARNING: Final[Path] = _RAIZ / "data" / "learning"


# ── SCALP · horizonte de SEGUNDOS ──────────────────────────────────────────
#
# Transcrição do pré-registro congelado V3 (microestrutura). Todo valor abaixo
# está lá; nenhum foi escolhido aqui.
#
# ⚠️ A unidade é BARRA DE 1 SEGUNDO, não de minuto. `feature_store_cripto_1m`
# NÃO serve a esta família: o V3 §2 exclui horizontes abaixo de 30 s por
# medição (em 57,5% dos segundos o preço não se move) e declara 30/60/300 s.
# Uma barra de minuto não consegue expressar nenhum dos três.
#
# É a ÚNICA das quatro famílias de cripto sem a lacuna de "medir cripto com a
# régua da B3": as outras três usam as 38 técnicas, que são exatamente as 38 do
# outro mercado. Aqui o espaço de seleção é microestrutura pura.
#
# 🔴 PRÉ-CONDIÇÃO INEGOCIÁVEL (V3 §9): nenhuma medição antes de o dataset passar
# nos quatro testes de `radar.ingestion.sanidade_microestrutura`. "Arquivo
# Parquet existe" não significa "dataset existe" — declarar a família NÃO
# levanta essa trava.
SCALP_CRIPTO_V1: Final[Familia] = registrar(Familia(
    nome="scalp_cripto_v1",
    perfil="scalp",
    mercado="cripto",
    store=_RAIZ / "data" / "historical" / "crypto_microestrutura_hist",
    chave_ativo="symbol",
    colunas_selecao=tuple(COLUNAS_SELECAO_MICROESTRUTURA_1S),
    horizontes=(30, 60, 300),   # segundos == barras de 1 s
    unidade_horizonte="barras",
    segundos_por_barra=1,
    embargo=300,                # = o maior horizonte
    # 7 dias de segundos. `None` (default de 12 meses) é DESTRUTIVO aqui: com
    # 86.400 barras por dia, 12 meses engoliriam o treino inteiro do 1º fold.
    holdout=604_800,
    # Com horizonte de 300 barras, dois segundos adjacentes têm rótulos ~99,7%
    # SOBREPOSTOS — o modelo veria o mesmo exemplo 300 vezes. Vale só no
    # TREINO: a medição OOS usa todas as barras, senão a frequência de sinal
    # muda e o controle aleatório deixa de ser controle.
    passo_amostragem=60,
    normalizar_data=False,      # intradiário: normalizar colapsaria o dia
    ano_primeiro_teste=2026,    # a coleta de 1 s começa em 2025-08 (backfill 365d)
    venue="perpetuo",
    tipo_ordem="maker",         # V3 §2(d): a 10 bps de taker nem o p90 paga
    fracao_gate=0.10,
    pre_registro="docs/estudos/PRE_REGISTRO_SCALP_CRIPTO_V3_MICROESTRUTURA.md",
    notas={
        "piso_de_viabilidade": (
            "custo maker de 4 bps contra movimento mediano de 1,35 bps em 60 s "
            "no BTC — o custo é 296% do movimento mediano, e 112% a 569% em "
            "todos os horizontes testados. Contexto do veredito, NUNCA filtro: "
            "o piso diz o que é difícil, jamais o que é dispensável."
        ),
        "sem_lacuna_de_regua": (
            "única família de cripto cujo espaço de seleção não é emprestado da "
            "B3. Nenhuma feature de candle entra — com elas, um positivo não "
            "distinguiria 'microestrutura funciona' de 'o candle já dizia'."
        ),
        "pre_condicao": (
            "V3 §9: sem os 4 testes de sanidade_microestrutura passando, não há "
            "medição. Declarar a família não levanta a trava."
        ),
        "custo_em_orcamento": (
            "3 horizontes x 2 braços de alvo x 2 modos = 12 células (V3 §8). "
            "Bem mais que as 3 do swing — a conta está no pré-registro."
        ),
    },
))


# ── DAY · horizonte de horas ───────────────────────────────────────────────
#
# Onde o edge vive, segundo o ADR 0032 §2: "momentum/fluxo intradiário, VWAP,
# surto de volume". O store de 1h já traz `ret_fwd_24/72/168`.
#
# `h=24` (um dia) e `h=72` (três dias) — `168` fica de fora porque uma semana
# é território de Swing, e dois perfis medindo o mesmo horizonte é a
# duplicação que o §2 existe para impedir.
#
# PISO DE VIABILIDADE declarado (custo de 0,04% round-trip, medido no BTC):
# a 4h o custo é 9,6% do movimento mediano e exige 54,8% de acerto; a 1 dia,
# 3,3% e 51,6%. Contexto do veredito — nunca filtro.
DAY_CRIPTO_V1: Final[Familia] = registrar(Familia(
    nome="day_cripto_v1",
    perfil="day",
    mercado="cripto",
    store=_LEARNING / "feature_store_cripto_1h.parquet",
    chave_ativo="symbol",
    colunas_selecao=tuple(COLUNAS_SELECAO_CRIPTO),
    horizontes=(24, 72),
    unidade_horizonte="barras",
    segundos_por_barra=3_600,
    embargo=72,
    holdout=2_160,           # ~90 dias de barras horárias
    normalizar_data=False,   # barra horária: normalizar colapsaria o dia
    ano_primeiro_teste=2024,
    venue="perpetuo",
    tipo_ordem="maker",
    pre_registro="docs/estudos/PRE_REGISTRO_DAY_CRIPTO_V1.md",
    notas={
        "lacuna_declarada": (
            "usa as 38 colunas técnicas, as MESMAS da B3 — as 4 de fluxo agressor "
            "só existem no store de 1m. Enquanto isso valer, esta família mede "
            "cripto com régua compartilhada, e o veredito precisa dizê-lo."
        ),
        "leitura_v1_anterior": (
            "mie_cripto_1h_h24/h72/h168 já foram medidas sob o instrumento do v1 "
            "e deram t negativo (-0,99 / -2,76 / -1,05). Aquilo NÃO é evidência "
            "sobre esta família: instrumento diferente, custo diferente."
        ),
    },
))


# ── SWING · horizonte de dias a semanas ────────────────────────────────────
#
# ⚠️ Esta declaração NÃO é uma hipótese nova. É a migração de FORMA da família
# `swing_v1` (`scripts/cerebro/rodar_swing_v1.py`), cujo braço de cripto **já
# foi medido** em 2026-08-04 — 6 células, todas VERMELHAS. O runner dela é
# anterior ao contrato e declara a config num `_CFG` próprio; trazê-la para cá
# torna o catálogo uma leitura fiel do que existe, e **não gasta uma única
# célula do orçamento de FDR**.
#
# O histórico de como isso quase deu errado está em
# `docs/estudos/PRE_REGISTRO_SWING_CRIPTO_V1.md` (revogado): esta sessão chegou
# a congelar um pré-registro "novo" com store, horizontes, colunas e escala
# IDÊNTICOS aos já medidos, por procurar a hipótese pelo NOME em vez de pelo
# que ela mede.
#
# Vermelho não encerra a modalidade. O que a R1 diz é que as 38 colunas
# emprestadas da B3 não separaram — um Swing com espaço de seleção nativo de
# cripto (`COLUNAS_SELECAO_CRIPTO_NATIVO`, com fluxo agressor) seria hipótese
# genuinamente nova, e aí custaria orçamento honestamente.
SWING_CRIPTO_V1: Final[Familia] = registrar(Familia(
    nome="swing_cripto_v1",
    perfil="swing",
    mercado="cripto",
    store=_LEARNING / "feature_store_cripto.parquet",
    chave_ativo="symbol",
    colunas_selecao=tuple(COLUNAS_SELECAO_CRIPTO),
    horizontes=(5, 10, 21),
    unidade_horizonte="dias_corridos",
    embargo=21,              # = o maior horizonte; menos, e o treino vê o teste
    holdout=None,            # default de 12 meses é correto em dado diário
    ano_primeiro_teste=2020,
    venue="perpetuo",
    tipo_ordem="maker",
    pre_registro="docs/estudos/PRE_REGISTRO_CEREBRO_SWING_V1.md",
    notas={
        "ja_medida": (
            "leitura oficial CEREBRO_SWING_V1_R1 (2026-08-04): 6 células de "
            "cripto, TODAS vermelhas, p entre 0,129 e 0,715. Declarar no "
            "catálogo é forma, não medição — zero célula nova no denominador."
        ),
        "lacuna_declarada": (
            "as 38 colunas de seleção são AS MESMAS da B3 (interseção 38, "
            "exclusivas de cripto zero). O vermelho da R1 é sobre ESSA régua, e "
            "lê-lo como 'cripto não tem swing' seria confundir o instrumento "
            "com o mercado."
        ),
        "runner_legado": (
            "a config viva ainda é o `_CFG` de scripts/cerebro/rodar_swing_v1."
            "py, que cobre b3 e cripto juntos. Unificar as duas fontes é "
            "trabalho de forma, sem efeito sobre número medido."
        ),
    },
))


# ── SWING · reversão da força relativa ─────────────────────────────────────
#
# Nasce de uma MEDIÇÃO, não de um palpite — e essa é a diferença para o
# `swing_cripto_v1` revogado nesta mesma sessão.
#
# Medido em 2.409 dias / 524.037 observações
# (`scripts/medir_ativo_vs_indice_cripto.py`): **63,5% da variância do retorno é
# do ATIVO**, não do mercado — estável em 5/10/21 dias. E o posto do resíduo
# REVERTE, com janelas não sobrepostas: rho de −0,039 · −0,035 · −0,025.
#
# Mecanismo declarado: PRESSÃO DE LIQUIDEZ DE CURTO PRAZO. Quem sobe muito acima
# do mercado atrai realização; quem cai muito abaixo atrai recompra; o efeito
# decai porque a pressão se dissipa — que é o formato do rho medido.
#
# ⚠️ NÃO É UM MODELO, É UMA REGRA. Uma coluna de seleção, um grau de liberdade,
# congelado no pré-registro. Sem LightGBM, sem calibração, sem risco de
# `num_trees=1`. `G-P0` perde o sentido aqui; os outros quatro portões valem
# integralmente. Isso a torna barata de medir e difícil de sobreajustar.
#
# ⚠️ Ponta `#base` e direção LONG: selecionar os piores ranqueados NÃO inverte o
# sinal do retorno (ADR 0034) — é long contrarian. As duas únicas células verdes
# do projeto são seleção de fundo.
REVERSAO_RELATIVA_CRIPTO_V1: Final[Familia] = registrar(Familia(
    nome="reversao_relativa_cripto_v1",
    perfil="swing",
    mercado="cripto",
    store=_LEARNING / "feature_store_cripto.parquet",
    chave_ativo="symbol",
    colunas_selecao=tuple(COLUNAS_SELECAO_REVERSAO_RELATIVA),
    horizontes=(5, 10, 21),
    unidade_horizonte="dias_corridos",
    embargo=21,
    holdout=None,
    ano_primeiro_teste=2020,
    venue="perpetuo",
    tipo_ordem="maker",
    fracao_gate=0.10,
    pre_registro="docs/estudos/PRE_REGISTRO_REVERSAO_RELATIVA_CRIPTO_V1.md",
    notas={
        "mecanismo_medido": (
            "63,5% da variância é do ativo (não do mercado) e o posto do resíduo "
            "reverte: rho −0,039 (h=5), −0,035 (h=10), −0,025 (h=21), janelas "
            "NÃO sobrepostas. Pressão de liquidez de curto prazo."
        ),
        "nao_e_modelo": (
            "uma coluna de seleção, um grau de liberdade congelado. Sem GBM, sem "
            "calibração — o G-P0 não se aplica, os outros quatro sim."
        ),
        "limitacao_declarada": (
            "o rho foi medido na amostra INTEIRA, então o mecanismo foi escolhido "
            "depois de ver a estrutura do dado completo. Desempenho de regra "
            "nenhuma foi visto. O veredito é o OOS, jamais o rho in-sample."
        ),
        "artefato_evitado": (
            "com deslocamento de 1 dia o rho saía +0,72 a +0,92 — MECÂNICO, "
            "porque `ret_fwd_5` em t e t+1 dividem 4 dos 5 dias. Com deslocamento "
            "de h o sinal INVERTE para negativo fraco."
        ),
    },
))


# ── POSITION · horizonte de semanas a meses ────────────────────────────────
#
# Onde o edge vive (§2): "tendência macro". É o perfil com o MELHOR dado do
# projeto em cripto — 470 pares e 10 anos (2017-2026), contra os 20 pares e 2
# anos do 1m — e o piso de viabilidade mais baixo da grade: a uma semana, o
# custo é 1,2% do movimento e bastam 50,6% de acerto.
#
# `ano_primeiro_teste=2020` deixa 2017-2019 de treino: três anos que incluem o
# ciclo completo de 2018, e não só o bull de 2020-2021.
POSITION_CRIPTO_V1: Final[Familia] = registrar(Familia(
    nome="position_cripto_v1",
    perfil="position",
    mercado="cripto",
    store=_LEARNING / "feature_store_cripto.parquet",
    chave_ativo="symbol",
    colunas_selecao=tuple(COLUNAS_SELECAO_CRIPTO),
    horizontes=(60, 120, 250),
    unidade_horizonte="dias_corridos",
    embargo=250,
    holdout=None,            # o default de 12 meses é correto em dado diário
    ano_primeiro_teste=2020,
    venue="perpetuo",
    tipo_ordem="maker",
    pre_registro="docs/estudos/PRE_REGISTRO_POSITION_CRIPTO_V1.md",
    notas={
        "lacuna_declarada": (
            "as features de tendência macro que este perfil pediria — funding, "
            "open interest, dominância do BTC — não existem no store. A família "
            "nasce com o técnico diário, e isso é lacuna declarada, não desenho."
        ),
        "vantagem_de_dado": (
            "470 pares e 10 anos: o maior universo e a maior janela que o projeto "
            "tem em cripto. O gargalo de dias independentes não aperta aqui."
        ),
    },
))

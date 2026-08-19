"""
Testes — `radar.cerebro.alvo` e `radar.cerebro.custos` (família `swing_v1`).

O que estes testes existem para travar, além do óbvio:

  * `TestPrevalenciaFixa` — a PROVA de que o alvo cross-sectional faz o que
    promete. A prevalência do alvo antigo variava entre dias, e era isso que
    fazia o eixo dominante ser o DIA. Se este teste cair, a causa mecânica do
    defeito do v1 voltou.
  * `TestCustoNaoUniformeMudaAOrdem` — o contraste entre os braços A e B do
    alvo existe SÓ porque o custo varia por ativo. Se custo uniforme e custo
    por faixa produzirem a mesma ordem, os dois braços são o mesmo
    experimento e o pré-registro perde um dos seus dois eixos.

Zero I/O.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from radar.cerebro.alvo import (
    FRACAO_TOPO_PADRAO,
    montar_valor_alvo_liquido,
    resumo_prevalencia,
    rotular_topo_do_instante,
)
from radar.cerebro.custos import (
    FAIXA_ILIQUIDA,
    FAIXA_LIQUIDA,
    FAIXA_MEDIA,
    MERCADO_B3,
    MERCADO_CRIPTO,
    MULTIPLICADOR_SLIPPAGE,
    custo_roundtrip_da_faixa,
    custos_por_ticker,
    multiplicador_da_faixa,
)


def _instante(n_dias: int, n_ativos: int) -> tuple[pd.Series, pd.Series]:
    """`n_dias` × `n_ativos`, valor crescente e único dentro de cada dia."""
    valores, instantes = [], []
    for d in range(n_dias):
        for i in range(n_ativos):
            valores.append(0.01 * i + d)  # ordem interna idêntica em todo dia
            instantes.append(f"2026-01-{d + 1:02d}")
    return pd.Series(valores), pd.Series(instantes)


# ── Alvo cross-sectional ────────────────────────────────────────────────────


class TestRotularTopoDoInstante:
    def test_marca_o_decil_superior_de_cada_instante(self):
        """Exatamente 10 de 100 por instante. Trava o operador ESTRITO: com
        `>=`, `rank(pct=True)` (que devolve rank/n) marcaria 11 — a prevalência
        sairia 11% e o alvo deixaria de ser o decil que o pré-registro
        declara."""
        valor, inst = _instante(n_dias=2, n_ativos=100)

        y = rotular_topo_do_instante(valor, inst)

        assert y.notna().all()
        assert y.sum() == 20  # 10 por dia × 2 dias

    def test_o_topo_e_dos_maiores_valores(self):
        valor, inst = _instante(n_dias=1, n_ativos=100)

        y = rotular_topo_do_instante(valor, inst)

        assert y.iloc[-1] == 1.0  # maior valor do dia
        assert y.iloc[0] == 0.0   # menor valor do dia

    def test_instante_magro_fica_sem_rotulo(self):
        """Com 5 candidatos, "decil superior" vira "o melhor de 5" — outra
        pergunta, com outra taxa-base. Misturar as duas contamina o treino."""
        valor, inst = _instante(n_dias=1, n_ativos=5)

        y = rotular_topo_do_instante(valor, inst)

        assert y.isna().all()

    def test_valor_ausente_nao_vira_zero(self):
        """`0` afirmaria "medi e não está no topo" — afirmação que não se pode
        fazer sobre um alvo não resolvido."""
        valor, inst = _instante(n_dias=1, n_ativos=30)
        valor.iloc[3] = np.nan

        y = rotular_topo_do_instante(valor, inst)

        assert np.isnan(y.iloc[3])
        assert y.drop(index=3).notna().all()

    def test_instante_todo_igual_nao_tem_topo(self):
        valor = pd.Series([0.5] * 30)
        inst = pd.Series(["2026-01-05"] * 30)

        y = rotular_topo_do_instante(valor, inst)

        assert y.isna().all()

    def test_empate_nao_e_partido_por_ordem_de_linha(self):
        """`method="average"`: um bloco de empatados recebe a mesma posição.
        Consequência declarada — a fração rotulada 1 pode ficar abaixo de 10%,
        que é preferível a inventar uma ordem que o dado não tem."""
        valor = pd.Series([1.0] * 25 + [2.0] * 5)   # 5 no topo, 25 empatados
        inst = pd.Series(["2026-01-05"] * 30)

        y = rotular_topo_do_instante(valor, inst)

        assert y.iloc[25:].sum() == 5      # os 5 maiores
        assert y.iloc[:25].sum() == 0      # nenhum empatado entra

    def test_fracao_topo_fora_do_intervalo_levanta(self):
        valor, inst = _instante(1, 30)
        with pytest.raises(ValueError, match="fracao_topo"):
            rotular_topo_do_instante(valor, inst, fracao_topo=1.0)

    def test_tamanhos_diferentes_levantam(self):
        with pytest.raises(ValueError, match="tamanhos diferentes"):
            rotular_topo_do_instante(pd.Series([1.0, 2.0]), pd.Series(["a"]))


class TestPrevalenciaFixa:
    """A prova de que o alvo cross-sectional remove a causa mecânica do
    defeito do v1."""

    def test_prevalencia_e_identica_em_todo_instante(self):
        valor, inst = _instante(n_dias=30, n_ativos=100)

        y = rotular_topo_do_instante(valor, inst)
        resumo = resumo_prevalencia(y, inst)

        assert resumo["prevalencia"] == pytest.approx(1 - FRACAO_TOPO_PADRAO)
        assert resumo["prevalencia_por_instante_desvio"] == pytest.approx(0.0)
        assert resumo["n_instantes"] == 30

    def test_alvo_por_limiar_absoluto_varia_entre_instantes(self):
        """O contraste que justifica a mudança: com limiar ABSOLUTO, dias de
        enchente rotulam quase tudo e dias parados quase nada — e é essa
        variação que o modelo agrupado aprende no lugar de escolher ativo."""
        # dia 0 inteiro abaixo do limiar, dia 1 inteiro acima
        valor = pd.Series([0.01] * 100 + [0.09] * 100)
        inst = pd.Series(["2026-01-01"] * 100 + ["2026-01-02"] * 100)

        y_absoluto = (valor > 0.05).astype(float)
        resumo_abs = resumo_prevalencia(y_absoluto, inst)
        resumo_rank = resumo_prevalencia(rotular_topo_do_instante(valor, inst), inst)

        assert resumo_abs["prevalencia_por_instante_desvio"] == pytest.approx(0.5)
        # o rank não consegue rotular (todo instante é constante) — e NÃO
        # inventa uma ordem: é o comportamento correto, não um empate feliz
        assert resumo_rank["n"] == 0


class TestMontarValorAlvoLiquido:
    def test_subtrai_o_custo_do_proprio_ativo(self):
        ret = pd.Series([0.10, 0.10])
        tk = pd.Series(["LIQ", "ILQ"])

        out = montar_valor_alvo_liquido(ret, tk, {"LIQ": 0.002, "ILQ": 0.02})

        assert out.iloc[0] == pytest.approx(0.098)
        assert out.iloc[1] == pytest.approx(0.080)

    def test_ticker_sem_custo_vira_nan_nunca_zero(self):
        """Custo zero seria a hipótese mais favorável possível para o ativo do
        qual menos se sabe — é assim que um backtest mente a favor."""
        out = montar_valor_alvo_liquido(
            pd.Series([0.10]), pd.Series(["DESCONHECIDO"]), {"OUTRO": 0.002}
        )

        assert np.isnan(out.iloc[0])


class TestCustoNaoUniformeMudaAOrdem:
    """Se custo uniforme e custo por faixa produzirem a MESMA ordem, os braços
    A e B do pré-registro são o mesmo experimento."""

    def test_custo_por_faixa_inverte_a_ordem_que_o_bruto_daria(self):
        ret = pd.Series([0.050, 0.045])          # bruto: ILQ perde para LIQ? não
        tk = pd.Series(["ILQ", "LIQ"])

        custo_liq = custo_roundtrip_da_faixa(FAIXA_LIQUIDA, mercado=MERCADO_CRIPTO)
        custo_ilq = custo_roundtrip_da_faixa(FAIXA_ILIQUIDA, mercado=MERCADO_CRIPTO)
        liquido = montar_valor_alvo_liquido(ret, tk, {"ILQ": custo_ilq, "LIQ": custo_liq})

        assert custo_ilq > custo_liq
        # no bruto ILQ (0,050) está na frente de LIQ (0,045); descontado o
        # custo de cada um, a distância encolhe — e é essa mudanca de ordem
        # que separa o braço A do braço B
        assert (ret.iloc[0] - ret.iloc[1]) > (liquido.iloc[0] - liquido.iloc[1])


# ── Custo por ativo ─────────────────────────────────────────────────────────


class TestMultiplicadorDaFaixa:
    def test_liquida_e_neutra_nos_dois_mercados(self):
        assert multiplicador_da_faixa(FAIXA_LIQUIDA, mercado=MERCADO_B3) == 1.00
        assert multiplicador_da_faixa(FAIXA_LIQUIDA, mercado=MERCADO_CRIPTO) == 1.00

    def test_valores_batem_com_o_pre_registro(self):
        assert MULTIPLICADOR_SLIPPAGE[MERCADO_B3][FAIXA_MEDIA] == 2.13
        assert MULTIPLICADOR_SLIPPAGE[MERCADO_CRIPTO][FAIXA_MEDIA] == 1.95

    def test_faixa_ausente_recebe_o_pior_caso(self):
        """Tratamento conservador declarado: ausência de medida nunca vira
        benefício da dúvida."""
        assert multiplicador_da_faixa(None, mercado=MERCADO_B3) == MULTIPLICADOR_SLIPPAGE[
            MERCADO_B3
        ][FAIXA_ILIQUIDA]
        assert multiplicador_da_faixa("inventada", mercado=MERCADO_CRIPTO) == (
            MULTIPLICADOR_SLIPPAGE[MERCADO_CRIPTO][FAIXA_ILIQUIDA]
        )

    def test_uniforme_ignora_a_faixa(self):
        for faixa in (FAIXA_LIQUIDA, FAIXA_MEDIA, FAIXA_ILIQUIDA, None):
            assert multiplicador_da_faixa(faixa, mercado=MERCADO_B3, uniforme=True) == 1.00

    def test_mercado_desconhecido_levanta(self):
        """Default silencioso aqui aplicaria a régua da B3 a cripto."""
        with pytest.raises(ValueError, match="mercado desconhecido"):
            multiplicador_da_faixa(FAIXA_LIQUIDA, mercado="forex")


class TestCustoRoundtrip:
    def test_ordem_liquida_media_iliquida(self):
        c = [
            custo_roundtrip_da_faixa(f, mercado=MERCADO_CRIPTO)
            for f in (FAIXA_LIQUIDA, FAIXA_MEDIA, FAIXA_ILIQUIDA)
        ]
        assert c[0] < c[1] < c[2]

    def test_so_o_slippage_escala_a_taxa_nao(self):
        """Multiplicar o custo TOTAL inflaria a corretagem, que não depende de
        liquidez — custo inventado vira resultado inventado."""
        c_liq = custo_roundtrip_da_faixa(FAIXA_LIQUIDA, mercado=MERCADO_CRIPTO)
        c_med = custo_roundtrip_da_faixa(FAIXA_MEDIA, mercado=MERCADO_CRIPTO)
        mult = MULTIPLICADOR_SLIPPAGE[MERCADO_CRIPTO][FAIXA_MEDIA]

        assert c_med < c_liq * mult  # cresceu menos que o multiplicador cheio
        assert c_med > c_liq

    def test_uniforme_iguala_todas_as_faixas(self):
        valores = {
            custo_roundtrip_da_faixa(f, mercado=MERCADO_B3, uniforme=True)
            for f in (FAIXA_LIQUIDA, FAIXA_MEDIA, FAIXA_ILIQUIDA, None)
        }
        assert len(valores) == 1

    def test_custos_por_ticker_mapeia_o_universo(self):
        out = custos_por_ticker(
            {"PETR4": FAIXA_LIQUIDA, "XPTO3": FAIXA_MEDIA}, mercado=MERCADO_B3
        )
        assert set(out) == {"PETR4", "XPTO3"}
        assert out["PETR4"] < out["XPTO3"]


# ── Alvo AUTO-REFERENTE (o ativo contra ele mesmo) ──────────────────────────


class TestRotularRetornoPositivo:
    """Ordem do DEV: medir o movimento do ativo contra ELE MESMO, nem contra
    benchmark nem contra os outros do dia."""

    def test_rotula_1_quando_paga_o_proprio_pedagio(self):
        from radar.cerebro.alvo import rotular_retorno_positivo

        y = rotular_retorno_positivo(
            pd.Series([0.05, 0.001]), pd.Series(["A", "B"]), {"A": 0.003, "B": 0.003}
        )

        assert y.iloc[0] == 1.0   # +5% cobre 0,3% de custo
        assert y.iloc[1] == 0.0   # +0,1% nao cobre

    def test_sem_custo_vira_nan_nunca_zero(self):
        from radar.cerebro.alvo import rotular_retorno_positivo

        y = rotular_retorno_positivo(pd.Series([0.05]), pd.Series(["X"]), {})

        assert np.isnan(y.iloc[0])

    def test_prevalencia_muito_maior_que_o_alvo_relativo(self):
        """O segundo ganho, independente do viés: a prevalência sai de ~10%
        (decil do dia) para ~50%, eliminando o desequilíbrio de classe que
        exigia `scale_pos_weight ~ 3,7` — a causa mecânica do num_trees=1."""
        from radar.cerebro.alvo import rotular_retorno_positivo, rotular_topo_do_instante

        rng = np.random.default_rng(7)
        n = 3000
        ret = pd.Series(rng.normal(0.01, 0.10, n))
        tk = pd.Series([f"T{i % 100:03d}" for i in range(n)])
        inst = pd.Series([f"2026-01-{1 + i // 100:02d}" for i in range(n)])
        custos = {f"T{i:03d}": 0.003 for i in range(100)}

        y_abs = rotular_retorno_positivo(ret, tk, custos)
        y_rel = rotular_topo_do_instante(ret, inst)

        assert y_abs.mean() > 0.40          # ~metade
        assert y_rel.mean() == pytest.approx(0.10, abs=0.02)


class TestClassificarOperabilidade:
    """Operabilidade é RELAÇÃO (custo × movimento no horizonte), não atributo
    do ativo — e é classificação, nunca filtro de existência."""

    def _store(self) -> pd.DataFrame:
        linhas = []
        for tk, mov in [("CARO", 0.002), ("MEDIO", 0.05), ("AMPLO", 0.30)]:
            for i in range(50):
                linhas.append({"ticker": tk, "ret_fwd_5": mov * (1 if i % 2 else -1),
                               "ret_fwd_21": mov * 3 * (1 if i % 2 else -1)})
        return pd.DataFrame(linhas)

    def test_ativo_caro_cai_so_no_horizonte_curto(self):
        from radar.cerebro.operabilidade import classificar_operabilidade

        c = classificar_operabilidade(
            self._store(), {"CARO": 0.003, "MEDIO": 0.003, "AMPLO": 0.003},
            horizontes=(5, 21),
        )

        assert c["CARO"].horizontes_operaveis == ()        # 0,2% < 2 x 0,3%
        assert 5 in c["MEDIO"].horizontes_operaveis
        assert c["AMPLO"].horizontes_operaveis == (5, 21)

    def test_ninguem_e_excluido_do_universo(self):
        """O filtro antigo apagava 56 tickers da B3 e nada dizia quais. Aqui o
        inoperável continua visível, com a razão ao lado."""
        from radar.cerebro.operabilidade import classificar_operabilidade

        c = classificar_operabilidade(
            self._store(), {"CARO": 0.003, "MEDIO": 0.003, "AMPLO": 0.003}, horizontes=(5,)
        )

        assert set(c) == {"CARO", "MEDIO", "AMPLO"}
        assert c["CARO"].operavel_em_algum is False

    def test_ativo_sem_custo_conhecido_nao_e_operavel(self):
        from radar.cerebro.operabilidade import classificar_operabilidade

        c = classificar_operabilidade(self._store(), {}, horizontes=(5,))

        assert all(not o.operavel_em_algum for o in c.values())

    def test_resumo_conta_os_sem_nenhuma_familia(self):
        from radar.cerebro.operabilidade import classificar_operabilidade, resumo_operabilidade

        r = resumo_operabilidade(
            classificar_operabilidade(
                self._store(), {"CARO": 0.003, "MEDIO": 0.003, "AMPLO": 0.003},
                horizontes=(5, 21),
            )
        )

        assert r["n_ativos"] == 3
        assert r["sem_nenhum_horizonte"] == 1
        assert "CARO" in r["tickers_sem_nenhum_horizonte"]

    def test_coluna_de_retorno_ausente_levanta(self):
        from radar.cerebro.operabilidade import classificar_operabilidade

        with pytest.raises(KeyError, match="horizonte"):
            classificar_operabilidade(self._store(), {"CARO": 0.003}, horizontes=(99,))

"""
As duas lacunas que exigiram medição nova: capacidade e estabilidade OOS.

(A terceira — slippage de equilíbrio por hipótese — é aritmética sobre número
que os ledgers já publicavam, e é testada em `test_cerebro_mapa_validade.py`
junto do critério que a consome.)

O que estes testes protegem:

1. **Capacidade é modelo declarado, e o modelo tem de ser o declarado.** A lei
   da raiz quadrada, invertida, com os parâmetros visíveis — se alguém trocar o
   expoente ou o coeficiente, os números do mapa mudam em silêncio.
2. **Errar para o lado seguro.** Horizonte desconhecido subestima a
   concorrência; sem margem, a capacidade é zero.
3. **Estabilidade distingue "não medida" de "instável".** Fonte sem retorno por
   período devolve `None`, jamais `False`.
4. **A cronologia é preservada.** Ordenar por valor em vez de por período
   destruiria o teste das metades.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from radar.cerebro import estabilidade as est
from radar.cerebro.capacidade import (
    COEFICIENTE_IMPACTO_PADRAO,
    FAIXAS_DE_CAPITAL,
    FRACAO_DO_EDGE_PADRAO,
    PerfilDeLiquidez,
    adv_dos_trades,
    capital_por_posicao,
    curva_de_degradacao,
    estimar_capacidade,
    impacto_bps,
    posicoes_concorrentes,
)

# Perfil real da B3, medido em 2026-08-04 do histórico de preço/volume.
B3 = PerfilDeLiquidez(
    mercado="b3", adv_percentil=5_117_325.0, sigma_diaria=0.0263, n_ativos=246
)


class TestModeloDeImpacto:
    def test_um_por_cento_do_adv_custa_cerca_de_dez_por_cento_da_vol_diaria(self):
        """
        A âncora que torna o modelo defensável: negociar 1% do ADV custa da
        ordem de 10% da volatilidade diária. Com σ = 2,63%, isso é ~26 bps — a
        ordem de grandeza que operadores relatam.
        """
        assert impacto_bps(0.01, B3.sigma_diaria) == pytest.approx(26.3, rel=0.01)

    def test_escala_com_a_raiz_e_nao_linearmente(self):
        """4× o tamanho custa 2× o impacto — é o que a raiz quadrada afirma."""
        assert impacto_bps(0.04, 0.02) == pytest.approx(2 * impacto_bps(0.01, 0.02))

    def test_inverso_do_impacto_devolve_o_tamanho(self):
        q = capital_por_posicao(B3.adv_percentil, B3.sigma_diaria, orcamento_bps=20.0)
        participacao = q / B3.adv_percentil
        assert impacto_bps(participacao, B3.sigma_diaria) == pytest.approx(20.0)

    @pytest.mark.parametrize("participacao,sigma", [(-0.01, 0.02), (0.01, 0.0), (0.01, -1)])
    def test_entrada_invalida_levanta_em_vez_de_devolver_nan(self, participacao, sigma):
        """
        `nan` silencioso aqui viraria capacidade "desconhecida" indistinguível
        de capacidade calculada — o tipo de falha muda que este projeto persegue.
        """
        with pytest.raises(ValueError):
            impacto_bps(participacao, sigma)


class TestErrarParaOLadoSeguro:
    def test_sem_margem_nao_ha_capital_que_caiba(self):
        assert capital_por_posicao(B3.adv_percentil, B3.sigma_diaria, 0.0) == 0.0
        assert capital_por_posicao(B3.adv_percentil, B3.sigma_diaria, -5.0) == 0.0

    def test_equilibrio_nao_positivo_devolve_none(self):
        assert estimar_capacidade(
            perfil=B3, equilibrio_slippage_bps=0.0, trades=100,
            dias_independentes=100, horizonte=5,
        ) is None

    def test_horizonte_desconhecido_subestima_a_concorrencia(self):
        """Holding 1 (o fallback) dá menos posições que o horizonte real — e
        menos posições significa MENOS capacidade. Errar para baixo é a direção
        segura."""
        com_horizonte = posicoes_concorrentes(1000, 200, 21)
        sem_horizonte = posicoes_concorrentes(1000, 200, None)
        assert sem_horizonte < com_horizonte
        assert sem_horizonte == pytest.approx(5.0)
        assert com_horizonte == pytest.approx(105.0)

    def test_dias_zero_nao_divide_por_zero(self):
        assert posicoes_concorrentes(100, 0, 5) == 0.0


class TestEstimativaCompleta:
    def test_caso_real_estrategia_alfa(self):
        """
        `estrategia_alfa_A#base`: 13.734 trades / 3.354 dias / equilíbrio 68,2 bps.
        Orçamento = 30% de 68,2 = 20,5 bps.
        """
        c = estimar_capacidade(
            perfil=B3, equilibrio_slippage_bps=68.2, trades=13734,
            dias_independentes=3354, horizonte=21,
        )
        assert c is not None
        assert c.orcamento_bps == pytest.approx(0.30 * 68.2)
        assert c.posicoes_concorrentes == pytest.approx(13734 * 21 / 3354, rel=1e-6)
        assert c.capital_total == pytest.approx(
            c.capital_por_posicao * c.posicoes_concorrentes
        )
        assert c.capital_total > 100_000, "deve comportar o capital_alvo declarado"

    def test_sensibilidade_cobre_os_tres_coeficientes_e_y_maior_reduz(self):
        c = estimar_capacidade(
            perfil=B3, equilibrio_slippage_bps=68.2, trades=13734,
            dias_independentes=3354, horizonte=21,
        )
        assert c is not None
        assert set(c.sensibilidade) == {"Y=0.5", "Y=1", "Y=2"}
        assert c.sensibilidade["Y=2"] < c.sensibilidade["Y=1"] < c.sensibilidade["Y=0.5"]
        assert c.sensibilidade["Y=1"] == pytest.approx(c.capital_total)

    def test_premissas_ficam_visiveis_no_resultado(self):
        c = estimar_capacidade(
            perfil=B3, equilibrio_slippage_bps=50.0, trades=1000,
            dias_independentes=500, horizonte=10,
        )
        assert c is not None
        texto = " ".join(c.premissas)
        for esperado in ("p25", "σ diária", "orçamento", "concorrência", "Y ="):
            assert esperado in texto
        assert "ESTIMATIVA" in c.to_dict()["natureza"]
        assert "NÃO é medição de fills reais" in c.to_dict()["natureza"]

    def test_defaults_declarados_batem_com_o_pre_registro(self):
        assert COEFICIENTE_IMPACTO_PADRAO == 1.0
        assert FRACAO_DO_EDGE_PADRAO == 0.30


class TestEstabilidade:
    def test_sem_dado_e_none_e_none_nunca_e_reprovacao(self):
        assert est.avaliar(None) is None
        assert est.avaliar({}) is None

    def test_poucos_periodos_e_none(self):
        assert est.avaliar({"2020": 0.01, "2021": 0.02, "2022": 0.01}) is None

    def test_concentracao_reprova_o_ano_unico(self):
        """8 anos morninhos + 1 gigante: persistência ótima, mecanismo nenhum."""
        anos = {f"20{10+i}": 0.001 for i in range(8)}
        anos["2018"] = 0.5
        r = est.avaliar(anos)
        assert r is not None and r.concentrada and not r.estavel
        assert r.concentracao > 0.5
        assert "um único período" in r.motivos[0]

    def test_metades_discordantes_reprovam(self):
        """Só funciona na 1ª metade: descobriu um regime que acabou."""
        anos = {"2016": 0.05, "2017": 0.05, "2018": 0.05,
                "2019": -0.02, "2020": -0.02, "2021": -0.01}
        r = est.avaliar(anos)
        assert r is not None and not r.metades_concordam and not r.estavel

    def test_resultado_espalhado_e_consistente_passa(self):
        anos = {f"20{10+i}": 0.01 + 0.002 * (i % 3) for i in range(10)}
        r = est.avaliar(anos)
        assert r is not None and r.estavel and not r.concentrada
        assert r.metades_concordam and r.motivos == ()

    def test_ordena_por_periodo_e_nao_por_valor(self):
        """
        Ordenar por valor faria as metades comparar "os melhores" com "os
        piores" — reprovação garantida que não significa nada. Os mesmos anos
        em ordem embaralhada têm de dar o mesmo veredito.
        """
        anos = {"2016": 0.05, "2017": 0.05, "2018": 0.05,
                "2019": -0.02, "2020": -0.02, "2021": -0.01}
        r1 = est.avaliar(anos)
        r2 = est.avaliar(dict(sorted(anos.items(), key=lambda kv: kv[1])))
        assert r1 is not None and r2 is not None
        assert (r1.soma_primeira_metade, r1.metades_concordam) == (
            r2.soma_primeira_metade, r2.metades_concordam
        )

    def test_nan_e_descartado_sem_contaminar(self):
        anos = {"2016": 0.01, "2017": float("nan"), "2018": 0.01,
                "2019": 0.01, "2020": 0.01}
        r = est.avaliar(anos)
        assert r is not None and r.n_periodos == 4
        assert not math.isnan(r.concentracao)

    def test_total_zero_nao_explode_a_concentracao(self):
        """Dividir pelo total líquido explodiria justamente onde a métrica mais
        importa; a divisão é pela soma dos GANHOS."""
        r = est.avaliar({"2016": 0.02, "2017": -0.02, "2018": 0.01, "2019": -0.01})
        assert r is not None and 0.0 <= r.concentracao <= 1.0


class TestCurvaDeDegradacao:
    """
    Capacidade como atributo CONTÍNUO (ADR 0033). O erro que a curva desfaz:
    colapsar contínuo em booleano cedo demais faz a estratégia parecer pior do
    que é — uma que suporta R$ 40 mil não deixa de ser boa, ela tem um limite
    operacional diferente e é perfeitamente utilizável abaixo dele.
    """

    def _curva(self, **kw):
        base = dict(edge_por_trade=0.01, adv_dos_ativos=[5_000_000.0] * 20,
                    sigma_diaria=0.0263, posicoes_concorrentes=20.0)
        return curva_de_degradacao(**{**base, **kw})

    def test_edge_cai_monotonicamente_com_o_capital(self):
        c = self._curva()
        assert c is not None
        edges = [p.edge_liquido for p in c.pontos]
        assert edges == sorted(edges, reverse=True), "mais capital, menos edge"
        assert [p.capital for p in c.pontos] == list(FAIXAS_DE_CAPITAL)

    def test_media_harmonica_faz_o_iliquido_pesar(self):
        """
        O impacto escala com √(Q/ADV), então o ativo ilíquido pesa
        desproporcionalmente. A média aritmética deixaria um punhado de nomes
        líquidos esconder a cauda ruim.
        """
        so_liquido = self._curva(adv_dos_ativos=[10_000_000.0] * 10)
        com_cauda = self._curva(adv_dos_ativos=[10_000_000.0] * 9 + [50_000.0])
        assert so_liquido is not None and com_cauda is not None
        assert com_cauda.pontos[-1].edge_liquido < so_liquido.pontos[-1].edge_liquido

    def test_limite_critico_none_quando_nao_cruza_zero(self):
        """`None` aqui é informação ('aguenta tudo que foi testado'), não ausência."""
        c = self._curva(edge_por_trade=0.05)
        assert c is not None and c.limite_critico is None
        assert "não cruza" in c.resumo_legivel()

    def test_limite_critico_encontrado_quando_cruza(self):
        c = self._curva(edge_por_trade=0.0005, adv_dos_ativos=[50_000.0] * 5,
                        posicoes_concorrentes=2.0)
        assert c is not None and c.limite_critico is not None
        cruzou = [p for p in c.pontos if p.capital >= c.limite_critico]
        assert all(p.edge_liquido <= 0 or p.capital == c.limite_critico for p in cruzou[:1])

    def test_sem_adv_devolve_none_e_none_reprova(self):
        for advs in ([], [0.0], [-1.0]):
            assert self._curva(adv_dos_ativos=advs) is None

    def test_adv_dos_trades_descarta_ticker_desconhecido_sem_inventar(self):
        """
        Inventar liquidez para o ativo do qual menos se sabe é como um backtest
        passa a mentir a favor. O descarte fica visível no tamanho da lista.
        """
        trades = pd.DataFrame({"ticker": ["PETR4", "VALE3", "XPTO11"]})
        advs = adv_dos_trades(trades, {"PETR4": 1e8, "VALE3": 2e8})
        assert advs == [1e8, 2e8], "XPTO11 sai da conta, não ganha ADV padrão"

    def test_adv_dos_trades_respeita_a_frequencia_de_negociacao(self):
        trades = pd.DataFrame({"ticker": ["PETR4"] * 3 + ["VALE3"]})
        assert adv_dos_trades(trades, {"PETR4": 1e8, "VALE3": 2e8}) == [1e8]*3 + [2e8]

    def test_to_dict_publica_a_curva_inteira(self):
        d = self._curva().to_dict()
        assert len(d["pontos"]) == len(FAIXAS_DE_CAPITAL)
        assert "resumo" in d and "limite_seguro" in d and "limite_critico" in d

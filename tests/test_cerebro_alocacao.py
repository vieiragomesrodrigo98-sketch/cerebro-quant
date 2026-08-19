"""
Alocação — quantas posições cabem, e quanto em cada.

O que estes testes protegem:

1. **O teto de carteira é sobre APOSTAS EFETIVAS, não contagem.** Medido nos 20
   pares de cripto: correlação média 0,627, e os 20 equivalem a ~2,2 apostas
   independentes. "4 posições" entrega 1,4.
2. **Existe um limite estrutural.** Com correlação alta, mais posições NÃO
   compram mais independência — e o módulo devolve o teto em vez de crescer
   para sempre.
3. **O multiplicador de confiança usa a faixa OBSERVADA, nunca 0-1.** A faixa
   alcançável de uma probabilidade calibrada é artefato do fold; normalizar por
   0-1 daria multiplicador constante quando o calibrador satura — que foi como
   o emissor v1 ficou mudo por um mês.
4. **O piso não é zero.** Posição pequena demais é dominada pelo custo fixo, e
   a decisão de não operar é do gate — não de um sizing que zera em silêncio.
"""

from __future__ import annotations

import numpy as np
import pytest

from radar.cerebro.alocacao import (
    CONFIANCA_MAXIMA,
    CONFIANCA_MINIMA,
    multiplicador_por_confianca,
    numero_efetivo_de_apostas,
    posicoes_para_apostas_efetivas,
)


class TestApostasEfetivas:
    def test_sem_correlacao_n_posicoes_sao_n_apostas(self):
        assert numero_efetivo_de_apostas(0.0, 5) == pytest.approx(5.0)

    def test_correlacao_total_e_uma_aposta_so(self):
        assert numero_efetivo_de_apostas(1.0, 20) == pytest.approx(1.0)

    def test_o_numero_medido_em_cripto(self):
        """ρ = 0,627 e 4 posições → 1,4 aposta independente. É o número que
        torna o teto por contagem a métrica errada."""
        assert numero_efetivo_de_apostas(0.627, 4) == pytest.approx(1.39, abs=0.01)

    def test_vinte_pares_de_cripto_valem_pouco_mais_de_duas_apostas(self):
        assert numero_efetivo_de_apostas(0.627, 20) == pytest.approx(1.55, abs=0.05)

    def test_matriz_de_correlacao_usa_razao_de_participacao(self):
        """Com matriz não se assume uniformidade — mede-se os autovalores."""
        identidade = np.eye(4)
        assert numero_efetivo_de_apostas(identidade) == pytest.approx(4.0)

    def test_matriz_de_tudo_correlacionado_e_uma_aposta(self):
        tudo_junto = np.ones((5, 5))
        assert numero_efetivo_de_apostas(tudo_junto) == pytest.approx(1.0, abs=0.01)

    def test_escalar_sem_n_levanta(self):
        with pytest.raises(ValueError, match="`n` é obrigatório"):
            numero_efetivo_de_apostas(0.5)


class TestPosicoesNecessarias:
    def test_sem_correlacao_e_um_para_um(self):
        assert posicoes_para_apostas_efetivas(3, correlacao=0.0) == 3

    def test_com_correlacao_precisa_de_mais_posicoes(self):
        assert posicoes_para_apostas_efetivas(2, correlacao=0.627) > 2

    def test_limite_estrutural_devolve_o_teto(self):
        """Com ρ alto, mais posições não compram mais independência — o limite
        é 1/ρ, e pedir mais que isso é impossível, não caro."""
        assert posicoes_para_apostas_efetivas(10, correlacao=0.627, teto=50) == 50

    def test_uma_aposta_e_uma_posicao(self):
        assert posicoes_para_apostas_efetivas(1, correlacao=0.9) == 1


class TestMultiplicadorPorConfianca:
    def test_no_piso_da_faixa_usa_o_minimo(self):
        assert multiplicador_por_confianca(0.138, piso=0.138, teto=0.323) == pytest.approx(
            CONFIANCA_MINIMA)

    def test_no_teto_da_faixa_usa_o_maximo(self):
        assert multiplicador_por_confianca(0.323, piso=0.138, teto=0.323) == pytest.approx(
            CONFIANCA_MAXIMA)

    def test_no_meio_da_faixa_fica_no_meio(self):
        m = multiplicador_por_confianca(0.2305, piso=0.138, teto=0.323)
        assert m == pytest.approx((CONFIANCA_MINIMA + CONFIANCA_MAXIMA) / 2, abs=0.01)

    def test_usa_a_faixa_observada_nao_zero_um(self):
        """O ponto todo: com calibrador saturando em 0,32, normalizar por 0-1
        daria multiplicador quase constante e o sizing perderia a informação."""
        saturado = multiplicador_por_confianca(0.323, piso=0.138, teto=0.323)
        se_fosse_0_1 = multiplicador_por_confianca(0.323, piso=0.0, teto=1.0)
        assert saturado > se_fosse_0_1

    def test_nunca_zera(self):
        """Posição pequena demais é dominada pelo custo fixo, e recusar é do
        gate — não de um sizing que zera em silêncio."""
        assert multiplicador_por_confianca(0.0, piso=0.138, teto=0.323) >= CONFIANCA_MINIMA

    def test_fora_da_faixa_e_limitado_nos_dois_lados(self):
        assert multiplicador_por_confianca(99.0, piso=0.1, teto=0.3) == pytest.approx(
            CONFIANCA_MAXIMA)
        assert multiplicador_por_confianca(-99.0, piso=0.1, teto=0.3) == pytest.approx(
            CONFIANCA_MINIMA)

    def test_faixa_degenerada_nao_divide_por_zero(self):
        assert multiplicador_por_confianca(0.2, piso=0.2, teto=0.2) == CONFIANCA_MAXIMA

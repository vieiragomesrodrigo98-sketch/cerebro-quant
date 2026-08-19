"""
O classificador de desfecho — Fase 3, ADR 0039.

⚠️ O que estes testes existem para impedir, e vem de erro cometido nesta mesma
sessão: um classificador que **sempre acha uma causa** é tão inútil quanto um que
nunca acha. `INCONCLUSIVO` precisa ser alcançável de verdade, e
`LEITURA_ERRADA` precisa poder sair — senão a auditoria semanal vira uma máquina
de absolver o modelo.

E a ORDEM das perguntas é testada explicitamente: ela é a doutrina, não detalhe
de implementação.
"""

from __future__ import annotations

import math

import pytest

from radar.cerebro.diagnostico_desfecho import (
    FRACAO_MOVIMENTO_OCORREU,
    FRACAO_MOVIMENTO_PARCIAL,
    SLIPPAGE_QUE_EXPLICA,
    Diagnostico,
    FatosDoDesfecho,
    consolidar,
    diagnosticar,
)


def _fatos(**kw) -> FatosDoDesfecho:
    """Tudo neutro por padrão — cada teste liga só o eixo que exercita."""
    base = dict(
        fracao_alvo_na_operacao=0.20,
        fracao_alvo_pos_saida=None,
        fracao_benchmark=None,
        slippage=None,
        regime_manteve=True,
    )
    base.update(kw)
    return FatosDoDesfecho(**base)


class TestInconclusivoEAlcancavel:
    """Sem dado não se inventa causa."""

    def test_sem_caminho_medido(self):
        assert diagnosticar(_fatos(fracao_alvo_na_operacao=None)).diagnostico is (
            Diagnostico.INCONCLUSIVO
        )

    def test_caminho_nan_tambem(self):
        laudo = diagnosticar(_fatos(fracao_alvo_na_operacao=math.nan))
        assert laudo.diagnostico is Diagnostico.INCONCLUSIVO

    def test_inconclusivo_nao_conta_contra_a_leitura(self):
        laudo = diagnosticar(_fatos(fracao_alvo_na_operacao=None))
        assert not laudo.evidencia_contra_a_leitura


class TestAsOitoClassesSaoAlcancaveis:
    """Classe que nunca sai é vocabulário morto — o defeito do portão de
    admissão desta mesma sessão, onde uma perna era inalcançável."""

    def test_leitura_correta(self):
        assert diagnosticar(_fatos(fracao_alvo_na_operacao=0.95)).diagnostico is (
            Diagnostico.LEITURA_CORRETA
        )

    def test_leitura_parcial(self):
        assert diagnosticar(_fatos(fracao_alvo_na_operacao=0.60)).diagnostico is (
            Diagnostico.LEITURA_PARCIAL
        )

    def test_leitura_errada(self):
        assert diagnosticar(_fatos(fracao_alvo_na_operacao=0.05)).diagnostico is (
            Diagnostico.LEITURA_ERRADA
        )

    def test_timing_errado(self):
        """Andou pouco dentro, alcançou o alvo na janela pós-saída."""
        laudo = diagnosticar(
            _fatos(fracao_alvo_na_operacao=0.10, fracao_alvo_pos_saida=0.98)
        )
        assert laudo.diagnostico is Diagnostico.TIMING_ERRADO

    def test_ativo_errado(self):
        """O mercado foi; este ativo não."""
        laudo = diagnosticar(
            _fatos(fracao_alvo_na_operacao=0.05, fracao_benchmark=0.80)
        )
        assert laudo.diagnostico is Diagnostico.ATIVO_ERRADO

    def test_regime_mudou(self):
        assert diagnosticar(_fatos(regime_manteve=False)).diagnostico is (
            Diagnostico.REGIME_MUDOU
        )

    def test_execucao_problematica(self):
        laudo = diagnosticar(_fatos(slippage=SLIPPAGE_QUE_EXPLICA))
        assert laudo.diagnostico is Diagnostico.EXECUCAO_PROBLEMATICA

    def test_todas_as_oito_saem_em_algum_cenario(self):
        """Guarda de cobertura do vocabulário — se alguém acrescentar uma classe
        sem caminho que a produza, este teste acusa."""
        cenarios = [
            _fatos(fracao_alvo_na_operacao=None),
            _fatos(fracao_alvo_na_operacao=0.95),
            _fatos(fracao_alvo_na_operacao=0.60),
            _fatos(fracao_alvo_na_operacao=0.05),
            _fatos(fracao_alvo_na_operacao=0.10, fracao_alvo_pos_saida=0.98),
            _fatos(fracao_alvo_na_operacao=0.05, fracao_benchmark=0.80),
            _fatos(regime_manteve=False),
            _fatos(slippage=0.01),
        ]
        saiu = {diagnosticar(f).diagnostico for f in cenarios}
        assert saiu == set(Diagnostico), f"classes sem cenário: {set(Diagnostico) - saiu}"


class TestAOrdemDasPerguntasEDoutrina:
    def test_execucao_vence_tudo(self):
        """Slippage que come um pedágio inteiro torna o resto ruído."""
        laudo = diagnosticar(
            _fatos(fracao_alvo_na_operacao=0.95, slippage=0.01, regime_manteve=False)
        )
        assert laudo.diagnostico is Diagnostico.EXECUCAO_PROBLEMATICA

    def test_regime_vem_antes_de_julgar_a_leitura(self):
        """Leitura coerente no instante não vira errada porque o mundo mudou."""
        laudo = diagnosticar(_fatos(fracao_alvo_na_operacao=0.02, regime_manteve=False))
        assert laudo.diagnostico is Diagnostico.REGIME_MUDOU

    def test_timing_vence_ativo_errado(self):
        """O caso concreto que a ordem inversa classificaria errado: movimento
        veio depois E o benchmark andou. É problema de PRAZO, não de seleção."""
        laudo = diagnosticar(
            _fatos(
                fracao_alvo_na_operacao=0.10,
                fracao_alvo_pos_saida=0.95,
                fracao_benchmark=0.80,
            )
        )
        assert laudo.diagnostico is Diagnostico.TIMING_ERRADO

    def test_pos_saida_so_vale_se_superar_o_que_houve_dentro(self):
        """Se o movimento já tinha acontecido DENTRO, não é timing — é leitura
        correta não capturada. Sem esta trava, `TIMING_ERRADO` engoliria todo
        `LEITURA_CORRETA`."""
        laudo = diagnosticar(
            _fatos(fracao_alvo_na_operacao=0.95, fracao_alvo_pos_saida=0.99)
        )
        assert laudo.diagnostico is Diagnostico.LEITURA_CORRETA


class TestFronteirasDeclaradas:
    @pytest.mark.parametrize(
        ("fracao", "esperado"),
        [
            (FRACAO_MOVIMENTO_OCORREU, Diagnostico.LEITURA_CORRETA),
            (FRACAO_MOVIMENTO_OCORREU - 0.01, Diagnostico.LEITURA_PARCIAL),
            (FRACAO_MOVIMENTO_PARCIAL, Diagnostico.LEITURA_PARCIAL),
            (FRACAO_MOVIMENTO_PARCIAL - 0.01, Diagnostico.LEITURA_ERRADA),
        ],
    )
    def test_fronteira_esta_onde_foi_declarada(self, fracao, esperado):
        assert diagnosticar(_fatos(fracao_alvo_na_operacao=fracao)).diagnostico is esperado

    def test_o_090_esta_ancorado_na_distribuicao_medida(self):
        """`partial_target` real fica entre 0,52 e 0,89; `target_hit` em 1,00.
        Acima de 0,90 só vive quem alcançou o alvo — por isso é a fronteira."""
        assert pytest.approx(0.90) == FRACAO_MOVIMENTO_OCORREU


class TestSoUmaClasseAcusaOModelo:
    """ADR 0039 §8 — a distinção que impede a auditoria de aposentar o
    especialista errado."""

    def test_apenas_leitura_errada_conta_contra(self):
        contra = {
            d
            for d in Diagnostico
            if diagnosticar(_fatos(fracao_alvo_na_operacao=0.05)).diagnostico is d
        }
        assert contra == {Diagnostico.LEITURA_ERRADA}

    @pytest.mark.parametrize(
        "fatos",
        [
            _fatos(fracao_alvo_na_operacao=0.10, fracao_alvo_pos_saida=0.98),
            _fatos(fracao_alvo_na_operacao=0.05, fracao_benchmark=0.80),
            _fatos(regime_manteve=False),
            _fatos(slippage=0.01),
            _fatos(fracao_alvo_na_operacao=None),
        ],
    )
    def test_as_outras_apontam_para_outra_camada(self, fatos):
        assert not diagnosticar(fatos).evidencia_contra_a_leitura


class TestConsolidacao:
    def test_separa_o_que_acusa_o_modelo_do_total(self):
        laudos = [
            diagnosticar(_fatos(fracao_alvo_na_operacao=0.05)),        # errada
            diagnosticar(_fatos(fracao_alvo_na_operacao=0.05)),        # errada
            diagnosticar(_fatos(regime_manteve=False)),                # regime
            diagnosticar(_fatos(fracao_alvo_na_operacao=None)),        # inconclusivo
        ]
        placar = consolidar(laudos)
        assert placar["_total"] == 4
        assert placar["_evidencia_contra_a_leitura"] == 2
        assert placar["_fracao_contra_a_leitura"] == pytest.approx(0.5)
        assert placar["_inconclusivos"] == 1

    def test_placar_vazio_nao_divide_por_zero(self):
        placar = consolidar([])
        assert placar["_total"] == 0
        assert placar["_fracao_contra_a_leitura"] == 0.0

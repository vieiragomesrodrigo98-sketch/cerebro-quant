"""
A árvore de causas do fracasso (ordem do DEV, 2026-08-04).

Os testes cobrem as duas metades da ordem, e a segunda importa tanto quanto a
primeira: (a) uma medição ruim não mata a hipótese; (b) a hipótese também não
sobrevive para sempre à base de reaberturas.

Cada ramo é testado com a INSTÂNCIA REAL que o motivou — nenhum caso é
inventado para o teste passar.
"""

from __future__ import annotations

import pytest

from radar.cerebro.diagnostico import (
    MAX_REABERTURAS_POR_CAUSA,
    Causa,
    Rodada,
    diagnosticar,
)

REGUA = dict(min_dias_distintos=15, min_eventos=50)


def _saudavel(**kw) -> Rodada:
    """Rodada que chega ao ramo terminal — o ponto de partida de cada teste."""
    base = dict(
        celula="teste", portoes_falhos=(), dias_distintos=1464, n_eventos=15483,
        t_definido=True, excesso_bruto=0.0005, excesso_liquido=0.00044,
        custo_aplicado_bps_lado=4.0, banda_custo_declarada_bps=(3.0, 10.0),
        t_ic_vs_rotulo=1.2, instrumento_aprendeu_noutro_alvo=False,
    )
    return Rodada(**{**base, **kw})


class TestOsSeisRamos:
    def test_instrumento_quebrado_vence_tudo(self):
        """Cérebro v1: 6 modelos com `num_trees=1`. Nada mais é interpretável."""
        d = diagnosticar(_saudavel(portoes_falhos=("G-P0",), dias_distintos=3), **REGUA)
        assert d.causa is Causa.INSTRUMENTO_QUEBRADO
        assert d.pode_reabrir and not d.causa.e_evidencia

    def test_amostra_curta_e_nao_estimavel(self):
        """`p9c57_triangulo_saida`: 192 células verdes, 11 dias distintos OOS."""
        d = diagnosticar(_saudavel(dias_distintos=11, n_eventos=552), **REGUA)
        assert d.causa is Causa.AMOSTRA_INSUFICIENTE
        assert not d.causa.e_evidencia, "11 dias não refutam nada"

    def test_n_abaixo_do_minimo_tambem_e_nao_estimavel(self):
        d = diagnosticar(_saudavel(n_eventos=20), **REGUA)
        assert d.causa is Causa.AMOSTRA_INSUFICIENTE

    def test_t_indefinido_e_nao_estimavel(self):
        d = diagnosticar(_saudavel(t_definido=False), **REGUA)
        assert d.causa is Causa.AMOSTRA_INSUFICIENTE

    def test_alvo_errado_quando_o_modelo_acerta_o_rotulo_e_perde_dinheiro(self):
        """`swing_v1`: Rank IC contra o próprio rótulo +0,106 (t +28,2), e perde."""
        d = diagnosticar(
            _saudavel(t_ic_vs_rotulo=28.2, excesso_liquido=-0.0011), **REGUA
        )
        assert d.causa is Causa.ALVO_ERRADO
        assert "o rótulo" in d.justificativa

    def test_dados_insuficientes_exigem_instrumento_provado_noutro_alvo(self):
        """`autoref_v1`: 1-4 árvores no alvo absoluto, 17-134 no relativo."""
        r = _saudavel(t_ic_vs_rotulo=0.2, excesso_liquido=-0.001,
                      instrumento_aprendeu_noutro_alvo=True)
        assert diagnosticar(r, **REGUA).causa is Causa.DADOS_INSUFICIENTES

    def test_sem_prova_noutro_alvo_nao_acusa_os_dados(self):
        """Sem o controle, "os dados são ruins" é indistinguível de ferramenta torta."""
        r = _saudavel(t_ic_vs_rotulo=0.2, excesso_liquido=-0.001,
                      instrumento_aprendeu_noutro_alvo=False)
        assert diagnosticar(r, **REGUA).causa is Causa.EVIDENCIA_CONTRA

    def test_ramo_terminal_e_a_unica_evidencia(self):
        """`p9c57_squeeze`: 15.483 trades, 1.464 dias, custo varrido → +0,044%."""
        d = diagnosticar(_saudavel(excesso_liquido=0.00044), **REGUA)
        assert d.causa is Causa.EVIDENCIA_CONTRA
        assert d.causa.e_evidencia
        assert not d.pode_reabrir
        assert "sobre o mercado, não sobre nós" in d.justificativa

    def test_apenas_um_dos_seis_desfechos_e_evidencia(self):
        assert [c for c in Causa if c.e_evidencia] == [Causa.EVIDENCIA_CONTRA]


class TestTravasContraReaberturaInfinita:
    """
    A árvore só é científica se puder terminar. Sem estas travas ela é uma
    máquina de nunca aceitar um negativo.
    """

    def test_custo_nao_dispara_sem_banda_declarada(self):
        """
        A trava mais importante: sem banda congelada ANTES da medição, "o custo
        estava alto demais" é ajustável até o resultado virar.
        """
        r = _saudavel(excesso_bruto=0.002, excesso_liquido=-0.001,
                      custo_aplicado_bps_lado=50.0, banda_custo_declarada_bps=None)
        assert diagnosticar(r, **REGUA).causa is not Causa.CUSTO_MAL_MODELADO

    def test_custo_dentro_da_banda_declarada_nao_e_desculpa(self):
        r = _saudavel(excesso_bruto=0.002, excesso_liquido=-0.001,
                      custo_aplicado_bps_lado=8.0, banda_custo_declarada_bps=(3.0, 10.0))
        assert diagnosticar(r, **REGUA).causa is not Causa.CUSTO_MAL_MODELADO

    def test_custo_fora_da_banda_declarada_reabre(self):
        r = _saudavel(excesso_bruto=0.002, excesso_liquido=-0.001,
                      custo_aplicado_bps_lado=50.0, banda_custo_declarada_bps=(3.0, 10.0))
        d = diagnosticar(r, **REGUA)
        assert d.causa is Causa.CUSTO_MAL_MODELADO and d.pode_reabrir

    def test_cota_esgotada_transforma_o_resultado_em_evidencia(self):
        r = _saudavel(
            portoes_falhos=("G-P0",),
            reaberturas_por_causa={Causa.INSTRUMENTO_QUEBRADO.value: MAX_REABERTURAS_POR_CAUSA},
        )
        d = diagnosticar(r, **REGUA)
        assert d.causa is Causa.INSTRUMENTO_QUEBRADO, (
            "a causa diagnosticada não pode ser apagada pela cota — perderíamos "
            "a informação de POR QUE falhou"
        )
        assert not d.pode_reabrir
        assert "diagnosticada errado" in d.justificativa

    def test_cota_de_uma_causa_nao_afeta_outra(self):
        r = _saudavel(
            dias_distintos=11,
            reaberturas_por_causa={Causa.INSTRUMENTO_QUEBRADO.value: 5},
        )
        d = diagnosticar(r, **REGUA)
        assert d.causa is Causa.AMOSTRA_INSUFICIENTE and d.pode_reabrir


class TestOrdemDaArvore:
    """A ordem vai do defeito mais nosso ao menos nosso, e não é negociável."""

    @pytest.mark.parametrize(
        "kw,esperada",
        [
            ({"portoes_falhos": ("G-P2",), "dias_distintos": 3, "t_ic_vs_rotulo": 30.0,
              "excesso_liquido": -0.01}, Causa.INSTRUMENTO_QUEBRADO),
            ({"dias_distintos": 3, "t_ic_vs_rotulo": 30.0, "excesso_liquido": -0.01},
             Causa.AMOSTRA_INSUFICIENTE),
            ({"t_ic_vs_rotulo": 30.0, "excesso_liquido": -0.01,
              "instrumento_aprendeu_noutro_alvo": True}, Causa.ALVO_ERRADO),
        ],
    )
    def test_primeiro_ramo_aplicavel_vence(self, kw, esperada):
        assert diagnosticar(_saudavel(**kw), **REGUA).causa is esperada

    def test_to_dict_expoe_e_evidencia(self):
        d = diagnosticar(_saudavel(dias_distintos=11), **REGUA).to_dict()
        assert d["e_evidencia"] is False and d["causa"] == "amostra_insuficiente"


class TestOLaudoTerminalSoAfirmaOQueFoiVerificado:
    """O ramo 6 é o que autoriza escrever que uma tese morreu.

    O defeito consertado em 2026-08-11
    ------------------------------------
    Os ramos 3 (custo) e 4 (alvo) **desligam quando o insumo não existe** — banda
    de custo não declarada, `t_ic_vs_rotulo` ausente. E mesmo assim o laudo
    terminal afirmava, textualmente, *"custo dentro da banda, alvo alinhado"*.

    Isso é a forma mais cara do erro que este módulo existe para impedir: um
    `EVIDENCIA_CONTRA` que **parece completo e não é**, e quem lê o veredito não
    tem como distinguir. O que foi ligado no `rodar_swing_v1` cai exatamente
    nesse caso — o Rank IC de lá é contra o RETORNO, não contra o rótulo —,
    então sem este conserto a primeira família diagnosticada nasceria com laudo
    falso.
    """

    def _rodada_terminal(self, **extra):
        return Rodada(
            celula="fam:c1#topo",
            portoes_falhos=(),
            dias_distintos=200,
            n_eventos=5_000,
            t_definido=True,
            excesso_liquido=-0.004,
            **extra,
        )

    def test_sem_banda_de_custo_o_laudo_declara_o_ramo_nao_avaliado(self):
        d = diagnosticar(self._rodada_terminal(), **REGUA)

        assert d.causa is Causa.EVIDENCIA_CONTRA
        assert "não foram avaliados" in d.justificativa.lower() or \
               "NÃO foram avaliados" in d.justificativa
        assert "nenhuma banda declarada" in d.justificativa
        assert "custo dentro da banda declarada" not in d.justificativa, (
            "afirmou que o custo passou num ramo que nem rodou"
        )

    def test_sem_ic_contra_o_rotulo_o_laudo_nao_afirma_alvo_alinhado(self):
        d = diagnosticar(self._rodada_terminal(), **REGUA)

        assert "alvo (IC contra o próprio rótulo não medido)" in d.justificativa
        assert "alvo alinhado" not in d.justificativa

    def test_com_os_dois_insumos_o_laudo_e_completo(self):
        """O par positivo: quando tudo foi avaliado, o laudo pode afirmar."""
        d = diagnosticar(
            self._rodada_terminal(
                custo_aplicado_bps_lado=8.0,
                banda_custo_declarada_bps=(5.0, 12.0),
                t_ic_vs_rotulo=0.4,
            ),
            **REGUA,
        )

        assert d.causa is Causa.EVIDENCIA_CONTRA
        assert "Todos os ramos foram avaliados" in d.justificativa
        assert "Ressalva" not in d.justificativa
        assert "custo dentro da banda declarada" in d.justificativa
        assert "alvo alinhado" in d.justificativa

    def test_a_causa_nao_muda_com_o_conserto(self):
        """A trava: o conserto é de LAUDO, não de classificação.

        Se ele tivesse mudado a causa, teria mexido em que células o projeto
        considera refutadas — e isso exigiria remedir, não reescrever texto.
        """
        sem = diagnosticar(self._rodada_terminal(), **REGUA)
        com = diagnosticar(
            self._rodada_terminal(
                custo_aplicado_bps_lado=8.0,
                banda_custo_declarada_bps=(5.0, 12.0),
                t_ic_vs_rotulo=0.4,
            ),
            **REGUA,
        )

        assert sem.causa is com.causa is Causa.EVIDENCIA_CONTRA
        assert sem.pode_reabrir == com.pode_reabrir is False

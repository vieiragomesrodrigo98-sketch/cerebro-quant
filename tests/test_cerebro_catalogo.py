"""
O catálogo de famílias — a cobertura dos 4 perfis, verificada por teste.

**Os 4 perfis SÃO o produto** (ordem do DEV): resultado negativo elimina
*aquela estratégia naquele perfil*, jamais o perfil. Logo a ausência de um
perfil no catálogo é **dívida declarada**, não escolha — e um teste que a
declara é melhor que um comentário que a esquece.
"""

from __future__ import annotations

import pytest

from radar.cerebro import catalogo
from radar.cerebro.familia import PERFIS, REGISTRO, hipoteses_duplicadas
from radar.mie.dataset import COLUNAS_SELECAO_CRIPTO


class TestCobertura:
    def test_day_e_position_estao_registradas(self):
        """As duas que faltavam dos 4 perfis."""
        assert catalogo.DAY_CRIPTO_V1.nome in REGISTRO
        assert catalogo.POSITION_CRIPTO_V1.nome in REGISTRO

    def test_cada_familia_declara_um_perfil_valido(self):
        for f in REGISTRO.values():
            assert f.perfil in PERFIS

    def test_nenhuma_hipotese_e_medida_duas_vezes(self):
        """Nada pode entrar DUAS vezes no denominador do FDR sem informação nova.

        ⚠️ Substitui `test_nenhum_perfil_tem_duas_familias_no_mesmo_mercado`,
        que exigia um-por-`(perfil, mercado)`. Aquela regra proibia o que o
        ADR 0032 §2 exige: os especialistas são separados por
        `motor × mercado × horizonte`, e o Router *"só existe depois de ≥2
        especialistas com zona de validade provada"* — com um-por-`(perfil,
        mercado)` nunca haveria dois especialistas de swing em cripto para o
        Router orquestrar.

        A regra vive em `radar.cerebro.familia`, não aqui, porque é invariante
        do catálogo. O motivo dela e o histórico da correção estão no docstring
        de `hipoteses_duplicadas` — inclusive a ressalva de que a correção foi
        feita por quem precisava dela.
        """
        colisoes = hipoteses_duplicadas(REGISTRO.values())
        assert not colisoes, "; ".join(
            f"{nova} repete {velha} ({onde})" for nova, velha, onde in colisoes
        )


class TestDay:
    def test_horizonte_de_horas_nao_invade_o_swing(self):
        """168h é uma semana — território de Swing. Dois perfis medindo o mesmo
        horizonte é a duplicação que o ADR 0032 §2 impede."""
        assert catalogo.DAY_CRIPTO_V1.horizontes == (24, 72)
        assert 168 not in catalogo.DAY_CRIPTO_V1.horizontes

    def test_nao_normaliza_data_por_ser_intradiario(self):
        assert catalogo.DAY_CRIPTO_V1.normalizar_data is False

    def test_declara_a_lacuna_das_features_nativas(self):
        """Usa as 38 da B3 porque o fluxo agressor só existe no store de 1m —
        e o veredito precisa dizer isso."""
        assert catalogo.DAY_CRIPTO_V1.colunas_selecao == tuple(COLUNAS_SELECAO_CRIPTO)
        assert "lacuna_declarada" in catalogo.DAY_CRIPTO_V1.notas


class TestPosition:
    def test_horizonte_de_semanas_a_meses(self):
        assert catalogo.POSITION_CRIPTO_V1.horizontes == (60, 120, 250)
        assert catalogo.POSITION_CRIPTO_V1.unidade_horizonte == "dias_corridos"

    def test_holdout_none_usa_o_default_correto_para_diario(self):
        """12 meses é certo em dado diário; só o intradiário precisa de outro."""
        assert catalogo.POSITION_CRIPTO_V1.holdout is None
        assert catalogo.POSITION_CRIPTO_V1.holdout_amostrado is None

    def test_ano_de_teste_deixa_um_ciclo_completo_no_treino(self):
        """2017-2019 de treino inclui o bear de 2018 — treinar só em bull
        ensinaria que tudo sobe."""
        assert catalogo.POSITION_CRIPTO_V1.ano_primeiro_teste == 2020


class TestCustoDeclaradoAntes:
    @pytest.mark.parametrize("f", [catalogo.DAY_CRIPTO_V1, catalogo.POSITION_CRIPTO_V1])
    def test_venue_e_tipo_de_ordem_vem_declarados(self, f):
        """Custo escolhido depois do resultado é o defeito, não a correção."""
        assert (f.venue, f.tipo_ordem) == ("perpetuo", "maker")

    @pytest.mark.parametrize("f", [catalogo.DAY_CRIPTO_V1, catalogo.POSITION_CRIPTO_V1])
    def test_embargo_cobre_o_maior_horizonte(self, f):
        assert f.embargo >= max(f.horizontes)

    @pytest.mark.parametrize("f", [catalogo.DAY_CRIPTO_V1, catalogo.POSITION_CRIPTO_V1])
    def test_aponta_para_um_pre_registro(self, f):
        """Família sem pré-registro não mede."""
        assert f.pre_registro.startswith("docs/estudos/PRE_REGISTRO_")

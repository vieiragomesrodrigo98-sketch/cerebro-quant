"""
O contrato de família — o que impede o Cérebro de ser alterado a cada perfil.

O que estes testes protegem:

1. **Família é declaração congelada.** Mudar um valor cria família NOVA
   (ADR 0032 §2); `frozen=True` fecha a porta para o monkeypatch que existia
   antes deste módulo (`swing._CFG["cripto"]` sendo mutado de fora).
2. **Embargo ≥ maior horizonte**, senão o treino vê o desfecho do teste. É
   verificado na construção, não na revisão.
3. **A conversão para a unidade amostrada é do contrato, não do chamador.**
   Foi um holdout declarado em barras sobre um dataset amostrado que devolveu
   zero folds em silêncio — duas vezes.
4. **Nome duplicado levanta.** Duas famílias homônimas colidiriam no Evidence
   Map, e duas hipóteses com o mesmo nome exibindo os números da primeira é o
   colapso de identidade que o mapa existe para expor.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from radar.cerebro.familia import (
    PERFIS,
    REGISTRO,
    Familia,
    familia,
    registrar,
)


def _f(**kw) -> Familia:
    base = dict(
        nome="teste_v1", perfil="swing", mercado="cripto",
        store=Path("data/x.parquet"), colunas_selecao=("a", "b"),
        horizontes=(10,), embargo=10, ano_primeiro_teste=2025,
    )
    return Familia(**{**base, **kw})


class TestDeclaracao:
    def test_familia_e_imutavel(self):
        """`frozen=True` fecha a porta do monkeypatch que existia antes."""
        from dataclasses import FrozenInstanceError

        f = _f()
        with pytest.raises(FrozenInstanceError):
            f.embargo = 999  # type: ignore[misc]

    def test_perfil_desconhecido_levanta(self):
        with pytest.raises(ValueError, match="perfil desconhecido"):
            _f(perfil="daytrade")

    def test_os_quatro_perfis_sao_aceitos(self):
        for p in PERFIS:
            assert _f(perfil=p).perfil == p

    def test_unidade_de_horizonte_desconhecida_levanta(self):
        with pytest.raises(ValueError, match="unidade_horizonte desconhecida"):
            _f(unidade_horizonte="semanas")

    def test_embargo_menor_que_horizonte_levanta(self):
        with pytest.raises(ValueError, match="o treino veria o desfecho"):
            _f(horizontes=(240,), embargo=60)

    def test_venue_sem_tipo_de_ordem_levanta(self):
        """Custo pela metade é pior que custo default: some sem avisar."""
        with pytest.raises(ValueError, match="declare os dois ou nenhum"):
            _f(venue="perpetuo")

    def test_passo_de_amostragem_zero_levanta(self):
        with pytest.raises(ValueError, match="passo_amostragem"):
            _f(passo_amostragem=0)


class TestUnidadeAmostrada:
    """A conversão que faltava, e cuja ausência devolvia zero folds em silêncio."""

    def test_sem_amostragem_a_unidade_nao_muda(self):
        f = _f(embargo=21, holdout=252)
        assert f.embargo_amostrado == 21
        assert f.holdout_amostrado == 252

    def test_com_amostragem_converte_os_dois(self):
        f = _f(horizontes=(240,), embargo=240, holdout=43_200, passo_amostragem=60)
        assert f.embargo_amostrado == 4      # 240 min / 60
        assert f.holdout_amostrado == 720    # 43.200 min / 60

    def test_nunca_converte_para_zero(self):
        """Embargo zero desligaria a proteção sem avisar."""
        f = _f(horizontes=(1,), embargo=1, passo_amostragem=60)
        assert f.embargo_amostrado == 1

    def test_holdout_none_continua_none(self):
        """`None` = usar o default do pipeline, e isso precisa atravessar."""
        assert _f(holdout=None, passo_amostragem=60).holdout_amostrado is None


class TestCfgDoPipeline:
    def test_entrega_a_unidade_amostrada_nao_a_declarada(self):
        f = _f(horizontes=(240,), embargo=240, passo_amostragem=60,
               unidade_horizonte="barras", segundos_por_barra=60)
        assert f.cfg_do_pipeline()["embargo"] == 4

    def test_colunas_de_treino_caem_para_as_de_selecao_quando_ausentes(self):
        f = _f(colunas_selecao=("a", "b"))
        assert f.cfg_do_pipeline()["colunas_x"] == ("a", "b")

    def test_colunas_de_treino_proprias_sao_preservadas(self):
        """O braço auto-referente treina em espaço maior que o de seleção."""
        f = _f(colunas_selecao=("a",), colunas_treino=("a", "b", "c"))
        assert f.cfg_do_pipeline()["colunas_x"] == ("a", "b", "c")


class TestRegistro:
    def test_registrar_e_recuperar(self):
        REGISTRO.pop("reg_v1", None)
        f = registrar(_f(nome="reg_v1"))
        assert familia("reg_v1") is f
        REGISTRO.pop("reg_v1", None)

    def test_nome_duplicado_levanta(self):
        REGISTRO.pop("dup_v1", None)
        registrar(_f(nome="dup_v1"))
        with pytest.raises(ValueError, match="já registrada"):
            registrar(_f(nome="dup_v1"))
        REGISTRO.pop("dup_v1", None)

    def test_desconhecida_levanta_listando_o_que_existe(self):
        with pytest.raises(KeyError, match="família desconhecida"):
            familia("nao_existe_v9")

"""
A ficha de oportunidade — e as duas travas que a impedem de virar decoração.

Ordem do DEV (13/08): *"o Cérebro não deveria produzir `SUI = 0.82` sozinho"*.
A ficha é `ativo + contexto + relação`, e o que a torna útil não é ter muitos
campos: é **publicar o buraco em vez de preenchê-lo**.

Duas travas, e as duas já foram violadas por este projeto em outra forma:

1. **Eixo omitido esconde o buraco.** Ficha que só traz o que tem medido faz
   *"não há dado"* e *"esqueci de calcular"* ficarem indistinguíveis — a mesma
   assimetria que manteve `NO_TRADE` e `SEM_DETECTOR` confundidos.
2. **Ausência sem valor, jamais `0.0`.** Publicar zero para o que não foi medido
   transforma ausência em afirmação. É a razão de `ContextoDeMercado.confianca`
   ser `None`.
"""

from __future__ import annotations

import pytest

from radar.cerebro.dossie import (
    DIMENSOES_OBRIGATORIAS,
    Dimensao,
    EstadoDaDimensao,
    FichaDeOportunidade,
    montar_ficha,
)


def _ficha(**kw):
    base = {
        "ativo": "SUIUSDT",
        "instante": "2026-07-29",
        "particao": "cripto:bull_lateral",
        "regime": "bull_lateral",
        "posto_no_universo": 0.97,
        "valores": {
            "beta_63": 1.4,
            "largura_banda_20_2": 0.11,
            "razao_atr_5_20": 1.3,
            "faixa_liquidez": "alta",
            "atr_14_pct": 0.06,
        },
    }
    return montar_ficha(**{**base, **kw})


class TestOsTresEstadosNaoSaoIntercambiaveis:
    """`NAO_DISPONIVEL` (não há fonte) e `NAO_MEDIDO` (há fonte, ninguém mediu)
    pedem trabalho oposto — confundi-los faz alguém procurar dado que não existe
    ou desistir de dado que existe."""

    def test_setor_e_indisponivel_porque_a_fonte_nao_existe(self) -> None:
        d = _ficha().dimensoes["posto_no_setor"]
        assert d.estado is EstadoDaDimensao.NAO_DISPONIVEL
        assert "setor" in (d.motivo or "")
        assert d.valor is None, "ausência publicada como valor vira afirmação"

    def test_expectativa_sem_leitura_e_nao_medida_e_nao_indisponivel(self) -> None:
        """Distinção que decide o próximo passo: os payoffs ESTÃO nos trades
        OOS; o que falta é a família ter sido medida."""
        d = _ficha().dimensoes["expectativa"]
        assert d.estado is EstadoDaDimensao.NAO_MEDIDO
        assert "CEREBRO_EXPECTATIVA_MEDIDA_POR_FAMILIA01" in (d.motivo or "")

    def test_expectativa_medida_entra_como_medido(self) -> None:
        d = _ficha(ev_medido=-0.00596).dimensoes["expectativa"]
        assert d.estado is EstadoDaDimensao.MEDIDO
        assert d.valor == pytest.approx(-0.00596)

    def test_ausencia_sem_motivo_e_recusada(self) -> None:
        with pytest.raises(ValueError, match="sem motivo"):
            Dimensao(nome="x", estado=EstadoDaDimensao.NAO_MEDIDO)

    def test_medido_sem_valor_e_recusado(self) -> None:
        with pytest.raises(ValueError, match="sem valor"):
            Dimensao(nome="x", estado=EstadoDaDimensao.MEDIDO)


class TestEixoOmitidoEscondeOBuraco:
    def test_ficha_incompleta_levanta(self) -> None:
        with pytest.raises(ValueError, match="omite"):
            FichaDeOportunidade(
                ativo="X", instante="2026-07-29", particao="cripto:bear",
                dimensoes={"regime": Dimensao("regime", EstadoDaDimensao.MEDIDO, "bear")},
            )

    def test_toda_ficha_declara_os_dez_eixos(self) -> None:
        assert set(_ficha().dimensoes) == set(DIMENSOES_OBRIGATORIAS)
        assert len(DIMENSOES_OBRIGATORIAS) == 10


class TestCoberturaVemAntesDaLeitura:
    """Ficha com 3 de 10 eixos não sustenta comparação com uma de 9 — mesma
    razão de a cobertura vir antes da repartição no diagnóstico de desfechos."""

    def test_cobertura_conta_so_o_medido(self) -> None:
        assert _ficha().medidas == 7
        assert _ficha().cobertura == pytest.approx(0.7)

    def test_a_expectativa_medida_sobe_a_cobertura(self) -> None:
        assert _ficha(ev_medido=-0.006).cobertura == pytest.approx(0.8)

    def test_coluna_ausente_derruba_a_cobertura_em_vez_de_virar_zero(self) -> None:
        f = _ficha(valores={})
        assert f.cobertura < 0.3
        for eixo in ("vs_benchmark", "momento", "liquidez", "risco", "estrutura"):
            d = f.dimensoes[eixo]
            assert d.estado is EstadoDaDimensao.NAO_MEDIDO
            assert d.valor is None

    def test_regime_desconhecido_nao_conta_como_medido(self) -> None:
        f = _ficha(regime="desconhecido")
        assert f.dimensoes["regime"].estado is EstadoDaDimensao.NAO_MEDIDO

    def test_ativo_fora_da_secao_nao_recebe_posto_zero(self) -> None:
        f = _ficha(posto_no_universo=None)
        assert f.dimensoes["posto_no_universo"].valor is None


class TestAFichaNaoJulga:
    """Mesma trava de `deteccao.Situacao`: a ficha REGISTRA onde o ativo está,
    jamais se isso é bom. Julgamento embutido na descrição destrói a
    decomposição de que a autópsia depende."""

    def test_o_ev_e_da_estrategia_e_nao_do_ativo(self) -> None:
        """Dois ativos diferentes da mesma célula recebem o MESMO `ev` — e isso
        é o desenho, não um bug. O que discrimina entre eles é
        `posto_no_universo`. Se um dia o `ev` variar por ativo, a expectativa
        virou score e o guardrail *resultado da estratégia ≠ propriedade do
        ativo* caiu."""
        a = _ficha(ativo="SUIUSDT", posto_no_universo=0.97, ev_medido=-0.006)
        b = _ficha(ativo="DOGEUSDT", posto_no_universo=0.02, ev_medido=-0.006)
        assert a.dimensoes["expectativa"].valor == b.dimensoes["expectativa"].valor
        assert a.dimensoes["posto_no_universo"].valor != b.dimensoes["posto_no_universo"].valor

    def test_nenhum_eixo_carrega_juizo_no_nome(self) -> None:
        proibidos = ("bom", "forte", "alta", "baixa", "otim", "confianca", "score")
        for eixo in DIMENSOES_OBRIGATORIAS:
            assert not any(p in eixo.lower() for p in proibidos), (
                f"eixo {eixo!r} carrega juízo — a ficha é descritiva"
            )

    def test_a_ficha_nao_tem_campo_de_confianca_nem_de_veredito(self) -> None:
        """O exemplo enviado pelo DEV termina com `OPORTUNIDADE: Alta` e
        `CONFIANÇA: 0.78`. Os dois exigiriam números que nenhuma família publica
        — e ficha com juízo fabricado é pior que ficha incompleta, porque PARECE
        pronta."""
        f = _ficha()
        for proibido in ("confianca", "oportunidade", "veredito", "score", "rank"):
            assert not hasattr(f, proibido), f"ficha ganhou o atributo {proibido!r}"
            assert proibido not in f.dimensoes, f"ficha ganhou o eixo {proibido!r}"

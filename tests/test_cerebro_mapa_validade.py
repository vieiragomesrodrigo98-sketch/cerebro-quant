"""
O Evidence Map — os 4 estados de validade (taxonomia do DEV, 04/08/2026).

O que estes testes protegem, em ordem de importância:

1. **`VALIDADO` é a única porta para capital**, e ela exige os seis critérios.
2. **`REFUTADO` exige PODER.** Sem teste de poder, um negativo vira
   `NAO_COMPROVADO` — falha para o lado de não enterrar hipótese boa.
3. **`NAO_ESTIMAVEL` nunca é evidência contra.**
4. **Capacidade não medida bloqueia `VALIDADO`** — alocar capital sem saber
   quanto a estratégia absorve é chutar o tamanho.

Os casos usam os números REAIS medidos no Ledger Técnico em 2026-08-04.
"""

from __future__ import annotations

import math

import pytest

from radar.cerebro.mapa_validade import (
    _ACAO_QUADRANTE,
    CATEGORIA_DO_CRITERIO,
    PRIORIDADE_DE_BLOQUEIO,
    QUADRANTES,
    Bloqueio,
    Dimensao,
    Direcao,
    Estado,
    EvidenciaOOS,
    PerfilDoProduto,
    classificar,
    equilibrio_slippage_bps_lado,
    erro_padrao,
    montar_mapa,
    persistencia_de,
)


def _ev(**kw) -> EvidenciaOOS:
    """Evidência que passa em TUDO — o ponto de partida de cada teste."""
    base = dict(
        hipotese="teste", mercado="cripto", horizonte=10,
        trades=3000, dias_independentes=400, excesso_liquido=0.004, t_nw=3.1,
        anos_positivos=8, anos_testados=10, efeito_minimo_relevante=0.001,
        equilibrio_slippage_bps=12.0, estavel_oos=True,
        capacidade_estimada=5_000_000.0, capital_alvo=100_000.0,
        direcao=Direcao.LONG,
        risco_dentro_da_tolerancia=True, max_drawdown=0.10,
    )
    return EvidenciaOOS(**{**base, **kw})


class TestValidado:
    def test_passa_nos_seis_criterios(self):
        v = classificar(_ev())
        assert v.estado is Estado.VALIDADO
        assert v.estado.recebe_capital
        assert v.criterios_faltantes == ()

    @pytest.mark.parametrize(
        "kw,faltante",
        [
            ({"t_nw": 1.9}, "OOS"),
            ({"excesso_liquido": -0.001}, "custos"),
            ({"equilibrio_slippage_bps": None}, "slippage de equilíbrio NÃO MEDIDO"),
            ({"equilibrio_slippage_bps": 2.0}, "margem mínima"),
            ({"anos_positivos": 3, "anos_testados": 10}, "persistência 30% < 70%"),
            ({"persistencia": None, "anos_positivos": 0, "anos_testados": 0},
             "persistência anual NÃO MEDIDA"),
            ({"estavel_oos": False}, "OOS instável"),
            ({"estavel_oos": None}, "OOS NÃO MEDIDA"),
            ({"capacidade_estimada": None}, "capacidade: NÃO MEDIDA"),
            ({"capacidade_estimada": 1_000.0}, "escala: NÃO ATENDE AO GATE"),
            ({"risco_dentro_da_tolerancia": None}, "risco: drawdown NÃO MEDIDO"),
            ({"risco_dentro_da_tolerancia": False}, "fora da tolerância"),
        ],
    )
    def test_qualquer_criterio_faltando_barra_o_capital(self, kw, faltante):
        v = classificar(_ev(**kw))
        assert not v.estado.recebe_capital
        assert any(faltante in c for c in v.criterios_faltantes), v.criterios_faltantes

    def test_capacidade_nao_medida_e_bloqueio_isolado(self):
        """Tudo perfeito, só a capacidade desconhecida — ainda assim não aloca."""
        v = classificar(_ev(capacidade_estimada=None))
        assert v.estado is not Estado.VALIDADO
        assert v.criterios_faltantes == ("capacidade: NÃO MEDIDA",)
        assert v.dimensoes["risco_drawdown"] is Dimensao.SIM, (
            "a dimensão nova não pode contaminar o bloqueio isolado"
        )


class TestRefutadoExigePoder:
    """A distinção que impede a taxonomia de virar carimbo."""

    def test_squeeze_real_nao_tem_poder_apesar_da_maior_amostra_do_projeto(self):
        """
        `p9c57_squeeze`: 15.483 trades, 1.464 dias, líquido +0,044%, t +0,14.

        `ep` = 0,311% → só detectaria **0,622%/trade**, 14× o efeito medido. A
        MAIOR amostra do projeto ainda assim não tem poder para refutar — e
        este teste existe para que ninguém volte a escrever que ela refutou.
        (Eu mesmo escrevi isso e errei uma casa decimal no `ep`; foi este teste
        que pegou.)
        """
        v = classificar(_ev(
            hipotese="p9c57_squeeze", trades=15483, dias_independentes=1464,
            excesso_liquido=0.00044, t_nw=0.14, anos_positivos=9, anos_testados=11,
            efeito_minimo_relevante=0.001, estavel_oos=True,
        ))
        assert v.estado is Estado.NAO_COMPROVADO
        assert v.poder_suficiente is False
        assert not v.estado.e_evidencia_contra
        assert v.efeito_minimo_detectavel == pytest.approx(0.00628, abs=1e-4)

    def test_refuta_quando_o_efeito_relevante_e_maior_que_o_detectavel(self):
        """
        A outra ponta: se o mínimo economicamente relevante fosse 1%/trade, a
        mesma amostra do `squeeze` (detecta 0,63%) TERIA poder — e aí o
        negativo refuta de verdade.
        """
        v = classificar(_ev(
            hipotese="p9c57_squeeze", trades=15483, dias_independentes=1464,
            excesso_liquido=0.00044, t_nw=0.14, anos_positivos=9, anos_testados=11,
            efeito_minimo_relevante=0.01,
        ))
        assert v.estado is Estado.REFUTADO
        assert v.poder_suficiente is True

    def test_fator_de_amostra_faltante_e_2_sobre_t_ao_quadrado(self):
        """
        `efeito_detectavel / efeito_medido = 2/|t|` — logo o fator de amostra
        que falta é `(2/t)²`. É a identidade que transforma "não deu" em um
        número acionável: `contracao_atr_saida` precisa de 2,2× mais dias,
        `squeeze` de 203×.
        """
        for t, fator in [(1.34, 2.23), (0.68, 8.65), (0.14, 204.1)]:
            v = classificar(_ev(t_nw=t, excesso_liquido=0.005,
                                efeito_minimo_relevante=0.001))
            assert v.efeito_minimo_detectavel is not None
            razao = v.efeito_minimo_detectavel / 0.005
            assert razao == pytest.approx(2.0 / t, rel=1e-6)
            assert (2.0 / t) ** 2 == pytest.approx(fator, rel=0.01)

    def test_contracao_atr_real_nao_tinha_poder_logo_e_nao_comprovado(self):
        """
        `p9c59_contracao_atr_saida`: 2.971 trades, 160 dias, +0,597%, t +1,34.
        `ep` ≈ 0,45% → só detectaria 0,89%, muito acima do mínimo relevante.
        Mesmo `t` abaixo da régua, isto NÃO refuta.
        """
        v = classificar(_ev(
            hipotese="p9c59_contracao_atr_saida", trades=2971, dias_independentes=160,
            excesso_liquido=0.00597, t_nw=1.34, anos_positivos=10, anos_testados=16,
            efeito_minimo_relevante=0.001,
        ))
        assert v.estado is Estado.NAO_COMPROVADO
        assert v.poder_suficiente is False
        assert not v.estado.e_evidencia_contra
        assert v.estado.opera_em_paper, "é assim que ele sai do limbo"

    def test_sem_efeito_minimo_declarado_nunca_refuta(self):
        """
        A trava contra enterrar por conveniência: sem o efeito mínimo relevante
        declarado ANTES, não há teste de poder — e o negativo degrada para
        NAO_COMPROVADO, nunca para REFUTADO.
        """
        v = classificar(_ev(t_nw=0.1, excesso_liquido=0.00001,
                            efeito_minimo_relevante=None))
        assert v.estado is Estado.NAO_COMPROVADO
        assert "não declarado" in v.justificativa

    def test_t_negativo_com_poder_e_refutado(self):
        v = classificar(_ev(t_nw=-2.5, excesso_liquido=-0.002,
                            efeito_minimo_relevante=0.01))
        assert v.estado is Estado.REFUTADO


class TestNaoEstimavel:
    def test_triangulo_saida_real_e_nao_estimavel(self):
        """
        `p9c57_triangulo_saida`: produzia 192 das 252 verdes in-sample do
        projeto e tem **11 dias independentes** OOS. Nunca foi refutada.
        """
        v = classificar(_ev(hipotese="p9c57_triangulo_saida", trades=552,
                            dias_independentes=11, t_nw=float("nan")))
        assert v.estado is Estado.NAO_ESTIMAVEL
        assert not v.estado.e_evidencia_contra
        assert not v.estado.recebe_capital

    def test_poucos_trades_e_nao_estimavel(self):
        assert classificar(_ev(trades=20)).estado is Estado.NAO_ESTIMAVEL

    def test_t_indefinido_e_nao_estimavel(self):
        assert classificar(_ev(t_nw=float("nan"))).estado is Estado.NAO_ESTIMAVEL

    def test_nao_estimavel_vence_mesmo_com_retorno_enorme(self):
        """Retorno gigante em amostra minúscula é o padrão do sobreajuste."""
        v = classificar(_ev(trades=552, dias_independentes=11, excesso_liquido=0.02591,
                            t_nw=float("nan")))
        assert v.estado is Estado.NAO_ESTIMAVEL
        assert not v.estado.recebe_capital


class TestPropriedadesDoEstado:
    def test_apenas_validado_recebe_capital(self):
        assert [e for e in Estado if e.recebe_capital] == [Estado.VALIDADO]

    def test_apenas_refutado_e_evidencia_contra(self):
        assert [e for e in Estado if e.e_evidencia_contra] == [Estado.REFUTADO]

    def test_apenas_nao_comprovado_vai_para_paper(self):
        assert [e for e in Estado if e.opera_em_paper] == [Estado.NAO_COMPROVADO]

    def test_toda_acao_esta_definida(self):
        for e in Estado:
            assert e.acao and isinstance(e.acao, str)


class TestErroPadrao:
    def test_derivado_de_excesso_e_t(self):
        assert erro_padrao(0.004, 2.0) == pytest.approx(0.002)

    @pytest.mark.parametrize("exc,t", [(0.004, 0.0), (0.004, float("nan")),
                                       (float("nan"), 2.0)])
    def test_indefinido_devolve_none(self, exc, t):
        assert erro_padrao(exc, t) is None


class TestMapaParticiona:
    def test_pool_so_tem_validadas_e_particao_e_completa(self):
        m = montar_mapa([
            _ev(hipotese="ok"),
            _ev(hipotese="fraca", t_nw=1.0, efeito_minimo_relevante=None),
            _ev(hipotese="curta", trades=10),
            _ev(hipotese="morta", t_nw=0.1, excesso_liquido=0.00001,
                efeito_minimo_relevante=0.5),
        ])
        assert [v.hipotese for v in m.pool] == ["ok"]
        r = m.resumo()
        assert (r["validado"], r["nao_comprovado"], r["nao_estimavel"],
                r["refutado"], r["total"]) == (1, 1, 1, 1, 4)
        assert (len(m.pool) + len(m.em_paper) + len(m.em_observacao)
                + len(m.fora)) == len(m.vereditos), "os 4 estados particionam o mapa"

    def test_to_dict_serializa_a_decisao(self):
        d = classificar(_ev(trades=10)).to_dict()
        assert d["estado"] == "nao_estimavel" and d["recebe_capital"] is False
        assert not math.isnan(0.0)  # sanidade do import


class TestPersistenciaNaoMedidaNuncaPassa:
    """
    O buraco que `EvidenciaOOS.persistencia` fecha: antes, fonte sem informação
    de anos fazia o critério de estabilidade ser **pulado em silêncio**, e a
    hipótese saía parecendo melhor do que foi medida. "Não medido" tratado como
    "passou" é como um backtest passa a mentir a favor.
    """

    def test_derivada_dos_anos_quando_a_fonte_nao_traz(self):
        assert persistencia_de(_ev(anos_positivos=8, anos_testados=10)) == pytest.approx(0.8)

    def test_explicita_vence_a_derivada(self):
        e = _ev(persistencia=0.42, anos_positivos=8, anos_testados=10)
        assert persistencia_de(e) == pytest.approx(0.42)

    def test_sem_nenhum_dos_dois_e_none_e_o_criterio_falha(self):
        e = _ev(persistencia=None, anos_positivos=0, anos_testados=0)
        assert persistencia_de(e) is None
        v = classificar(e)
        assert v.estado is not Estado.VALIDADO
        assert "persistência anual NÃO MEDIDA" in "; ".join(v.criterios_faltantes)

    def test_persistencia_explicita_baixa_reprova(self):
        v = classificar(_ev(persistencia=0.64, anos_positivos=0, anos_testados=0))
        assert any("64%" in c for c in v.criterios_faltantes)


class TestMapaRealGerado:
    """
    Invariantes sobre `data/mapa_validade.json` — o artefato que o Cérebro vai
    ler. Falham se o builder regredir.
    """

    @pytest.fixture(scope="class")
    def mapa(self):
        import json
        from pathlib import Path
        p = Path(__file__).resolve().parents[2] / "data" / "mapa_validade.json"
        if not p.exists():
            pytest.skip("mapa ainda não gerado (scripts/build_mapa_validade.py)")
        return json.loads(p.read_text(encoding="utf-8"))

    def test_ids_de_hipotese_sao_unicos(self, mapa):
        """
        Nome duplicado fez a 1ª geração exibir duas medições diferentes com os
        números da primeira — colapso de identidade, o bug que este projeto
        repete.
        """
        ids = [h["hipotese"] for h in mapa["hipoteses"]]
        assert len(ids) == len(set(ids)), sorted(
            i for i in set(ids) if ids.count(i) > 1
        )

    def test_pool_so_tem_validadas(self, mapa):
        pool = {h["hipotese"] for h in mapa["hipoteses"] if h["recebe_capital"]}
        assert pool == set(mapa["pool"])
        for h in mapa["hipoteses"]:
            if h["recebe_capital"]:
                assert h["estado"] == "validado"

    def test_lacunas_declaram_o_que_fechou_e_o_que_segue_aberto(self, mapa):
        """
        As lacunas deixaram de ser todas ABERTAS: três foram fechadas nesta
        sessão. O relatório precisa dizer QUAIS, e continuar nomeando as que
        seguem abertas — sumir com uma lacuna aberta seria fingir cobertura.
        """
        texto = " ".join(mapa["lacunas_sistemicas"])
        assert "FECHADA" in texto and "ABERTA" in texto
        assert "drawdown" in texto, "a lacuna que o DEV pediu e não foi medida"
        assert "43c4a030" in texto, "a régua pré-registrada tem de ser rastreável"

    def test_toda_hipotese_reporta_todas_as_dimensoes(self, mapa):
        """
        Sete desde que  entrou (ADR 0033). O conjunto é
        verificado por igualdade, não por inclusão: dimensão que somem do
        relatório é informação perdida sem ninguém notar.
        """
        esperadas = {
            "evidencia_oos", "retorno_liquido", "margem_de_custo",
            "persistencia", "estabilidade_oos", "capacidade", "risco_drawdown",
        }
        for h in mapa["hipoteses"]:
            if h["estado"] != "nao_estimavel":
                assert set(h["dimensoes"]) == esperadas, h["hipotese"]

    def test_toda_refutada_teve_poder_de_verdade(self, mapa):
        """
        Este teste antes afirmava `refutadas == []` — e estava certo ENQUANTO
        ninguém tinha declarado o efeito mínimo relevante. A régua foi
        pré-registrada (commit `43c4a030`, ANTES desta geração), então refutar
        passou a ser possível. O que ele guarda agora é mais forte: **toda**
        refutação tem de ser sustentada por poder, nunca por `t` baixo.
        """
        for h in mapa["hipoteses"]:
            if h["estado"] == "refutado":
                assert h["poder_suficiente"] is True, h["hipotese"]
                assert h["efeito_minimo_detectavel"] is not None
                assert "tinha poder" in h["justificativa"], h["hipotese"]


class TestDoisEixosOrtogonais:
    """
    Evidência e elegibilidade operacional são eixos SEPARADOS.

    O achado que forçou isto: as duas hipóteses com maior `t` OOS do projeto
    são **short** num produto **long-only**. Somá-las à régua de evidência as
    marcaria "reprovadas", o que leria como *"a evidência é fraca"* quando a
    verdade é *"a evidência existe e o produto não a alcança"* — conclusões
    diferentes, ações diferentes.
    """

    def _perfeita(self, **kw):
        return _ev(anos_positivos=10, anos_testados=10, capacidade_estimada=5_000_000,
                   capital_alvo=100_000, **kw)

    def test_long_perfeita_entra_no_pool(self):
        v = classificar(self._perfeita(direcao=Direcao.LONG))
        assert v.estado is Estado.VALIDADO and v.entra_no_pool
        assert v.quadrante == "VALIDADA"

    def test_short_perfeita_tem_evidencia_e_nao_entra_no_pool(self):
        v = classificar(self._perfeita(direcao=Direcao.SHORT))
        assert v.estado is Estado.VALIDADO, "a evidência não foi tocada"
        assert not v.entra_no_pool
        assert v.quadrante == "ALPHA_NAO_EXECUTAVEL"
        assert not v.estado.e_evidencia_contra, "inelegível NUNCA é refutada"
        assert "não executa short" in v.motivos_inelegibilidade[0]

    def test_direcao_nao_declarada_e_inelegivel_com_motivo(self):
        """Omissão nunca vira elegibilidade por conveniência."""
        v = classificar(self._perfeita(direcao=Direcao.DESCONHECIDA))
        assert not v.elegivel_no_produto
        assert "não declarada" in v.motivos_inelegibilidade[0]

    def test_elegibilidade_vale_ate_nas_nao_estimaveis(self):
        """O eixo é independente: não depende de haver estatística."""
        v = classificar(_ev(trades=10, direcao=Direcao.SHORT))
        assert v.estado is Estado.NAO_ESTIMAVEL
        assert not v.elegivel_no_produto and v.motivos_inelegibilidade

    def test_perfil_customizado_libera_o_short(self):
        perfil = PerfilDoProduto(direcoes_suportadas=frozenset({Direcao.LONG, Direcao.SHORT}))
        v = classificar(self._perfeita(direcao=Direcao.SHORT), perfil_produto=perfil)
        assert v.entra_no_pool and v.quadrante == "VALIDADA"

    def test_mapa_separa_pool_de_evidencia_sem_elegibilidade(self):
        m = montar_mapa([
            self._perfeita(hipotese="long_ok", direcao=Direcao.LONG),
            self._perfeita(hipotese="short_ok", direcao=Direcao.SHORT),
        ])
        assert [v.hipotese for v in m.pool] == ["long_ok"]
        assert [v.hipotese for v in m.evidencia_sem_elegibilidade] == ["short_ok"]
        assert m.resumo()["evidencia_sem_elegibilidade"] == 1


class TestDimensoesTemTresEstados:
    def test_nao_medida_e_diferente_de_nao(self):
        """
        `NAO` diz *esta hipótese não serve*; `NAO_MEDIDA` diz *vá medir*. Ambos
        bloqueiam VALIDADO, mas as instruções são opostas.
        """
        nao_medida = classificar(_ev(estavel_oos=None, capacidade_estimada=None))
        assert nao_medida.dimensoes["estabilidade_oos"] is Dimensao.NAO_MEDIDA
        assert nao_medida.dimensoes["capacidade"] is Dimensao.NAO_MEDIDA

        instavel = classificar(_ev(estavel_oos=False, capacidade_estimada=0.0,
                                   capital_alvo=100_000))
        assert instavel.dimensoes["estabilidade_oos"] is Dimensao.NAO
        assert instavel.dimensoes["capacidade"] is Dimensao.NAO

    def test_todas_as_seis_dimensoes_sempre_presentes(self):
        v = classificar(_ev())
        assert set(v.dimensoes) == {
            "evidencia_oos", "retorno_liquido", "margem_de_custo",
            "persistencia", "estabilidade_oos", "capacidade", "risco_drawdown",
        }

    def test_dimensoes_chegam_em_todos_os_ramos_menos_o_nao_estimavel(self):
        for kw in ({"t_nw": 1.0}, {"t_nw": 0.1, "excesso_liquido": 1e-5,
                                   "efeito_minimo_relevante": 0.5}):
            assert classificar(_ev(**kw)).dimensoes, kw

    def test_equilibrio_derivado_do_excesso(self):
        """`s = excesso/2` porque o round-trip encarece em `2s`."""
        assert equilibrio_slippage_bps_lado(0.013644) == pytest.approx(68.22)
        assert equilibrio_slippage_bps_lado(0.0010) == pytest.approx(5.0)


class TestMatrizDeQuadrantes:
    """
    A matriz evidência × elegibilidade (desenho do DEV), com **4 linhas** e não
    3 — porque "medi e ficou abaixo da régua" e "não consegui medir" são as duas
    coisas que este sistema inteiro existe para separar, e colapsá-las no topo
    reintroduziria o erro que o quarto veredito corrigiu na base.
    """

    def _forte(self, **kw):
        return _ev(anos_positivos=10, anos_testados=10, capacidade_estimada=5_000_000,
                   capital_alvo=100_000, **kw)

    @pytest.mark.parametrize(
        "kw,quadrante",
        [
            ({"direcao": Direcao.LONG}, "VALIDADA"),
            ({"direcao": Direcao.SHORT}, "ALPHA_NAO_EXECUTAVEL"),
            ({"direcao": Direcao.LONG, "t_nw": 1.0}, "CANDIDATA"),
            ({"direcao": Direcao.SHORT, "t_nw": 1.0}, "CANDIDATA_NAO_EXECUTAVEL"),
            ({"direcao": Direcao.LONG, "trades": 10}, "NAO_ESTIMAVEL"),
            ({"direcao": Direcao.SHORT, "trades": 10}, "NAO_ESTIMAVEL"),
            ({"direcao": Direcao.LONG, "t_nw": 0.1, "excesso_liquido": 1e-5,
              "efeito_minimo_relevante": 0.5}, "REFUTADA"),
            ({"direcao": Direcao.SHORT, "t_nw": 0.1, "excesso_liquido": 1e-5,
              "efeito_minimo_relevante": 0.5}, "REFUTADA"),
        ],
    )
    def test_cada_celula_da_matriz(self, kw, quadrante):
        assert classificar(self._forte(**kw)).quadrante == quadrante

    def test_nao_estimavel_e_refutada_nao_se_dividem_por_coluna(self):
        """
        Sem estatística ou com refutação, a elegibilidade não muda NADA que se
        possa fazer com a hipótese — quadrante sem ação distinta é ruído.
        """
        for kw in ({"trades": 10}, {"t_nw": 0.1, "excesso_liquido": 1e-5,
                                    "efeito_minimo_relevante": 0.5}):
            long_ = classificar(self._forte(direcao=Direcao.LONG, **kw)).quadrante
            short = classificar(self._forte(direcao=Direcao.SHORT, **kw)).quadrante
            assert long_ == short

    def test_todo_quadrante_tem_acao_distinta(self):
        acoes = [_ACAO_QUADRANTE[q] for q in QUADRANTES]
        assert len(set(acoes)) == len(QUADRANTES), (
            "dois quadrantes com a mesma ação não precisariam ser dois"
        )
        assert all(a for a in acoes), "quadrante sem ação é rótulo, não decisão"

    def test_alpha_nao_executavel_manda_preservar_e_nao_descartar(self):
        v = classificar(self._forte(direcao=Direcao.SHORT))
        assert "PRESERVAR" in v.acao_do_quadrante
        assert "DECISÃO DE ARQUITETURA" in v.acao_do_quadrante
        assert not v.estado.e_evidencia_contra

    def test_criterios_que_faltam_ordena_a_fila(self):
        """1 critério é um passo; 5 é outra pesquisa."""
        um = classificar(_ev(direcao=Direcao.LONG, anos_positivos=6, anos_testados=10,
                             capacidade_estimada=5_000_000, capital_alvo=100_000))
        muitos = classificar(_ev(direcao=Direcao.LONG, t_nw=1.0, anos_positivos=3,
                                 anos_testados=10, estavel_oos=False,
                                 capacidade_estimada=1.0, capital_alvo=100_000))
        assert um.criterios_que_faltam == 1
        assert muitos.criterios_que_faltam > um.criterios_que_faltam

    def test_resumo_expoe_a_matriz_inteira(self):
        m = montar_mapa([self._forte(hipotese="a", direcao=Direcao.LONG),
                         self._forte(hipotese="b", direcao=Direcao.SHORT)])
        r = m.resumo()
        assert r["quadrante_validada"] == 1
        assert r["quadrante_alpha_nao_executavel"] == 1
        assert sum(r[f"quadrante_{q.lower()}"] for q in QUADRANTES) == r["total"]
        assert set(m.por_quadrante()) == set(QUADRANTES)


class TestTipoDeBloqueio:
    """
    A terceira dimensão: estado diz ONDE a hipótese está, `criterios_que_faltam`
    diz QUÃO PERTO, e o bloqueio diz **O QUE exatamente a impede** — que é o que
    manda a hipótese para a fila certa, com o dono certo.
    """

    def test_falha_de_evidencia_e_de_execucao_vao_para_categorias_diferentes(self):
        v = classificar(_ev(direcao=Direcao.SHORT, anos_positivos=6, anos_testados=10,
                            capacidade_estimada=5_000_000, capital_alvo=100_000))
        b = v.bloqueios
        assert "persistência" in b["evidencia"][0]
        assert "não executa short" in b["execucao"][0]
        assert v.bloqueio_dominante is Bloqueio.EVIDENCIA

    def test_risco_passou_a_disputar_quando_virou_mensuravel(self):
        """
        Enquanto drawdown era NÃO MEDIDA em 100% das hipóteses, `RISCO` ficava
        fora da disputa de propósito — teria sido o dominante de todas as 50,
        verdade inútil. Agora que há número (`radar.cerebro.risco`), ele
        disputa — e vai por ÚLTIMO, porque não adianta avaliar risco de algo
        sem edge, sem margem e sem execução.
        """
        assert Bloqueio.RISCO in PRIORIDADE_DE_BLOQUEIO
        assert PRIORIDADE_DE_BLOQUEIO[-1] is Bloqueio.RISCO

        ok = classificar(_ev(direcao=Direcao.LONG, anos_positivos=10, anos_testados=10,
                             capacidade_estimada=5_000_000, capital_alvo=100_000))
        assert ok.bloqueio_dominante is Bloqueio.NENHUM

        so_risco = classificar(_ev(direcao=Direcao.LONG, anos_positivos=10,
                                   anos_testados=10, capacidade_estimada=5_000_000,
                                   capital_alvo=100_000,
                                   risco_dentro_da_tolerancia=False,
                                   max_drawdown=0.42))
        assert so_risco.bloqueio_dominante is Bloqueio.RISCO
        assert "42.0%" in so_risco.bloqueios["risco"][0]

    def test_prioridade_evidencia_antes_de_economia_antes_de_execucao(self):
        """Não se otimiza execução de algo sem edge."""
        assert PRIORIDADE_DE_BLOQUEIO == (
            Bloqueio.EVIDENCIA, Bloqueio.ECONOMIA, Bloqueio.EXECUCAO, Bloqueio.RISCO
        )
        tudo = classificar(_ev(direcao=Direcao.SHORT, t_nw=1.0, excesso_liquido=-0.01,
                               anos_positivos=3, anos_testados=10,
                               capacidade_estimada=1.0, capital_alvo=100_000,
                               risco_dentro_da_tolerancia=False, max_drawdown=0.42))
        assert tudo.bloqueio_dominante is Bloqueio.EVIDENCIA
        assert set(tudo.bloqueios) >= {"evidencia", "economia", "execucao", "risco"}

    def test_risco_so_aparece_quando_de_fato_bloqueia(self):
        """
        `RISCO` deixou de ser acrescentado incondicionalmente. Hipótese com
        drawdown dentro da tolerância NÃO tem entrada de risco — senão a
        categoria nunca esvaziaria e, com `RISCO` na prioridade, dominaria
        TODA hipótese, inclusive as que passam.
        """
        ok = classificar(_ev(direcao=Direcao.LONG, anos_positivos=10,
                             anos_testados=10, capacidade_estimada=5_000_000,
                             capital_alvo=100_000))
        assert "risco" not in ok.bloqueios

    def test_nao_estimavel_bloqueia_por_evidencia_e_nao_por_nenhum(self):
        """
        Sem dimensões computadas o bloqueio sairia `NENHUM` — leitura perigosa:
        a hipótese não está desimpedida, ela não foi medida.
        """
        v = classificar(_ev(trades=10, direcao=Direcao.LONG))
        assert v.estado is Estado.NAO_ESTIMAVEL
        assert v.bloqueio_dominante is Bloqueio.EVIDENCIA
        assert "nenhum teste aconteceu" in v.bloqueios["evidencia"][0]

    def test_todo_criterio_tem_categoria_declarada(self):
        v = classificar(_ev())
        assert set(v.dimensoes) == set(CATEGORIA_DO_CRITERIO), (
            "critério sem categoria cairia em EVIDENCIA por default e mentiria "
            "sobre a natureza do bloqueio"
        )

"""
Testes — `radar.cerebro.contrato` (Contrato de Pensamento, ADR 0032).

Os 4 portões existem porque o MIE v1 falhou nos 3 primeiros ao mesmo tempo e
nenhum teste percebeu. Por isso este arquivo não se contenta em exercitar as
funções: ele **reproduz o defeito real medido** e prova que o portão o pega.

  * `TestReproducaoDoDefeitoDoMieV1` — o caso patológico exato (12 valores
    distintos em 283 candidatos, constantes dentro do dia; gate 0,60 contra
    teto 0,1873). Se algum dia estes testes ficarem verdes com o defeito
    presente, o Contrato virou decoração.

Zero I/O: todos os frames são sintéticos e minúsculos.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from radar.cerebro.contrato import (
    CONTEXTO_GLOBAL,
    DIMENSOES_CONTEXTO,
    MINIMO_VALORES_DISTINTOS,
    MOTIVO_CONSTANTE_NO_INSTANTE,
    MOTIVO_POUCOS_VALORES,
    MOTIVO_SAIDA_OK,
    MOTIVO_SAIDA_VAZIA,
    ContextoForaDoVocabularioError,
    ContratoDePensamentoError,
    EmissorMudoError,
    EspacoDeSelecaoInvalidoError,
    LimiarInalcancavelError,
    colunas_constantes_no_instante,
    diagnosticar_saida,
    exigir_saida_que_discrimina,
    validar_contextos,
    validar_espaco_de_selecao,
    validar_limiar_alcancavel,
)


def _frame_dois_dias() -> pd.DataFrame:
    """2 dias × 3 ativos. `breadth` é constante no dia (varia só entre dias),
    `rsi` varia entre ativos, `constante_geral` nunca varia."""
    return pd.DataFrame(
        {
            "date": ["2026-01-05"] * 3 + ["2026-01-06"] * 3,
            "ticker": ["AAAA4", "BBBB4", "CCCC4"] * 2,
            "breadth": [0.42, 0.42, 0.42, 0.61, 0.61, 0.61],
            "rsi": [30.0, 55.0, 70.0, 28.0, 51.0, 74.0],
            "constante_geral": [1.0] * 6,
        }
    )


# ── G-P1 · Separação seleção × condição ─────────────────────────────────────


class TestColunasConstantesNoInstante:
    def test_pega_coluna_constante_no_dia_que_varia_entre_dias(self):
        constantes = colunas_constantes_no_instante(
            _frame_dois_dias(), ["breadth", "rsi"], chave_instante="date"
        )
        assert constantes == ("breadth",)

    def test_pega_coluna_constante_em_tudo(self):
        constantes = colunas_constantes_no_instante(
            _frame_dois_dias(), ["constante_geral"], chave_instante="date"
        )
        assert constantes == ("constante_geral",)

    def test_coluna_que_separa_ativos_nao_e_acusada(self):
        assert colunas_constantes_no_instante(
            _frame_dois_dias(), ["rsi"], chave_instante="date"
        ) == ()

    def test_instante_com_uma_linha_e_ignorado(self):
        """Num dia de 1 ativo TODA coluna é trivialmente constante. Contar
        esse dia reprovaria uma feature legítima só porque o universo teve um
        dia magro — acusar sem evidência é o mesmo erro que o módulo combate."""
        df = pd.DataFrame(
            {"date": ["2026-01-05", "2026-01-06", "2026-01-06"], "rsi": [30.0, 40.0, 70.0]}
        )
        assert colunas_constantes_no_instante(df, ["rsi"], chave_instante="date") == ()

    def test_frame_so_com_instantes_de_uma_linha_nao_acusa_ninguem(self):
        df = pd.DataFrame({"date": ["2026-01-05", "2026-01-06"], "rsi": [30.0, 40.0]})
        assert colunas_constantes_no_instante(df, ["rsi"], chave_instante="date") == ()

    def test_nan_nao_conta_como_valor_entao_coluna_meio_nan_nao_e_constante(self):
        """Uma coluna `NaN` para metade dos ativos e com valor na outra metade
        SEPARA os dois grupos — é exatamente o tipo de separação silenciosa que
        precisa aparecer, não ser absolvida."""
        df = pd.DataFrame(
            {"date": ["2026-01-05"] * 4, "f": [np.nan, np.nan, 0.3, 0.3]}
        )
        assert colunas_constantes_no_instante(df, ["f"], chave_instante="date") == ()

    def test_coluna_toda_nan_e_constante(self):
        df = pd.DataFrame({"date": ["2026-01-05"] * 3, "f": [np.nan] * 3})
        assert colunas_constantes_no_instante(df, ["f"], chave_instante="date") == ("f",)

    def test_frame_vazio_nao_acusa(self):
        df = pd.DataFrame({"date": [], "f": []})
        assert colunas_constantes_no_instante(df, ["f"], chave_instante="date") == ()

    def test_coluna_ausente_levanta_em_vez_de_ignorar(self):
        """Typo no nome da feature reprovaria o portão em silêncio — features
        já morreram por typo neste projeto."""
        with pytest.raises(KeyError, match="ausente"):
            colunas_constantes_no_instante(_frame_dois_dias(), ["nao_existe"], chave_instante="date")

    def test_chave_de_instante_ausente_levanta(self):
        with pytest.raises(KeyError, match="chave de instante"):
            colunas_constantes_no_instante(_frame_dois_dias(), ["rsi"], chave_instante="barra")


class TestValidarEspacoDeSelecao:
    def test_espaco_limpo_passa_e_o_portao_continua_vivo(self):
        """A asserção do estado vem JUNTO de propósito. "Não levantou" sozinho
        é uma asserção fraca: um refactor que transformasse
        `validar_espaco_de_selecao` num `pass` deixaria este teste verde para
        sempre — e uma guarda que para de guardar em silêncio é exatamente a
        classe de falha que originou este Contrato."""
        frame = _frame_dois_dias()

        assert colunas_constantes_no_instante(frame, ["rsi"], chave_instante="date") == ()
        validar_espaco_de_selecao(frame, ["rsi"], chave_instante="date", motor="teste")

    def test_espaco_com_contexto_levanta_com_o_nome_da_coluna(self):
        with pytest.raises(EspacoDeSelecaoInvalidoError) as exc:
            validar_espaco_de_selecao(
                _frame_dois_dias(), ["rsi", "breadth"], chave_instante="date", motor="mie_b3_h21"
            )
        assert "breadth" in str(exc.value)
        assert "mie_b3_h21" in str(exc.value)

    def test_e_subclasse_do_erro_base(self):
        assert issubclass(EspacoDeSelecaoInvalidoError, ContratoDePensamentoError)


# ── G-P2 · A saída tem que discriminar ──────────────────────────────────────


class TestDiagnosticarSaida:
    def test_saida_variada_passa(self):
        pred = np.linspace(0.1, 0.9, 50)
        d = diagnosticar_saida(pred, chave_instante=["2026-01-05"] * 50)
        assert d.mudo is False
        assert d.motivo == MOTIVO_SAIDA_OK
        assert d.n_valores_distintos == 50

    def test_saida_constante_e_muda(self):
        d = diagnosticar_saida([0.18] * 100, chave_instante=["2026-01-05"] * 100)
        assert d.mudo is True
        assert d.motivo == MOTIVO_POUCOS_VALORES

    def test_saida_vazia_e_muda(self):
        d = diagnosticar_saida([], chave_instante=[])
        assert d.mudo is True
        assert d.motivo == MOTIVO_SAIDA_VAZIA
        assert d.n_total == 0

    def test_universo_pequeno_nao_e_reprovado_por_aritmetica(self):
        """Com 3 candidatos é impossível ter 20 valores distintos. Exigir o
        impossível é o mesmo defeito de escala que o G-P3 combate — o piso é
        `min(minimo, n_total)`."""
        d = diagnosticar_saida([0.1, 0.5, 0.9], chave_instante=["2026-01-05"] * 3)
        assert d.mudo is False

    def test_universo_pequeno_mas_constante_continua_mudo(self):
        d = diagnosticar_saida([0.5, 0.5, 0.5], chave_instante=["2026-01-05"] * 3)
        assert d.mudo is True

    def test_varia_no_tempo_mas_nao_dentro_do_dia_e_mudo(self):
        """O caso patológico do MIE v1: o número muda ao longo dos dias (segue
        o regime) e é IDÊNTICO entre ativos do mesmo dia. Passa folgado no
        critério de variedade global e tem que morrer no critério do
        instante."""
        dias = [f"2026-01-{d:02d}" for d in range(1, 26)]
        pred = [0.10 + 0.01 * i for i, _ in enumerate(dias) for _ in range(10)]
        instantes = [dia for dia in dias for _ in range(10)]
        d = diagnosticar_saida(pred, chave_instante=instantes)
        assert d.n_valores_distintos >= MINIMO_VALORES_DISTINTOS
        assert d.mudo is True
        assert d.motivo == MOTIVO_CONSTANTE_NO_INSTANTE

    def test_saida_toda_nan_e_muda(self):
        d = diagnosticar_saida([np.nan] * 50, chave_instante=["2026-01-05"] * 50)
        assert d.mudo is True
        assert d.n_valores_distintos == 0

    def test_rank_ic_nao_finito_derruba(self):
        pred = np.linspace(0.1, 0.9, 50)
        d = diagnosticar_saida(
            pred, chave_instante=["2026-01-05"] * 50, rank_ic_medio=float("nan")
        )
        assert d.mudo is True

    def test_rank_ic_transportado_para_o_diagnostico(self):
        d = diagnosticar_saida(
            np.linspace(0.1, 0.9, 50), chave_instante=["2026-01-05"] * 50, rank_ic_medio=0.031
        )
        assert d.rank_ic_medio == pytest.approx(0.031)
        assert d.mudo is False

    def test_tamanhos_diferentes_levantam(self):
        with pytest.raises(ValueError, match="tamanhos diferentes"):
            diagnosticar_saida([0.1, 0.2], chave_instante=["2026-01-05"])


class TestExigirSaidaQueDiscrimina:
    def test_saida_boa_nao_levanta_e_o_portao_continua_vivo(self):
        d = diagnosticar_saida(np.linspace(0.1, 0.9, 50), chave_instante=["2026-01-05"] * 50)

        assert d.mudo is False
        exigir_saida_que_discrimina(d, motor="teste")

    def test_saida_muda_levanta_com_os_numeros(self):
        d = diagnosticar_saida([0.18] * 283, chave_instante=["2026-01-05"] * 283)
        with pytest.raises(EmissorMudoError) as exc:
            exigir_saida_que_discrimina(d, motor="mie_cripto_h21")
        assert "283" in str(exc.value)
        assert "mie_cripto_h21" in str(exc.value)


# ── G-P3 · O limiar tem que ser alcançável ──────────────────────────────────


class TestValidarLimiarAlcancavel:
    def test_limiar_abaixo_do_teto_passa(self):
        teto = 0.1873

        assert teto > 0.10  # a premissa do caso, explícita
        validar_limiar_alcancavel(0.10, teto, motor="m", escala="probabilidade calibrada")

    def test_fronteira_e_exatamente_o_teto(self):
        """Duas coisas num teste só, e de propósito: o teto é alcançável
        (`>=` no gate significa que o próprio teto emite, então a fronteira
        não pode reprovar) E o menor valor representável ACIMA dele já
        reprova. É o `nextafter` que prova que o portão está vivo — sem ele,
        um `pass` no lugar da função deixaria o teste verde."""
        teto = 0.1873

        validar_limiar_alcancavel(teto, teto, motor="m", escala="probabilidade calibrada")

        with pytest.raises(LimiarInalcancavelError):
            validar_limiar_alcancavel(
                math.nextafter(teto, 1.0), teto, motor="m", escala="probabilidade calibrada"
            )

    def test_limiar_acima_do_teto_levanta(self):
        with pytest.raises(LimiarInalcancavelError) as exc:
            validar_limiar_alcancavel(0.60, 0.1873, motor="mie_cripto_h21", escala="prob")
        assert "INALCANÇÁVEL" in str(exc.value)

    def test_limiar_nan_levanta(self):
        with pytest.raises(LimiarInalcancavelError):
            validar_limiar_alcancavel(float("nan"), 0.5, motor="m", escala="prob")

    def test_teto_nan_levanta(self):
        with pytest.raises(LimiarInalcancavelError):
            validar_limiar_alcancavel(0.5, float("nan"), motor="m", escala="prob")

    def test_limiar_abaixo_do_piso_levanta(self):
        """Espelho do gate inalcançável: um limiar abaixo do menor valor
        possível aprova TUDO, e é igualmente silencioso."""
        with pytest.raises(LimiarInalcancavelError, match="ABAIXO do piso"):
            validar_limiar_alcancavel(0.01, 0.30, motor="m", escala="prob", piso=0.21)

    def test_piso_none_nao_checa_o_espelho(self):
        """Sem `piso`, o mesmo limiar que reprovaria COM piso passa — é o par
        que prova que o parâmetro é opcional de verdade, e não que a checagem
        do espelho está morta."""
        validar_limiar_alcancavel(0.01, 0.30, motor="m", escala="prob")

        with pytest.raises(LimiarInalcancavelError):
            validar_limiar_alcancavel(0.01, 0.30, motor="m", escala="prob", piso=0.21)


# ── G-P4 · Vocabulário de contexto ──────────────────────────────────────────


class TestValidarContextos:
    def test_global_passa(self):
        assert CONTEXTO_GLOBAL not in DIMENSOES_CONTEXTO  # não é dimensão, é a célula não-condicionada
        validar_contextos([CONTEXTO_GLOBAL], motor="m")

    def test_dimensoes_declaradas_passam(self):
        rotulos = [f"{d}:algum_valor" for d in DIMENSOES_CONTEXTO]

        assert len(rotulos) == len(DIMENSOES_CONTEXTO)
        validar_contextos(rotulos, motor="m")

        with pytest.raises(ContextoForaDoVocabularioError):
            validar_contextos([*rotulos, "dimensao_inventada:x"], motor="m")

    def test_dimensao_desconhecida_levanta(self):
        with pytest.raises(ContextoForaDoVocabularioError, match="fase_da_lua"):
            validar_contextos(["fase_da_lua:cheia"], motor="m")

    def test_rotulo_livre_sem_dimensao_levanta(self):
        with pytest.raises(ContextoForaDoVocabularioError):
            validar_contextos(["bear"], motor="m")

    def test_dimensao_sem_valor_levanta(self):
        """`"regime:"` nomeia a dimensão sem dizer o recorte — reabre a porta
        do subgrupo conveniente que o ADR 0031 §4 fecha."""
        with pytest.raises(ContextoForaDoVocabularioError):
            validar_contextos(["regime:"], motor="m")

    def test_lista_vazia_passa(self):
        """Nada a validar não é violação — mas o par com um rótulo inválido
        prova que o silêncio é do conjunto vazio, não do validador."""
        validar_contextos([], motor="m")

        with pytest.raises(ContextoForaDoVocabularioError):
            validar_contextos(["invalido"], motor="m")


# ── O defeito real, reproduzido ─────────────────────────────────────────────


class TestReproducaoDoDefeitoDoMieV1:
    """Se estes testes ficarem verdes com o defeito presente, o Contrato virou
    decoração. Os números vêm da medição de 2026-08-04 (PRD + artefatos)."""

    def test_saida_do_mie_v1_e_declarada_muda(self):
        """Medido em PRD: 2.580 decisões, 12 valores distintos de probabilidade
        calibrada no universo inteiro, 283 pares por dia. Reproduzido aqui com
        os 2 valores que dominavam um único dia (0,1812 em 246 pares e 0,1615
        em 36)."""
        pred = [0.1812] * 246 + [0.1615] * 36
        d = diagnosticar_saida(pred, chave_instante=["2026-08-03"] * len(pred))

        assert d.n_total == 282
        assert d.n_valores_distintos == 2
        assert d.mudo is True
        assert d.motivo == MOTIVO_POUCOS_VALORES

    def test_gate_de_producao_contra_teto_real_levanta(self):
        """`mie_cripto_h21`: gate 0,60 contra teto isotônico 0,1873. Um mês de
        abstenção 100% lida como saúde — esta asserção é o minuto zero."""
        with pytest.raises(LimiarInalcancavelError):
            validar_limiar_alcancavel(
                0.60, 0.1873, motor="mie_cripto_h21", escala="probabilidade calibrada"
            )

    def test_todas_as_celulas_de_producao_reprovam(self):
        """Os 9 artefatos medidos em 2026-08-04, com os gates que estavam
        configurados. Nenhum passa — a mudez era universal, não de uma célula."""
        tetos_e_gates = [
            ("mie_b3_h5", 0.60, 0.2879),
            ("mie_b3_h21", 0.55, 0.2455),
            ("mie_cripto_h21", 0.60, 0.1873),
        ]
        for nome, gate, teto in tetos_e_gates:
            with pytest.raises(LimiarInalcancavelError):
                validar_limiar_alcancavel(gate, teto, motor=nome, escala="probabilidade calibrada")

    def test_espaco_de_features_do_v1_reprova_no_g_p1(self):
        """As 7 colunas constantes no dia (5 numéricas + 2 categóricas de
        regime) estavam DENTRO do espaço de entrada do modelo — e a de maior
        gain era uma delas (`breadth_sma50`)."""
        df = pd.DataFrame(
            {
                "date": ["2026-01-05"] * 3,
                "breadth_sma50": [0.55] * 3,
                "ibov_dist_ema50": [0.02] * 3,
                "dia_da_semana": [1] * 3,
                "regime_mie": ["bull_lateral"] * 3,
                "rsi_14": [30.0, 55.0, 70.0],
            }
        )
        with pytest.raises(EspacoDeSelecaoInvalidoError) as exc:
            validar_espaco_de_selecao(
                df,
                ["breadth_sma50", "ibov_dist_ema50", "dia_da_semana", "regime_mie", "rsi_14"],
                chave_instante="date",
                motor="mie_b3_v1",
            )
        mensagem = str(exc.value)
        assert "breadth_sma50" in mensagem
        assert "rsi_14" not in mensagem  # a única que de fato separa ativos

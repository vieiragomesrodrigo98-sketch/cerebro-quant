"""
A autópsia — onde os trades ganharam e perderam, sem virar p-hacking.

O que estes testes protegem:

1. **Nenhum eixo derivado enxerga o futuro.** Alterar uma barra POSTERIOR não
   pode mexer no `adtv21`/`atr14` de hoje. É a Regra 5 do projeto virada em
   asserção: sem o `.shift(1)` a partição usaria informação que a decisão não
   tinha, e a leitura mediria o futuro em vez do contexto.
2. **Piso antes do número.** Célula com poucos trades ou poucos dias sai
   `estimavel=False` e o `t` dela **não** é reportado — "não consegui medir"
   nunca pode ser lido como "medi e reprovei" (o 4º veredito existe por isso).
3. **Eixo constante é G-P1, não "sem efeito".** Barra diária não tem horário; a
   leitura precisa dizer NÃO ESTIMÁVEL em vez de fingir que mediu.
4. **O omnibus distingue sinal de ruído nos dois sentidos** — acha o eixo que
   carrega informação e, mais importante, **não** acha o que não carrega, que é
   o caso onde o `max |t|` de muitas células vazias engana.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from radar.cerebro.autopsia import (
    BLOCOS_HORARIO,
    JANELA_ADTV,
    LeituraDeCelula,
    bloco_horario,
    derivar_liquidez_e_volatilidade,
    em_tercos,
    ler_eixo,
    omnibus_permutacao,
)


def _ohlcv(n=120, ticker="AAAA3", seed=1):
    rng = np.random.default_rng(seed)
    datas = pd.date_range("2020-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({
        "ticker": ticker,
        "date": datas,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": rng.uniform(1e6, 5e6, n),
    })


def _trades(n_dias=90, n_tickers=6, efeito_por_grupo=None, seed=7):
    """Trades sintéticos com um eixo `grupo` de 3 valores.

    `efeito_por_grupo` injeta um deslocamento real na média de cada grupo —
    `None` produz ruído puro, que é o caso que o omnibus precisa NÃO aprovar."""
    rng = np.random.default_rng(seed)
    datas = pd.date_range("2021-01-01", periods=n_dias, freq="D")
    linhas = []
    for i in range(n_tickers):
        grupo = ("baixa", "media", "alta")[i % 3]
        desloc = (efeito_por_grupo or {}).get(grupo, 0.0)
        for d in datas:
            linhas.append({
                "evidencia": "tec:estrategia@abc",
                "date": d,
                "ticker": f"T{i}",
                "grupo": grupo,
                "liquido": float(rng.normal(desloc, 0.02)),
            })
    return pd.DataFrame(linhas)


class TestSemLookAhead:
    def test_barra_futura_nao_altera_indicador_de_hoje(self):
        """O teste central: mexer no futuro não pode mexer no passado."""
        base = _ohlcv()
        derivado = derivar_liquidez_e_volatilidade(base)

        adulterado = base.copy()
        ultimas = adulterado.index[-10:]
        adulterado.loc[ultimas, "volume"] *= 1000.0
        adulterado.loc[ultimas, "high"] *= 5.0
        derivado_adulterado = derivar_liquidez_e_volatilidade(adulterado)

        corte = len(base) - 10
        pd.testing.assert_series_equal(
            derivado["adtv21"].iloc[:corte], derivado_adulterado["adtv21"].iloc[:corte]
        )
        pd.testing.assert_series_equal(
            derivado["atr14"].iloc[:corte], derivado_adulterado["atr14"].iloc[:corte]
        )

    def test_indicador_do_dia_nao_usa_a_barra_do_proprio_dia(self):
        """`shift(1)`: o valor de hoje é média que TERMINA ontem."""
        base = _ohlcv(n=60)
        derivado = derivar_liquidez_e_volatilidade(base)
        i = 40
        esperado = base["volume"].iloc[i - JANELA_ADTV:i].mean()
        assert derivado["adtv21"].iloc[i] == pytest.approx(esperado)

    def test_primeira_barra_sai_nan_nunca_zero(self):
        derivado = derivar_liquidez_e_volatilidade(_ohlcv())
        assert np.isnan(derivado["adtv21"].iloc[0])
        assert np.isnan(derivado["atr14"].iloc[0])

    def test_cada_ticker_tem_janela_propria(self):
        """Sem `groupby`, a janela vazaria de um ativo para o seguinte."""
        dois = pd.concat([_ohlcv(ticker="AAAA3", seed=1), _ohlcv(ticker="BBBB3", seed=2)])
        derivado = derivar_liquidez_e_volatilidade(dois)
        primeiro_de_bbbb = derivado[derivado["ticker"] == "BBBB3"].iloc[0]
        assert np.isnan(primeiro_de_bbbb["adtv21"])


class TestCortes:
    def test_tercos_produzem_tres_faixas(self):
        faixas = em_tercos(pd.Series(range(300)))
        assert set(faixas.dropna().unique()) == {"baixa", "media", "alta"}

    def test_serie_constante_nao_inventa_faixa(self):
        """Sem variação não há terço — e `NaN` vira NÃO ESTIMÁVEL na leitura."""
        faixas = em_tercos(pd.Series([5.0] * 50))
        assert faixas.isna().all()

    def test_horario_em_quatro_blocos(self):
        datas = pd.Series(pd.date_range("2021-01-01", periods=24, freq="h"))
        assert set(bloco_horario(datas).unique()) == set(BLOCOS_HORARIO)

    def test_barra_diaria_produz_bloco_unico(self):
        """B3: hora é sempre 00:00, logo o eixo é constante — não é eixo."""
        datas = pd.Series(pd.date_range("2021-01-01", periods=30, freq="D"))
        assert set(bloco_horario(datas).unique()) == {"00-05"}


class TestLerEixo:
    def test_celula_abaixo_do_piso_nao_reporta_t(self):
        curto = _trades(n_dias=5, n_tickers=6)
        leituras = ler_eixo(curto, "grupo")
        assert leituras, "eixo com 3 valores deve produzir leitura"
        for leitura in leituras:
            assert leitura.estimavel is False
            assert np.isnan(leitura.t_nw)
            assert "abaixo do piso" in leitura.motivo

    def test_celula_com_amostra_reporta_t(self):
        leituras = ler_eixo(_trades(), "grupo")
        assert all(leitura.estimavel for leitura in leituras)
        assert all(not np.isnan(leitura.t_nw) for leitura in leituras)

    def test_eixo_constante_e_g_p1_nao_sem_efeito(self):
        df = _trades()
        df["grupo"] = "unico"
        (leitura,) = ler_eixo(df, "grupo")
        assert leitura.estimavel is False
        assert "CONSTANTE" in leitura.motivo
        assert "G-P1" in leitura.motivo

    def test_eixo_ausente_nao_levanta_devolve_nao_estimavel(self):
        (leitura,) = ler_eixo(_trades(), "horario")
        assert leitura.estimavel is False
        assert "ausente" in leitura.motivo

    def test_to_dict_serializa_nan_como_none(self):
        leitura = LeituraDeCelula("x", "v", 1, 1, float("nan"), float("nan"), False, "m")
        assert leitura.to_dict()["t_nw"] is None
        assert leitura.to_dict()["excesso_medio"] is None


class TestOmnibus:
    def test_ruido_puro_nao_se_distingue(self):
        """O caso que importa: sem sinal, o `max |t|` real fica dentro do nulo."""
        resultado = omnibus_permutacao(
            _trades(efeito_por_grupo=None), "grupo", n_permutacoes=15, lag=5
        )
        assert resultado.distingue_de_ruido is False
        assert resultado.p_valor > 0.05

    def test_efeito_forte_e_detectado(self):
        """E o contrário também: com sinal plantado, o eixo aparece."""
        resultado = omnibus_permutacao(
            _trades(efeito_por_grupo={"alta": 0.05, "baixa": -0.05}, seed=3),
            "grupo", n_permutacoes=15, lag=5,
        )
        assert resultado.distingue_de_ruido is True
        assert resultado.max_t_real > resultado.max_t_nulo_mediana

    def test_permutacao_preserva_o_numero_de_celulas(self):
        """O nulo tem de ter a MESMA grade — senão compara coisas diferentes."""
        resultado = omnibus_permutacao(_trades(), "grupo", n_permutacoes=5, lag=5)
        assert resultado.n_celulas == 3
        assert resultado.n_permutacoes == 5

    def test_e_reprodutivel_pela_seed(self):
        a = omnibus_permutacao(_trades(), "grupo", n_permutacoes=8, lag=5, seed=42)
        b = omnibus_permutacao(_trades(), "grupo", n_permutacoes=8, lag=5, seed=42)
        assert a.p_valor == b.p_valor
        assert a.max_t_nulo_media == b.max_t_nulo_media

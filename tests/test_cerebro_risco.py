"""
Curva de equity e drawdown — a lacuna de RISCO.

O que estes testes protegem:

1. **Ausência é `None`, e `None` reprova.** Quarta aplicação do princípio nesta
   linhagem: sem trades não há curva, e "não medi" jamais pode virar "passou".
2. **A curva agrega por DIA antes de acumular.** Acumular linha a linha faria o
   drawdown depender da ordem arbitrária das linhas dentro do dia.
3. **`recovery_dias = None` significa NUNCA recuperou** — é pior que um número
   grande, não melhor, e o teste amarra isso.
4. **`calmar = None` com drawdown zero** é "não computável", não "infinitamente
   bom".
"""

from __future__ import annotations

import pandas as pd
import pytest

from radar.cerebro.risco import (
    DRAWDOWN_MAXIMO_TOLERADO,
    MINIMO_PONTOS_CURVA,
    avaliar,
    curva_de_equity,
)


def _trades(liquidos, inicio="2020-01-01", por_dia=1, horizonte=1):
    """Um trade por dia consecutivo, salvo `por_dia` > 1.

    `horizonte=1` é o caso NÃO SOBREPOSTO: cada posição vive um passo, fecha
    antes da próxima abrir, e a carteira tem uma posição por vez. É o cenário em
    que a curva antiga (soma aritmética) e a nova (patrimônio composto) medem a
    mesma coisa a menos da composição — por isso é o default aqui.

    `horizonte>1` abre `horizonte` posições ao mesmo tempo, que é o defeito de
    `CEREBRO_EQUITY_HORIZONTE_SOBREPOSTO01`.
    """
    datas, vals = [], []
    d = pd.Timestamp(inicio)
    for i, v in enumerate(liquidos):
        datas.extend([d + pd.Timedelta(days=i)] * por_dia)
        vals.extend([v] * por_dia)
    entrada = pd.DatetimeIndex(datas)
    return pd.DataFrame(
        {
            "date": entrada,
            "exit_date": entrada + pd.Timedelta(days=horizonte),
            "liquido": vals,
        }
    )


class TestAusenciaNuncaViraAprovacao:
    @pytest.mark.parametrize("df", [
        pd.DataFrame(),
        pd.DataFrame({"date": [], "liquido": []}),
        pd.DataFrame({"date": [pd.Timestamp("2020-01-01")]}),   # sem `liquido`
        pd.DataFrame({"liquido": [0.01]}),                       # sem `date`
    ])
    def test_sem_dado_devolve_none(self, df):
        assert curva_de_equity(df) is None
        assert avaliar(df) is None

    def test_poucos_pontos_devolve_none(self):
        """Drawdown de meia dúzia de pontos é ruído, não métrica."""
        df = _trades([0.01] * (MINIMO_PONTOS_CURVA - 1))
        assert curva_de_equity(df) is None
        assert avaliar(df) is None

    def test_no_limite_ja_mede(self):
        assert curva_de_equity(_trades([0.01] * MINIMO_PONTOS_CURVA)) is not None


class TestCurvaAgregaPorDia:
    def test_dois_trades_no_mesmo_dia_sao_um_ponto(self):
        """
        Acumular linha a linha faria a curva depender da ordem arbitrária das
        linhas dentro do dia e inflaria a contagem de dias em drawdown.
        """
        um = curva_de_equity(_trades([0.01] * 40, por_dia=1))
        dois = curva_de_equity(_trades([0.01] * 40, por_dia=2))
        assert um is not None and dois is not None
        assert len(um) == len(dois) == 40
        assert um.iloc[-1] == pytest.approx(dois.iloc[-1])

    def test_curva_e_patrimonio_composto_e_ordenada(self):
        """A curva e PATRIMONIO (capital inicial 1), nao retorno acumulado.

        40 passos de +1% compoem para 1,01^40 = 1,4889 — nao somam 0,40. A
        diferenca nao e cosmetica: e ela que impede a curva de cruzar zero, que
        era como saiam os drawdowns de 583%.
        """
        c = curva_de_equity(_trades([0.01] * 40))
        assert c is not None
        assert c.is_monotonic_increasing
        assert c.iloc[-1] == pytest.approx(1.01 ** 40)
        assert c.iloc[-1] != pytest.approx(0.40), "somou em vez de compor"
        assert c.index.is_monotonic_increasing


class TestDrawdown:
    def test_curva_so_de_alta_tem_drawdown_zero(self):
        r = avaliar(_trades([0.01] * 40))
        assert r is not None
        assert r.max_drawdown == pytest.approx(0.0)
        assert r.dentro_da_tolerancia
        assert r.calmar is None, "drawdown zero não é Calmar infinito, é não computável"

    def test_queda_e_recuperacao_medidas(self):
        """
        Sobe 20, cai 10, sobe 20. O drawdown e RELATIVO ao pico, nao absoluto —
        e a curva COMPOE: o pico e 1,01^20 e o fundo 1,01^20 * 0,99^10.

        Comparar a queda ABSOLUTA (0,10) com uma tolerancia percentual seria
        misturar grandezas — foi o erro que fez 17 de 17 hipoteses saírem "fora
        da tolerancia" na primeira ligacao ao mapa.
        """
        r = avaliar(_trades([0.01] * 20 + [-0.01] * 10 + [0.01] * 20))
        assert r is not None
        pico, fundo = 1.01 ** 20, 1.01 ** 20 * 0.99 ** 10
        assert r.max_drawdown == pytest.approx((pico - fundo) / pico, abs=1e-9)
        assert r.duracao_drawdown_dias == 10
        # 11, nao 10: recuperar de 0,99^10 = 0,9044 exige 1,01^k >= 1,1057, ou
        # seja k >= 10,10. Com soma aritmetica dariam exatos 10 — a diferenca de
        # um passo e a composicao aparecendo, e e ela que se quer amarrada aqui.
        assert r.recovery_dias == 11

    def test_nunca_recuperou_e_none_e_isso_e_pior(self):
        r = avaliar(_trades([0.01] * 20 + [-0.01] * 25))
        assert r is not None
        assert r.recovery_dias is None
        assert not r.dentro_da_tolerancia
        assert any("nunca recuperou" in m for m in r.motivos)

    def test_drawdown_acima_da_tolerancia_reprova(self):
        r = avaliar(_trades([0.01] * 20 + [-0.03] * 15 + [0.01] * 10))
        assert r is not None
        assert r.max_drawdown > DRAWDOWN_MAXIMO_TOLERADO
        assert not r.dentro_da_tolerancia
        assert any("tolerância" in m for m in r.motivos)

    def test_tolerancia_e_parametrizavel_e_declarada(self):
        df = _trades([0.01] * 20 + [-0.01] * 12 + [0.01] * 15)
        assert avaliar(df, drawdown_maximo=0.50) is not None
        assert avaliar(df, drawdown_maximo=0.50).dentro_da_tolerancia
        assert not avaliar(df, drawdown_maximo=0.01).dentro_da_tolerancia

    def test_calmar_e_retorno_sobre_drawdown(self):
        r = avaliar(_trades([0.01] * 20 + [-0.01] * 10 + [0.01] * 20))
        assert r is not None and r.calmar is not None
        assert r.calmar == pytest.approx(r.retorno_total / r.max_drawdown)

    def test_mesma_media_drawdowns_diferentes(self):
        """
        O ponto do módulo: o retorno médio por trade é insensível à ORDEM, e a
        ordem é o que decide se dá para operar. Duas séries com a MESMA média
        têm drawdowns muito diferentes.
        """
        alternada = avaliar(_trades([0.01, -0.005] * 20))
        concentrada = avaliar(_trades([-0.005] * 20 + [0.01] * 20))
        assert alternada is not None and concentrada is not None
        assert alternada.retorno_total == pytest.approx(concentrada.retorno_total)
        assert concentrada.max_drawdown > alternada.max_drawdown

    def test_to_dict_serializa(self):
        r = avaliar(_trades([0.01] * 40))
        assert r is not None
        d = r.to_dict()
        assert d["n_pontos"] == 40 and d["dentro_da_tolerancia"] is True
        assert d["calmar"] is None


class TestGuardaDePrecoPositivo:
    """
    FIN-003: divisão por preço em `regime.py`/`regime_cripto.py`.

    Preço ≤ 0 numa série de fechamento é corrupção de dado, e o efeito dela
    seria `inf` viajando em silêncio até virar feature de regime — o padrão de
    falha muda que o Contrato de Pensamento proíbe. A guarda falha ALTO.
    """

    def test_preco_zero_levanta_com_mensagem_util(self):
        from radar.mie.regime import _exigir_precos_positivos
        s = pd.Series([100.0, 101.0, 0.0, 99.0])
        with pytest.raises(ValueError, match=r"inf.*silencioso"):
            _exigir_precos_positivos(s, "teste")

    def test_preco_negativo_levanta(self):
        from radar.mie.regime import _exigir_precos_positivos
        with pytest.raises(ValueError, match=r"<= 0"):
            _exigir_precos_positivos(pd.Series([100.0, -1.0]), "teste")

    def test_nan_nao_e_erro(self):
        """
        As janelas móveis (`min_periods`) produzem NaN legítimo no início da
        série; reprová-lo quebraria o caminho normal.

        A segunda metade existe porque a primeira, sozinha, passaria igual se a
        guarda virasse um `pass` — teste que só prova ausência de exceção não
        distingue "aceitou o NaN" de "não olhou nada" (QA-010).
        """
        from radar.mie.regime import _exigir_precos_positivos
        _exigir_precos_positivos(pd.Series([float("nan"), 100.0, 101.0]), "teste")

        with pytest.raises(ValueError, match=r"<= 0"):
            _exigir_precos_positivos(pd.Series([float("nan"), -1.0]), "teste")

    def test_serie_saudavel_passa(self):
        """As duas guardas aceitam série boa — e continuam vivas.

        O par positivo/negativo em cada guarda é o que impede o teste de
        sobreviver a uma delas virando no-op (QA-010).
        """
        from radar.historical.regime_cripto import (
            _exigir_precos_positivos as guarda_cripto,
        )
        from radar.mie.regime import _exigir_precos_positivos
        _exigir_precos_positivos(pd.Series([100.0, 101.0, 99.5]), "teste")
        guarda_cripto(pd.Series([60000.0, 61000.0]), "teste")

        with pytest.raises(ValueError):
            _exigir_precos_positivos(pd.Series([100.0, 0.0]), "teste")
        with pytest.raises(ValueError):
            guarda_cripto(pd.Series([60000.0, -1.0]), "teste")


class TestBloqueioP0BPatrimonio:
    """
    ADR 0034, bloqueio P0-B — **patrimônio não fica negativo**.

    Esta classe MUDOU de significado quando `CEREBRO_EQUITY_HORIZONTE_SOBREPOSTO01`
    foi corrigido, e a mudança é o registro do conserto.

    Antes: a curva era `1 + cumsum(retorno)`, então 40 perdas de 5% somavam −2,0
    e o equity ia a −1,0. O bloqueio existia para pegar isso, e pegava 20 das 42
    hipóteses do mapa.

    Agora a curva COMPÕE, e 40 perdas de 5% dão `0,95^40 = 0,129` — drawdown de
    87%, catastrófico e **medido**, jamais negativo. O bloqueio deixou de ser um
    detector de defeito conhecido e virou uma **invariante**: só dispara em
    RUÍNA de verdade (retorno de passo ≤ −100%). Mantê-lo é deliberado — remover
    uma trava porque "agora não acontece" é como o próximo defeito passa calado.
    """

    def test_curva_sa_e_patrimonio_valido(self):
        r = avaliar(_trades([0.01, -0.02, 0.015] * 15))
        assert r is not None
        assert r.patrimonio_valido is True

    def test_perda_catastrofica_agora_e_medida_e_nao_invalida(self):
        """O cenário que antes invalidava a curva agora produz um número honesto.

        É o par que prova que o conserto fez efeito: com a curva antiga este
        mesmo `avaliar` devolvia `patrimonio_valido=False` e drawdown de 200%.
        """
        r = avaliar(_trades([-0.05] * 40))
        assert r is not None
        assert r.patrimonio_valido is True
        assert r.max_drawdown == pytest.approx(1.0 - 0.95 ** 40)
        assert r.max_drawdown < 1.0, "patrimônio composto nunca perde mais que tudo"
        assert not r.dentro_da_tolerancia, "87% de queda segue reprovando na tolerância"

    def test_ruina_de_verdade_ainda_invalida(self):
        """A invariante continua viva: retorno de passo ≤ −100% zera e trava.

        Sem este teste o bloqueio viraria código morto — passaria verde para
        sempre e ninguém saberia se ele ainda olha alguma coisa.
        """
        r = avaliar(_trades([0.01] * 20 + [-1.5] + [0.01] * 20))
        assert r is not None
        assert r.patrimonio_valido is False
        assert any("RUÍNA" in m for m in r.motivos)
        assert r.dentro_da_tolerancia is False

    def test_ruina_nao_ressuscita_a_carteira(self):
        """Depois da ruína o patrimônio fica em zero — não volta a subir.

        Clampar o produto acumulado em vez do fator de cada passo pareceria
        equivalente e não é: um fator negativo sobrevive multiplicando e inverte
        o sinal no passo seguinte, ressuscitando a conta.
        """
        c = curva_de_equity(_trades([0.01] * 20 + [-1.5] + [0.05] * 20))
        assert c is not None
        depois = c.to_numpy()[21:]
        assert (depois == 0.0).all(), "carteira ressuscitou depois de zerar"

    def test_to_dict_publica_a_flag(self):
        r = avaliar(_trades([0.01] * 20 + [-1.5] + [0.01] * 20))
        assert r is not None
        assert r.to_dict()["patrimonio_valido"] is False


class TestSobreposicaoDeHorizonte:
    """`CEREBRO_EQUITY_HORIZONTE_SOBREPOSTO01` — o defeito que este módulo tinha.

    Com horizonte `h` e entrada a cada passo, `h` posições ficam abertas ao mesmo
    tempo. Somar o retorno TOTAL de cada uma no instante de entrada equivale a
    operar alavancado `h` vezes.
    """

    def test_horizonte_maior_nao_multiplica_o_patrimonio(self):
        """MESMOS trades, MESMO retorno por trade — só a janela muda.

        Um horizonte 10x maior não pode render 10x mais: o capital é o mesmo,
        dividido entre as posições abertas. O que muda é a frequência com que ele
        gira, não a alavancagem.
        """
        curto = curva_de_equity(_trades([0.01] * 60, horizonte=1))
        longo = curva_de_equity(_trades([0.01] * 60, horizonte=10))
        assert curto is not None and longo is not None

        assert longo.iloc[-1] < curto.iloc[-1], (
            "posição que dura 10 passos rende MENOS por passo, não igual: "
            "o mesmo resultado total diluído numa janela 10x maior"
        )
        # ~1% ao longo de 10 passos, com ~10 posicoes abertas dividindo o capital.
        assert longo.iloc[-1] == pytest.approx(1.001 ** 60, rel=0.02)

    def test_exit_date_ausente_e_nao_medida_e_nao_chute(self):
        """Sem saber quando fecha não há como contar quantas ficam abertas.

        A alternativa seria inferir a janela do `horizonte`, e ela exige adivinhar
        a duração da barra — que não é derivável do parquet: `swing_v1` tem
        barra=dia e `scalp_cripto_v2` tem barra=1min amostrada de 60 em 60, e os
        dois publicam só um inteiro em `horizonte`.
        """
        sem = _trades([0.01] * 40).drop(columns=["exit_date"])

        assert curva_de_equity(sem) is None
        assert avaliar(sem) is None

    def test_exit_date_nulo_tambem_e_nao_medida(self):
        """`NaT` é o que o produtor gravava antes da correção — e não pode passar."""
        nulo = _trades([0.01] * 40).assign(exit_date=pd.NaT)

        assert curva_de_equity(nulo) is None

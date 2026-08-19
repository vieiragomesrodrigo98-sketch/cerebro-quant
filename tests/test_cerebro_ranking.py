"""
Testes — `radar.cerebro.ranking` (estágio RANKING do Cérebro v2, ADR 0032).

Esta função foi PROMOVIDA a núcleo vindo de três scripts de pesquisa onde
vivia duplicada literalmente. Ela é o mecanismo por trás das duas únicas
células verdes do projeto (`short_bottom_10%_bear`, t_NW +2,83, persistência
100%) — então estes testes existem menos para provar que ela funciona e mais
para **fixar o comportamento exato** que aquelas células mediram. Cada
asserção aqui é uma trava de reprodutibilidade: se uma delas mudar, os números
das células verdes deixam de ser reproduzíveis.

Zero I/O.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from radar.cerebro.ranking import (
    MODO_ALEATORIO,
    MODO_BOTTOM,
    MODO_TOP,
    MODOS_VALIDOS,
    selecionar_percentil_instante,
)


def _pool(n_por_dia: int = 10, dias: int = 2) -> pd.DataFrame:
    """`prob` crescente e única dentro de cada dia — sem empate, para os testes
    de ordenação não dependerem da regra de desempate."""
    linhas = []
    for d in range(dias):
        for i in range(n_por_dia):
            linhas.append(
                {
                    "date": f"2026-01-{d + 1:02d}",
                    "ticker": f"T{i:02d}",
                    "prob": 0.10 + 0.01 * i,
                }
            )
    return pd.DataFrame(linhas)


class TestSelecaoBasica:
    def test_top_pega_os_maiores_de_cada_dia(self):
        out = selecionar_percentil_instante(_pool(), 0.2, MODO_TOP)

        assert len(out) == 4  # 2 por dia × 2 dias
        assert set(out["ticker"]) == {"T08", "T09"}

    def test_bottom_pega_os_menores_de_cada_dia(self):
        out = selecionar_percentil_instante(_pool(), 0.2, MODO_BOTTOM)

        assert len(out) == 4
        assert set(out["ticker"]) == {"T00", "T01"}

    def test_cada_instante_e_independente(self):
        """Dia magro e dia cheio no mesmo pool: a fração é aplicada DENTRO de
        cada dia, nunca sobre o total."""
        pool = pd.concat(
            [
                _pool(n_por_dia=10, dias=1),
                pd.DataFrame(
                    [{"date": "2026-02-01", "ticker": "X", "prob": 0.9}]
                ),
            ],
            ignore_index=True,
        )

        out = selecionar_percentil_instante(pool, 0.2, MODO_TOP)

        assert (out["date"] == "2026-01-01").sum() == 2
        assert (out["date"] == "2026-02-01").sum() == 1

    def test_colunas_preservadas(self):
        out = selecionar_percentil_instante(_pool(), 0.5, MODO_TOP)
        assert list(out.columns) == ["date", "ticker", "prob"]


class TestTamanhoDaSelecao:
    """`k = min(n, max(1, round(n * frac)))` — cada parte tem consequência
    medida e por isso está travada aqui."""

    def test_piso_de_um_candidato_por_instante(self):
        """`frac=10%` sobre 3 candidatos daria 0,3 → 0. O piso de 1 mantém o
        instante vivo, mas a frequência EFETIVA vira 33%, não 10% — quem lê a
        célula precisa saber, e por isso o piso é testado, não escondido."""
        pool = _pool(n_por_dia=3, dias=1)

        out = selecionar_percentil_instante(pool, 0.10, MODO_TOP)

        assert len(out) == 1

    def test_teto_no_tamanho_do_instante(self):
        pool = _pool(n_por_dia=4, dias=1)

        out = selecionar_percentil_instante(pool, 5.0, MODO_TOP)

        assert len(out) == 4

    def test_arredondamento_bancario_do_python_esta_travado(self):
        """`round(2.5) == 2` em Python (arredondamento para o par), não 3.
        As células verdes foram medidas com este comportamento; trocar por
        `int(n*frac + 0.5)` mudaria a contagem de trades em todo instante de
        tamanho par com frac=0,5 — e portanto os números publicados."""
        pool = _pool(n_por_dia=5, dias=1)

        out = selecionar_percentil_instante(pool, 0.5, MODO_TOP)

        assert round(5 * 0.5) == 2  # a premissa, explícita
        assert len(out) == 2

    def test_frac_zero_ainda_devolve_um_por_instante(self):
        out = selecionar_percentil_instante(_pool(n_por_dia=10, dias=2), 0.0, MODO_TOP)
        assert len(out) == 2


class TestModoAleatorio:
    def test_exige_rng(self):
        with pytest.raises(ValueError, match="rng"):
            selecionar_percentil_instante(_pool(), 0.2, MODO_ALEATORIO)

    def test_mesma_semente_mesmo_resultado(self):
        a = selecionar_percentil_instante(
            _pool(), 0.3, MODO_ALEATORIO, rng=np.random.default_rng(7)
        )
        b = selecionar_percentil_instante(
            _pool(), 0.3, MODO_ALEATORIO, rng=np.random.default_rng(7)
        )

        assert a.equals(b)

    def test_sementes_diferentes_podem_divergir(self):
        a = selecionar_percentil_instante(
            _pool(n_por_dia=50, dias=1), 0.2, MODO_ALEATORIO, rng=np.random.default_rng(1)
        )
        b = selecionar_percentil_instante(
            _pool(n_por_dia=50, dias=1), 0.2, MODO_ALEATORIO, rng=np.random.default_rng(2)
        )

        assert set(a["ticker"]) != set(b["ticker"])

    def test_respeita_a_mesma_contagem_dos_outros_modos(self):
        """O controle só é controle se tiver a MESMA frequência de sinal do
        braço que ele controla."""
        pool = _pool(n_por_dia=10, dias=2)
        top = selecionar_percentil_instante(pool, 0.3, MODO_TOP)
        aleatorio = selecionar_percentil_instante(
            pool, 0.3, MODO_ALEATORIO, rng=np.random.default_rng(3)
        )

        assert len(top) == len(aleatorio)


class TestBordas:
    def test_pool_vazio_devolve_frame_vazio_com_schema(self):
        vazio = _pool().iloc[0:0]

        out = selecionar_percentil_instante(vazio, 0.2, MODO_TOP)

        assert out.empty
        assert list(out.columns) == ["date", "ticker", "prob"]

    def test_modo_invalido_levanta(self):
        """Typo em `"botom"` cairia num default e inverteria o sinal da
        estratégia inteira — por isso levanta em vez de assumir."""
        with pytest.raises(ValueError, match="modo desconhecido"):
            selecionar_percentil_instante(_pool(), 0.2, "botom")

    def test_modo_invalido_levanta_mesmo_com_pool_vazio(self):
        """A única divergência declarada em relação à cópia original: lá a
        validação só acontecia dentro do laço, então um pool vazio engolia o
        typo em silêncio."""
        with pytest.raises(ValueError, match="modo desconhecido"):
            selecionar_percentil_instante(_pool().iloc[0:0], 0.2, "botom")

    def test_colunas_de_score_e_instante_parametrizaveis(self):
        pool = _pool().rename(columns={"prob": "score", "date": "barra"})

        out = selecionar_percentil_instante(
            pool, 0.2, MODO_TOP, col_score="score", col_instante="barra"
        )

        assert set(out["ticker"]) == {"T08", "T09"}

    def test_modos_validos_sao_os_tres(self):
        assert MODOS_VALIDOS == (MODO_TOP, MODO_BOTTOM, MODO_ALEATORIO)


class TestFormaDaCelulaVerde:
    def test_bottom_10_pct_por_dia_reproduz_a_forma_medida(self):
        """`short_bottom_10%_bear`: 10% do fundo, por dia, sobre uma seção
        transversal de dezenas de pares. Aqui em miniatura — o que se trava é
        a FORMA (quantos por dia, quais), não o número da célula."""
        pool = _pool(n_por_dia=30, dias=3)

        out = selecionar_percentil_instante(pool, 0.10, MODO_BOTTOM)

        assert len(out) == 9  # round(30*0.10) = 3 por dia × 3 dias
        assert set(out["ticker"]) == {"T00", "T01", "T02"}


# ── construir_trades_topo — Top-N com custo POR ATIVO ───────────────────────


class TestConstruirTradesTopo:
    def _meta(self, n_dias: int = 2, n_ativos: int = 20) -> pd.DataFrame:
        linhas = []
        for d in range(n_dias):
            for i in range(n_ativos):
                linhas.append(
                    {
                        "date": f"2026-01-{d + 1:02d}",
                        "ticker": f"T{i:02d}",
                        "ret_fwd": 0.01 * i,
                    }
                )
        return pd.DataFrame(linhas)

    def test_dispara_o_topo_e_desconta_o_custo_do_proprio_ativo(self):
        from radar.cerebro.ranking import construir_trades_topo

        meta = self._meta()
        score = np.tile(np.arange(20, dtype=float), 2)  # score = ordem do ticker
        custos = {f"T{i:02d}": 0.001 * (i + 1) for i in range(20)}

        out = construir_trades_topo(
            meta, score, frac=0.10, custos_por_ticker=custos, col_retorno="ret_fwd"
        )

        assert len(out) == 4  # 2 por dia × 2 dias
        assert set(out["ticker"]) == {"T18", "T19"}
        esperado = 0.01 * 19 - 0.001 * 20
        assert out.loc[out.ticker == "T19", "excesso_liquido"].iloc[0] == pytest.approx(esperado)

    def test_ticker_sem_custo_vira_nan_e_a_linha_permanece(self):
        """Descartar mudaria a frequência de sinal — justamente o que o
        controle aleatório precisa reproduzir. O NaN fica visível para a
        leitura reportar quantos trades ficaram sem custo."""
        from radar.cerebro.ranking import construir_trades_topo

        meta = self._meta(n_dias=1, n_ativos=20)
        score = np.arange(20, dtype=float)

        out = construir_trades_topo(
            meta, score, frac=0.10, custos_por_ticker={"T18": 0.002}, col_retorno="ret_fwd"
        )

        assert len(out) == 2
        assert out["excesso_liquido"].isna().sum() == 1  # T19 sem custo

    def test_controle_aleatorio_tem_a_mesma_frequencia(self):
        from radar.cerebro.ranking import construir_trades_topo

        meta = self._meta()
        score = np.tile(np.arange(20, dtype=float), 2)
        custos = {f"T{i:02d}": 0.002 for i in range(20)}

        topo = construir_trades_topo(
            meta, score, frac=0.10, custos_por_ticker=custos, col_retorno="ret_fwd"
        )
        aleatorio = construir_trades_topo(
            meta, score, frac=0.10, custos_por_ticker=custos, col_retorno="ret_fwd",
            modo=MODO_ALEATORIO, rng=np.random.default_rng(5),
        )

        assert len(topo) == len(aleatorio)

    def test_duplicata_de_ativo_instante_levanta(self):
        from radar.cerebro.ranking import construir_trades_topo

        meta = pd.DataFrame(
            [{"date": "2026-01-05", "ticker": "T", "ret_fwd": 0.1}] * 2
        )
        with pytest.raises(ValueError, match="duplicada"):
            construir_trades_topo(
                meta, np.array([1.0, 2.0]), frac=0.5,
                custos_por_ticker={"T": 0.002}, col_retorno="ret_fwd",
            )

    def test_tamanhos_diferentes_levantam(self):
        from radar.cerebro.ranking import construir_trades_topo

        with pytest.raises(ValueError, match="tamanhos diferentes"):
            construir_trades_topo(
                self._meta(), np.array([1.0]), frac=0.1,
                custos_por_ticker={}, col_retorno="ret_fwd",
            )

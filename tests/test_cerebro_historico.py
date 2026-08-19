"""
Testes — `radar.cerebro.historico` (a memória de leituras do Cérebro).

O que este módulo existe para impedir está no teste
`test_piora_entre_desenhos_e_detectada`: a família `swing_v1` mediu Rank IC
−0,0268 na mesma célula em que a Fase 1 tinha medido −0,0211, e a leitura foi
escrita sem confrontar os dois. **Piorou, e ninguém teria notado.** Sem
comparação com o histórico não existe auto-calibração — só uma sequência de
opiniões independentes sobre o mesmo mercado.

Zero I/O fora de `tmp_path`.
"""

from __future__ import annotations

import pytest

from radar.cerebro.historico import (
    DELTA_IRRELEVANTE,
    DIRECAO_IGUAL,
    DIRECAO_MELHOROU,
    DIRECAO_PIOROU,
    DIRECAO_PRIMEIRA,
    RegistroLeitura,
    carregar,
    comparar_com_historico,
    registrar,
    resumo_de_aprendizado,
)


def _leitura(ic: float, *, quando: str, desenho: str = "d", mercado: str = "cripto",
             horizonte: int = 21, familia: str = "f") -> RegistroLeitura:
    return RegistroLeitura(
        familia=familia, mercado=mercado, horizonte=horizonte, medido_em=quando,
        rank_ic_medio=ic, rank_ic_t=ic * 100, n_dias=2000, desenho=desenho, veredito="v",
    )


class TestPersistencia:
    def test_arquivo_ausente_devolve_vazio(self, tmp_path):
        assert carregar(tmp_path / "nao_existe.jsonl") == []

    def test_registrar_e_carregar_preserva_o_registro(self, tmp_path):
        caminho = tmp_path / "h.jsonl"
        r = _leitura(-0.02, quando="2026-08-01T00:00:00")

        registrar(caminho, r)

        assert carregar(caminho) == [r]

    def test_append_only_nao_sobrescreve(self, tmp_path):
        """Uma leitura ruim não pode ser apagada para o histórico ficar bonito
        — o contador de tentativas do ADR 0032 §9 depende de ser completo."""
        caminho = tmp_path / "h.jsonl"
        registrar(caminho, _leitura(-0.02, quando="2026-08-01T00:00:00"))
        registrar(caminho, _leitura(-0.03, quando="2026-08-04T00:00:00"))

        assert len(carregar(caminho)) == 2

    def test_linha_ilegivel_levanta(self, tmp_path):
        """Histórico parcial lido como completo faria uma PIORA parecer
        primeira leitura."""
        caminho = tmp_path / "h.jsonl"
        registrar(caminho, _leitura(-0.02, quando="2026-08-01T00:00:00"))
        with caminho.open("a", encoding="utf-8") as f:
            f.write("{quebrado\n")

        with pytest.raises(ValueError, match="ilegível"):
            carregar(caminho)


class TestComparacao:
    def test_primeira_leitura_do_eixo_nao_inventa_delta(self):
        c = comparar_com_historico(_leitura(-0.02, quando="2026-08-04T00:00:00"), [])

        assert c.direcao == DIRECAO_PRIMEIRA
        assert c.delta_rank_ic is None

    def test_piora_entre_desenhos_e_detectada(self):
        """O caso REAL: Fase 1 mediu −0,0211 em cripto/h21; swing_v1 mediu
        −0,0268 na mesma célula, com instrumento e alvo novos. Piorou."""
        antes = _leitura(-0.0211, quando="2026-08-01T04:21:31", desenho="alvo por limiar absoluto")
        agora = _leitura(-0.0268, quando="2026-08-04T06:00:00", desenho="alvo cross-sectional")

        c = comparar_com_historico(agora, [antes])

        assert c.direcao == DIRECAO_PIOROU
        assert c.delta_rank_ic == pytest.approx(-0.0057)
        assert "alvo cross-sectional" in c.licao and "limiar absoluto" in c.licao

    def test_melhora_e_detectada(self):
        c = comparar_com_historico(
            _leitura(0.03, quando="2026-08-04T00:00:00"),
            [_leitura(-0.02, quando="2026-08-01T00:00:00")],
        )
        assert c.direcao == DIRECAO_MELHOROU

    def test_delta_dentro_do_ruido_e_igual(self):
        """Chamar de melhora um delta menor que o ruído da medida é o mesmo
        erro do `t` inflado, na direção do otimismo."""
        antes = _leitura(-0.0200, quando="2026-08-01T00:00:00")
        agora = _leitura(-0.0200 + DELTA_IRRELEVANTE / 2, quando="2026-08-04T00:00:00")

        assert comparar_com_historico(agora, [antes]).direcao == DIRECAO_IGUAL

    def test_compara_com_a_melhor_anterior_nao_com_a_ultima(self):
        """Comparar com a última deixaria uma sequência de pioras pequenas
        passar por 'estável' enquanto a distância para o melhor cresce —
        mesmo motivo de champion/challenger comparar com o campeão."""
        melhor = _leitura(0.05, quando="2026-08-01T00:00:00", desenho="o bom")
        ultima = _leitura(-0.01, quando="2026-08-02T00:00:00", desenho="o ruim")
        agora = _leitura(0.00, quando="2026-08-04T00:00:00")

        c = comparar_com_historico(agora, [melhor, ultima])

        assert c.direcao == DIRECAO_PIOROU
        assert c.anterior is melhor

    def test_eixos_diferentes_nao_se_comparam(self):
        """B3/h5 e cripto/h21 não são o mesmo experimento — comparar produziria
        um número enganoso."""
        outro_eixo = _leitura(0.05, quando="2026-08-01T00:00:00", mercado="b3", horizonte=5)

        c = comparar_com_historico(_leitura(-0.02, quando="2026-08-04T00:00:00"), [outro_eixo])

        assert c.direcao == DIRECAO_PRIMEIRA

    def test_leitura_posterior_nao_entra_como_anterior(self):
        futura = _leitura(0.05, quando="2026-12-01T00:00:00")

        c = comparar_com_historico(_leitura(-0.02, quando="2026-08-04T00:00:00"), [futura])

        assert c.direcao == DIRECAO_PRIMEIRA


class TestResumoDeAprendizado:
    def test_conta_tentativas_por_eixo(self):
        hist = [
            _leitura(-0.01, quando="2026-08-01T00:00:00"),
            _leitura(-0.02, quando="2026-08-02T00:00:00"),
            _leitura(+0.01, quando="2026-08-03T00:00:00", mercado="b3", horizonte=5),
        ]

        r = resumo_de_aprendizado(hist)

        assert r["n_leituras"] == 3
        assert r["n_eixos"] == 2
        assert r["eixos"]["cripto/h21"]["tentativas"] == 2

    def test_marca_quando_nenhum_desenho_ja_deu_positivo(self):
        """O painel que nenhuma leitura isolada mostra: 6 eixos, 15 tentativas,
        e `algum_positivo=False` em todos."""
        hist = [_leitura(-0.01, quando="2026-08-01T00:00:00"),
                _leitura(-0.02, quando="2026-08-02T00:00:00")]

        r = resumo_de_aprendizado(hist)

        assert r["eixos"]["cripto/h21"]["algum_positivo"] is False

    def test_guarda_o_melhor_desenho_ja_visto(self):
        hist = [
            _leitura(-0.05, quando="2026-08-01T00:00:00", desenho="ruim"),
            _leitura(-0.01, quando="2026-08-02T00:00:00", desenho="menos ruim"),
        ]

        r = resumo_de_aprendizado(hist)

        assert r["eixos"]["cripto/h21"]["melhor_desenho"] == "menos ruim"

    def test_historico_vazio_nao_quebra(self):
        r = resumo_de_aprendizado([])
        assert r == {"n_leituras": 0, "n_eixos": 0, "tentativas_totais": 0, "eixos": {}}

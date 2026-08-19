"""
O Evidence Research Loop — e a trava que impede ele de virar p-hacking.

O que estes testes protegem, em ordem de importância:

1. **`data_cutoff` igual = rodada REJEITADA.** Re-medir o mesmo histórico não é
   uma segunda observação, é a mesma contada duas vezes.
2. **`FALTA_METRICA` executa já e não gasta multiplicidade**; `FALTA_AMOSTRA`
   espera e gasta. São custos opostos e não podem ser confundidos.
3. **Métrica ANTES de amostra**, sempre — medir é de graça e pode reprovar por
   outro caminho, poupando meses de espera.
4. **O denominador do FDR conta RODADAS, não hipóteses.**
"""

from __future__ import annotations

import pytest

from radar.cerebro.mapa_validade import (
    Dimensao,
    Direcao,
    Estado,
    EvidenciaOOS,
    classificar,
)
from radar.cerebro.pesquisa import (
    MINIMO_DIAS_NOVOS,
    Decisao,
    Necessidade,
    RegistroDeRodadas,
    Rodada,
    avaliar_necessidade,
)


def _veredito(**kw):
    base = dict(
        hipotese="h", mercado="b3", horizonte=21, trades=3000, dias_independentes=400,
        excesso_liquido=0.004, t_nw=3.1, direcao=Direcao.LONG, anos_positivos=9,
        anos_testados=10, efeito_minimo_relevante=0.001, equilibrio_slippage_bps=20.0,
        estavel_oos=True, capacidade_estimada=5_000_000.0, capital_alvo=100_000.0,
        risco_dentro_da_tolerancia=True, max_drawdown=0.10,
    )
    return classificar(EvidenciaOOS(**{**base, **kw}))


def _rodada(cutoff="2026-05-01", dias=400, n=1, **kw):
    base = dict(research_run_id="R001", hipotese="h", data_cutoff=cutoff,
                dias_independentes=dias, rodada_n=n, estado="nao_comprovado")
    return Rodada(**{**base, **kw})


class TestBifurcacaoMetricaVersusAmostra:
    def test_falta_metrica_executa_ja_e_nao_gasta_multiplicidade(self):
        """
        Capacidade/estabilidade não medidas são pergunta que nunca foi feita
        sobre o dado que já existe — não repetem teste nenhum.
        """
        d = avaliar_necessidade(
            _veredito(capacidade_estimada=None, estavel_oos=None),
            dias_independentes=400, ultima=_rodada(),
        )
        assert d.necessidade is Necessidade.FALTA_METRICA
        assert d.executar_agora
        assert not d.necessidade.custa_multiplicidade
        assert set(d.metricas_faltando) == {"capacidade", "estabilidade_oos"}

    def test_falta_amostra_espera_e_gasta_multiplicidade(self):
        d = avaliar_necessidade(
            _veredito(t_nw=1.0), dias_independentes=410, ultima=_rodada(dias=400),
        )
        assert d.necessidade is Necessidade.FALTA_AMOSTRA
        assert not d.executar_agora
        assert d.necessidade.custa_multiplicidade

    def test_metrica_vem_antes_de_amostra(self):
        """
        Com métrica faltando E amostra curta, mede-se a métrica primeiro: é de
        graça e pode reprovar por outro caminho, poupando a espera inteira.
        Inverter gastaria meses esperando dado para descobrir que a capacidade
        era R$ 300.
        """
        d = avaliar_necessidade(
            _veredito(t_nw=1.0, capacidade_estimada=None),
            dias_independentes=401, ultima=_rodada(dias=400),
        )
        assert d.necessidade is Necessidade.FALTA_METRICA
        assert d.executar_agora

    def test_refutada_nao_gasta_rodada(self):
        d = avaliar_necessidade(
            _veredito(t_nw=0.1, excesso_liquido=1e-5, efeito_minimo_relevante=0.5),
            dias_independentes=400,
        )
        assert d.necessidade is Necessidade.ENCERRADA
        assert not d.executar_agora
        assert "informação NOVA" in d.acao

    def test_sem_criterio_faltando_nao_ha_o_que_fazer(self):
        d = avaliar_necessidade(_veredito(), dias_independentes=400, ultima=_rodada())
        assert d.necessidade is Necessidade.NENHUMA
        assert not d.executar_agora


class TestTravaDeDataCutoff:
    """A trava que separa evidência nova de re-contagem do mesmo histórico."""

    def test_avanco_insuficiente_de_corte_bloqueia(self):
        d = avaliar_necessidade(
            _veredito(t_nw=1.0),
            dias_independentes=400 + MINIMO_DIAS_NOVOS - 1, ultima=_rodada(dias=400),
        )
        assert not d.executar_agora
        assert "MESMO histórico" in d.motivo

    def test_avanco_suficiente_libera_e_avisa_do_custo(self):
        d = avaliar_necessidade(
            _veredito(t_nw=1.0),
            dias_independentes=400 + MINIMO_DIAS_NOVOS, ultima=_rodada(dias=400, n=2),
        )
        assert d.executar_agora
        assert d.dias_novos_desde_ultima == MINIMO_DIAS_NOVOS
        assert "CONTA no denominador do FDR" in d.motivo
        assert "rodada 3" in d.motivo

    def test_primeira_rodada_sempre_executa(self):
        d = avaliar_necessidade(_veredito(t_nw=1.0), dias_independentes=400, ultima=None)
        assert d.executar_agora and d.rodadas_gastas == 0

    def test_registrar_com_mesmo_cutoff_levanta(self, tmp_path):
        """
        A trava dura. Sem ela, o cron semanal registraria 52 "observações" por
        ano sobre o mesmo histórico — p-hacking automatizado com aparência de
        rigor.
        """
        reg = RegistroDeRodadas.carregar(tmp_path / "r.jsonl")
        reg.registrar(_rodada(cutoff="2026-05-01", n=1))
        with pytest.raises(ValueError, match="mesma observação contada duas vezes"):
            reg.registrar(_rodada(cutoff="2026-05-01", n=2))

    def test_cutoff_diferente_e_aceito(self, tmp_path):
        reg = RegistroDeRodadas.carregar(tmp_path / "r.jsonl")
        reg.registrar(_rodada(cutoff="2026-05-01", n=1))
        reg.registrar(_rodada(cutoff="2026-08-01", dias=460, n=2))
        assert reg.tentativas_de("h") == 2
        assert reg.ultima_de("h").rodada_n == 2


class TestRegistroAppendOnly:
    def test_persiste_e_recarrega(self, tmp_path):
        p = tmp_path / "r.jsonl"
        reg = RegistroDeRodadas.carregar(p)
        reg.registrar(_rodada(cutoff="2026-05-01", n=1))
        reg.registrar(_rodada(cutoff="2026-08-01", dias=460, n=2, hipotese="outra"))
        assert RegistroDeRodadas.carregar(p).tentativas_de("h") == 1
        assert RegistroDeRodadas.carregar(p).denominador_fdr() == 2

    def test_denominador_conta_rodadas_e_nao_hipoteses(self, tmp_path):
        """
        Uma hipótese re-medida 3× gastou 3 testes. Contar hipóteses tornaria o
        corte de Benjamini-Hochberg artificialmente generoso exatamente na
        direção que favorece o pesquisador.
        """
        reg = RegistroDeRodadas.carregar(tmp_path / "r.jsonl")
        for i, cut in enumerate(("2026-01-01", "2026-04-01", "2026-08-01"), start=1):
            reg.registrar(_rodada(cutoff=cut, dias=400 + 60 * i, n=i))
        assert reg.tentativas_de("h") == 3
        assert reg.denominador_fdr() == 3, "3 tentativas, não 1 hipótese"

    def test_ultima_de_hipotese_inexistente_e_none(self, tmp_path):
        reg = RegistroDeRodadas.carregar(tmp_path / "r.jsonl")
        assert reg.ultima_de("nao_existe") is None
        assert reg.tentativas_de("nao_existe") == 0


class TestVocabulario:
    def test_so_falta_amostra_custa_multiplicidade(self):
        assert [n for n in Necessidade if n.custa_multiplicidade] == [
            Necessidade.FALTA_AMOSTRA
        ]

    def test_toda_necessidade_tem_acao(self):
        assert all(n.acao for n in Necessidade)

    def test_decisao_serializa_o_custo(self):
        d = avaliar_necessidade(
            _veredito(capacidade_estimada=None), dias_independentes=400
        ).to_dict()
        assert d["custa_multiplicidade"] is False
        assert d["executar_agora"] is True

    def test_dimensao_nao_medida_e_o_gatilho_de_falta_metrica(self):
        """O gatilho é `NAO_MEDIDA`, nunca `NAO` — a diferença é o módulo todo."""
        v = _veredito(capacidade_estimada=None)
        assert v.dimensoes["capacidade"] is Dimensao.NAO_MEDIDA
        assert avaliar_necessidade(
            v, dias_independentes=400
        ).necessidade is Necessidade.FALTA_METRICA

        v2 = _veredito(capacidade_estimada=1.0)
        assert v2.dimensoes["capacidade"] is Dimensao.NAO
        assert avaliar_necessidade(
            v2, dias_independentes=400
        ).necessidade is not Necessidade.FALTA_METRICA

    def test_decisao_e_imutavel(self):
        d = avaliar_necessidade(_veredito(), dias_independentes=400)
        assert isinstance(d, Decisao)
        with pytest.raises(AttributeError):
            d.executar_agora = True  # type: ignore[misc]

    def test_estado_refutado_curto_circuita_antes_das_metricas(self):
        """Refutada com métrica faltando ainda assim não gasta rodada."""
        d = avaliar_necessidade(
            _veredito(t_nw=0.1, excesso_liquido=1e-5, efeito_minimo_relevante=0.5,
                      capacidade_estimada=None),
            dias_independentes=400,
        )
        assert d.necessidade is Necessidade.ENCERRADA
        assert Estado.REFUTADO is Estado.REFUTADO  # vocabulário estável


class TestNaoEstimavelNaoCaiEmNenhuma:
    """
    Terceira aparição da MESMA classe de bug nesta sessão: `nao_estimavel` não
    produz dimensões, e todo default que pergunta "falta alguma coisa?" responde
    "não" — declarando sucesso justamente sobre a hipótese que menos foi medida.

    Aconteceu em `bloqueio_dominante` (saía `NENHUM`), em `persistencia_de`
    (critério pulado em silêncio) e aqui. O padrão é: **ausência de medição
    imitando aprovação**.
    """

    def test_nao_estimavel_e_falta_amostra_e_nao_nenhuma(self):
        d = avaliar_necessidade(_veredito(trades=10), dias_independentes=8)
        assert d.necessidade is Necessidade.FALTA_AMOSTRA
        assert d.necessidade is not Necessidade.NENHUMA
        assert "não estimável" in d.motivo
        assert "Precisa de DADO" in d.motivo

    def test_nao_estimavel_nao_executa_agora_e_conta_no_fdr(self):
        d = avaliar_necessidade(_veredito(dias_independentes=5, trades=60),
                                dias_independentes=5)
        assert not d.executar_agora
        assert d.necessidade.custa_multiplicidade

    def test_nao_estimavel_vence_ate_metrica_faltando(self):
        """Sem estatística, medir capacidade não muda nada — falta é de DADO."""
        d = avaliar_necessidade(
            _veredito(trades=10, capacidade_estimada=None, estavel_oos=None),
            dias_independentes=8,
        )
        assert d.necessidade is Necessidade.FALTA_AMOSTRA

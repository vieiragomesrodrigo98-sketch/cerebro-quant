"""
A oportunidade como unidade — e o dedup que a torna medível.

O que estes testes protegem:

1. **Oportunidade sem fim declarado não existe.** Sem condição de invalidação
   ela nunca termina e o ativo fica travado para sempre (§8.3 trava 4).
2. **O `opportunity_id` é DETERMINÍSTICO.** Um `uuid4()` faria a reexecução de
   um dia gerar oportunidades "novas" e inflar a contagem — o defeito exato que
   o dedup existe para impedir.
3. **Dedup mantém a PRIMEIRA de cada balde, nunca a melhor.** Escolher a melhor
   seria olhar o resultado para decidir o que contar.
4. **Ordem pendente ocupa o ativo.** É o que obriga a ordem a ter expiração:
   uma que nunca preenche sequestraria o ativo indefinidamente.

O número que justifica tudo isto: quando o clustering por dia foi corrigido
neste projeto, o `t` caiu de **5,08 para 0,30** (S142-143).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from radar.cerebro.oportunidade import (
    MOTIVOS_DE_FIM,
    Acao,
    Estado,
    Oportunidade,
    chave_de_setup,
    dedup_por_setup,
)

T0 = datetime(2026, 8, 6, 12, 0, tzinfo=UTC)


def _op(**kw) -> Oportunidade:
    base = dict(ativo="BTCUSDT", motor="scalp_cripto_v2", criada_em=T0,
                invalidacao=MOTIVOS_DE_FIM)
    return Oportunidade(**{**base, **kw})


class TestFimDeclarado:
    def test_sem_invalidacao_levanta(self):
        with pytest.raises(ValueError, match="sem condição de invalidação"):
            _op(invalidacao=())

    def test_fechar_exige_motivo(self):
        with pytest.raises(ValueError, match="exige motivo"):
            _op().com_estado(Estado.FECHADA, em=T0)

    def test_fechar_com_motivo_nao_declarado_levanta(self):
        """O fim tem de ser um dos que a própria oportunidade declarou."""
        with pytest.raises(ValueError, match="não está entre as invalidações"):
            _op().com_estado(Estado.FECHADA, em=T0, motivo="mudei_de_ideia")

    def test_fechar_com_motivo_declarado_funciona(self):
        f = _op().com_estado(Estado.FECHADA, em=T0, motivo="stop_tocado")
        assert f.estado is Estado.FECHADA
        assert f.motivo_fim == "stop_tocado"
        assert not f.viva


class TestIdentidade:
    def test_id_e_deterministico(self):
        """Recalcular o mesmo dia não pode gerar oportunidade 'nova'."""
        assert _op().id == _op().id

    def test_motores_diferentes_sao_oportunidades_diferentes(self):
        """Duas famílias no mesmo ativo NÃO são a mesma oportunidade."""
        assert _op(motor="scalp_cripto_v2").id != _op(motor="position_cripto_v1").id

    def test_ativos_diferentes_sao_oportunidades_diferentes(self):
        assert _op(ativo="BTCUSDT").id != _op(ativo="ETHUSDT").id

    def test_instantes_diferentes_sao_oportunidades_diferentes(self):
        assert _op(criada_em=T0).id != _op(criada_em=T0 + timedelta(minutes=1)).id


class TestOcupacaoDoAtivo:
    def test_ordem_pendente_ocupa_o_ativo(self):
        """E é por isso que ordem pendente precisa de expiração."""
        assert Estado.ORDEM.ocupa_o_ativo is True

    def test_posicao_aberta_ocupa(self):
        assert Estado.ABERTA.ocupa_o_ativo is True

    def test_criada_ainda_nao_ocupa(self):
        assert Estado.CRIADA.ocupa_o_ativo is False

    def test_fechada_libera(self):
        assert Estado.FECHADA.ocupa_o_ativo is False


class TestDedupEstatistico:
    """§8.3 trava 1 — o que separa 5 ordens de 5 setups."""

    def _cinco_no_mesmo_dia(self) -> list[Oportunidade]:
        return [_op(criada_em=T0 + timedelta(hours=h)) for h in range(5)]

    def test_cinco_oportunidades_no_mesmo_balde_viram_uma(self):
        r = dedup_por_setup(self._cinco_no_mesmo_dia(), janela_de=lambda o: o.criada_em.date())
        assert len(r) == 1

    def test_mantem_a_primeira_nunca_a_melhor(self):
        """Escolher a melhor seria olhar o resultado para decidir o que contar."""
        r = dedup_por_setup(self._cinco_no_mesmo_dia(), janela_de=lambda o: o.criada_em.date())
        assert r[0].criada_em == T0

    def test_baldes_diferentes_contam_separado(self):
        ops = [_op(criada_em=T0), _op(criada_em=T0 + timedelta(days=1))]
        r = dedup_por_setup(ops, janela_de=lambda o: o.criada_em.date())
        assert len(r) == 2

    def test_motores_diferentes_nao_deduplicam_entre_si(self):
        """Scalp e Position no mesmo ativo e dia são setups independentes."""
        ops = [_op(motor="scalp_cripto_v2"), _op(motor="position_cripto_v1")]
        r = dedup_por_setup(ops, janela_de=lambda o: o.criada_em.date())
        assert len(r) == 2

    def test_ativos_diferentes_nao_deduplicam_entre_si(self):
        ops = [_op(ativo="BTCUSDT"), _op(ativo="ETHUSDT")]
        r = dedup_por_setup(ops, janela_de=lambda o: o.criada_em.date())
        assert len(r) == 2

    def test_a_chave_e_motor_ativo_janela(self):
        assert chave_de_setup(_op(), janela="2026-08-06") == (
            "scalp_cripto_v2", "BTCUSDT", "2026-08-06")

    def test_lista_vazia_nao_quebra(self):
        assert dedup_por_setup([], janela_de=lambda o: None) == []


class TestVocabularioDeAcoes:
    def test_as_oito_acoes_do_adr_existem(self):
        assert {a.value for a in Acao} == {
            "abrir", "aumentar", "manter", "reduzir",
            "realizar_parcial", "mover_stop", "fechar", "reentrar",
        }

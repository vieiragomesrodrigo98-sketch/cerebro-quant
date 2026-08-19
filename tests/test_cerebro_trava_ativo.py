"""
A trava exclusiva por ativo — e a métrica que o ADR exige junto com ela.

O que estes testes protegem:

1. **Uma posição lógica por ativo por conta**, que é exatamente uma posição
   física na exchange. É o que dispensa hedge mode, subcontas e netting.
2. **O dono continua podendo.** A trava impede que OUTRO motor entre, não
   congela o ativo — reforço e reentrada são decisões da oportunidade.
3. **Ordem pendente expira.** Sem expiração, uma ordem que nunca preenche
   sequestra o ativo para sempre; com ela, o ativo volta.
4. **O custo de oportunidade é medido, não assumido.** O ADR manda medir quantos
   sinais morreram por ativo travado por outro ANTES de discutir prioridade por
   motor — inventar prioridade sem o número seria complexidade não medida.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from radar.cerebro.trava_ativo import CustoDeOportunidade, TravasPorAtivo

T0 = datetime(2026, 8, 6, 12, 0, tzinfo=UTC)
SCALP = "scalp_cripto_v2"
POSITION = "position_cripto_v1"


class TestExclusividade:
    def test_ativo_livre_qualquer_motor_opera(self):
        t = TravasPorAtivo()
        assert t.pode_operar("BTCUSDT", SCALP, agora=T0)[0] is True

    def test_outro_motor_e_bloqueado(self):
        t = TravasPorAtivo()
        t.travar(ativo="BTCUSDT", motor=SCALP, oportunidade_id="a1", agora=T0)
        pode, motivo = t.pode_operar("BTCUSDT", POSITION, agora=T0)
        assert pode is False
        assert SCALP in motivo

    def test_o_proprio_dono_continua_podendo(self):
        """A trava impede OUTRO motor, não congela o ativo: reforço e reentrada
        são decisões da oportunidade."""
        t = TravasPorAtivo()
        t.travar(ativo="BTCUSDT", motor=SCALP, oportunidade_id="a1", agora=T0)
        assert t.pode_operar("BTCUSDT", SCALP, agora=T0)[0] is True

    def test_outro_ativo_nao_e_afetado(self):
        t = TravasPorAtivo()
        t.travar(ativo="BTCUSDT", motor=SCALP, oportunidade_id="a1", agora=T0)
        assert t.pode_operar("ETHUSDT", POSITION, agora=T0)[0] is True

    def test_travar_ativo_de_outro_motor_levanta(self):
        """Chegar aqui sem passar por pode_operar é erro de programação —
        retorno silencioso deixaria o chamador achar que travou."""
        t = TravasPorAtivo()
        t.travar(ativo="BTCUSDT", motor=SCALP, oportunidade_id="a1", agora=T0)
        with pytest.raises(ValueError, match="já travado por"):
            t.travar(ativo="BTCUSDT", motor=POSITION, oportunidade_id="b1", agora=T0)


class TestLiberacao:
    def test_liberar_devolve_o_ativo_para_qualquer_motor(self):
        t = TravasPorAtivo()
        t.travar(ativo="BTCUSDT", motor=SCALP, oportunidade_id="a1", agora=T0)
        t.liberar("BTCUSDT")
        assert t.pode_operar("BTCUSDT", POSITION, agora=T0)[0] is True

    def test_liberar_ativo_livre_e_no_op(self):
        """Fechar duas vezes não pode quebrar o fluxo de fechamento.

        "No-op" tem duas metades: não levanta **e** não deixa o ativo em estado
        estranho. Sem a asserção, o teste passaria mesmo que `liberar` travasse o
        ativo em vez de soltá-lo (QA-010).
        """
        t = TravasPorAtivo()
        t.liberar("BTCUSDT")
        t.liberar("BTCUSDT")

        assert t.pode_operar("BTCUSDT", POSITION, agora=T0)[0] is True
        assert t.pode_operar("BTCUSDT", SCALP, agora=T0)[0] is True


class TestExpiracaoDeOrdemPendente:
    def test_ordem_pendente_expira_e_libera(self):
        t = TravasPorAtivo()
        t.travar(ativo="BTCUSDT", motor=SCALP, oportunidade_id="a1", agora=T0,
                 expira_apos=timedelta(hours=1))
        assert t.pode_operar("BTCUSDT", POSITION, agora=T0)[0] is False
        depois = T0 + timedelta(hours=2)
        assert t.pode_operar("BTCUSDT", POSITION, agora=depois)[0] is True

    def test_posicao_aberta_nao_expira(self):
        """Sai quando fecha, e o fechamento tem motivo declarado."""
        t = TravasPorAtivo()
        t.travar(ativo="BTCUSDT", motor=SCALP, oportunidade_id="a1", agora=T0)
        muito_depois = T0 + timedelta(days=365)
        assert t.pode_operar("BTCUSDT", POSITION, agora=muito_depois)[0] is False

    def test_preencher_remove_a_expiracao(self):
        t = TravasPorAtivo()
        t.travar(ativo="BTCUSDT", motor=SCALP, oportunidade_id="a1", agora=T0,
                 expira_apos=timedelta(hours=1))
        t.promover_para_posicao("BTCUSDT")
        depois = T0 + timedelta(days=30)
        assert t.dono("BTCUSDT", agora=depois) == SCALP


class TestCustoDeOportunidade:
    """§8.3 — métrica OBRIGATÓRIA, não telemetria opcional."""

    def test_bloqueio_e_contabilizado(self):
        t = TravasPorAtivo()
        t.travar(ativo="BTCUSDT", motor=SCALP, oportunidade_id="a1", agora=T0)
        t.pode_operar("BTCUSDT", POSITION, agora=T0)
        t.pode_operar("BTCUSDT", POSITION, agora=T0)
        assert t.custo.total == 2
        assert t.custo.por_motor_bloqueado() == {POSITION: 2}
        assert t.custo.por_motor_dono() == {SCALP: 2}

    def test_o_dono_operando_nao_conta_como_custo(self):
        t = TravasPorAtivo()
        t.travar(ativo="BTCUSDT", motor=SCALP, oportunidade_id="a1", agora=T0)
        t.pode_operar("BTCUSDT", SCALP, agora=T0)
        assert t.custo.total == 0

    def test_ativo_livre_nao_conta_como_custo(self):
        t = TravasPorAtivo()
        t.pode_operar("BTCUSDT", SCALP, agora=T0)
        assert t.custo.total == 0

    def test_mede_o_motor_mais_rapido_ganhando_o_ativo(self):
        """A consequência que o ADR manda medir: com cadências diferentes, o
        scalp trava o ativo antes de o position sequer olhar."""
        custo = CustoDeOportunidade()
        t = TravasPorAtivo(custo)
        for ativo in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            t.travar(ativo=ativo, motor=SCALP, oportunidade_id=f"o{ativo}", agora=T0)
            t.pode_operar(ativo, POSITION, agora=T0)
        assert custo.por_motor_dono() == {SCALP: 3}
        assert custo.to_dict()["total"] == 3

"""
Testes do guardrail (`radar.cerebro.politica`) — emendas 1, 3, 5 e 7 do ADR 0037.

Três travas, e nenhuma delas é sobre o caminho feliz:

1. **`EMITIR_SINAL` exige FDR.** Sem isso, o portão de promoção não existe.
2. **Hipótese é imutável no lote em execução.** É a brecha pela qual autonomia
   viraria garimpo automatizado.
3. **`NENHUMA_ACAO` é resposta, não silêncio** — e carrega motivo.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from radar.cerebro.orcamento import Orcamento
from radar.cerebro.politica import (
    Acao,
    AcaoPermitida,
    Evento,
    HipoteseImutavelError,
    Missao,
    Validade,
    exigir_hipotese_imutavel,
    proxima_acao,
)
from radar.cerebro.teses import Tese

_T0 = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)
_COM_ESPACO = Orcamento(teto=180, gasto=57, alvo="estrategia_alfa", q=0.05)
_ESGOTADO = Orcamento(teto=57, gasto=57, alvo="estrategia_alfa", q=0.05)


def _tese(**ajustes) -> Tese:
    base = {
        "identificador": "t1",
        "assunto": "estrategia_alfa × bull_lateral",
        "motivo": "amostra insuficiente",
        "condicao_de_despertar": "15 dias independentes",
        "criada_em": _T0,
        "acordar_em": _T0 + timedelta(days=30),
        "amostra_exigida": 15,
        "amostra_atual": 9,
    }
    return Tese(**{**base, **ajustes})


class TestEmitirExigeFDR:
    """A trava nº 1 — o portão de promoção."""

    def test_emite_quando_sobrevive_ao_fdr(self) -> None:
        r = proxima_acao(
            evento=Evento.BARRA_1S,
            missao=Missao.EXECUCAO,
            orcamento=_COM_ESPACO,
            assunto="btc",
            validade=Validade("scalp_v3", sobrevive_fdr=True, estado="comprovado"),
        )
        assert r.acao is Acao.EMITIR_SINAL

    def test_nao_emite_quando_nao_sobrevive_ao_fdr(self) -> None:
        r = proxima_acao(
            evento=Evento.BARRA_1S,
            missao=Missao.EXECUCAO,
            orcamento=_COM_ESPACO,
            assunto="btc",
            validade=Validade("scalp_v3", sobrevive_fdr=False, estado="nao_comprovado"),
        )
        assert r.acao is Acao.NENHUMA_ACAO
        assert "não sobrevive ao FDR" in r.motivo

    def test_nao_emite_sobre_assunto_sem_entrada_no_mapa(self) -> None:
        """Emitir sobre assunto não medido é afirmar sem evidência — e o
        default fechado é o que impede a lacuna de virar permissão."""
        r = proxima_acao(
            evento=Evento.BARRA_1S,
            missao=Missao.EXECUCAO,
            orcamento=_COM_ESPACO,
            assunto="moeda_qualquer",
            validade=None,
        )
        assert r.acao is Acao.NENHUMA_ACAO
        assert "sem entrada no mapa de validade" in r.motivo

    def test_missao_de_pesquisa_nunca_emite(self) -> None:
        """Emenda 5: são objetivos diferentes. Pesquisa que emite é pesquisa que
        virou produto sem passar pela escada de promoção."""
        r = proxima_acao(
            evento=Evento.BARRA_1S,
            missao=Missao.PESQUISA,
            orcamento=_COM_ESPACO,
            assunto="btc",
            validade=Validade("scalp_v3", sobrevive_fdr=True, estado="comprovado"),
        )
        assert r.acao is not Acao.EMITIR_SINAL


class TestHipoteseImutavel:
    """A trava nº 2 — emenda 3."""

    def test_alterar_hipotese_do_lote_em_execucao_e_recusado(self) -> None:
        with pytest.raises(HipoteseImutavelError, match="fila de pesquisa"):
            exigir_hipotese_imutavel(
                hipotese="scalp_cripto_v3_h1",
                lote_em_execucao=["scalp_cripto_v3_h1", "scalp_cripto_v3_h5"],
            )

    def test_hipotese_fora_do_lote_pode_ser_definida(self) -> None:
        """A guarda devolve `None` — e o `is None` NÃO é decoração: sem ele o
        teste passaria mesmo se a função virasse um `pass`, provando nada."""
        assert exigir_hipotese_imutavel(
            hipotese="ideia_nova", lote_em_execucao=["scalp_cripto_v3_h1"]
        ) is None

    def test_sem_lote_em_execucao_nada_esta_travado(self) -> None:
        assert exigir_hipotese_imutavel(
            hipotese="qualquer", lote_em_execucao=None
        ) is None

    def test_a_mensagem_ensina_o_caminho_certo(self) -> None:
        """Recusa que não diz o que fazer no lugar vira contorno criativo."""
        with pytest.raises(HipoteseImutavelError, match="pré-registre um lote NOVO"):
            exigir_hipotese_imutavel(hipotese="x", lote_em_execucao=["x"])


class TestEventoNaoEOrdem:
    """A trava nº 3 — emenda 7."""

    def test_barra_fechada_pode_terminar_em_nenhuma_acao(self) -> None:
        """`BARRA_FECHOU → EXECUTAR` seria event handler. O Cérebro consulta
        estado, tese, validade e orçamento antes."""
        r = proxima_acao(
            evento=Evento.BARRA_1S, missao=Missao.PESQUISA, orcamento=_COM_ESPACO
        )
        assert r.acao is Acao.NENHUMA_ACAO
        assert r.age is False

    def test_nenhuma_acao_sempre_carrega_motivo(self) -> None:
        """Ignorar grátis E invisível faria o agente ignorar em todo lugar e
        parecer seletivo — a mudez do emissor v1 com nome melhor."""
        r = proxima_acao(
            evento=Evento.NOTICIA_CLASSIFICADA,
            missao=Missao.PESQUISA,
            orcamento=_COM_ESPACO,
        )
        assert r.motivo.strip()

    def test_acao_sem_motivo_e_impossivel_de_construir(self) -> None:
        with pytest.raises(ValueError, match="decisão sem rastro"):
            AcaoPermitida(acao=Acao.NENHUMA_ACAO, motivo="  ")

    def test_nunca_devolve_none(self) -> None:
        for evento in Evento:
            for missao in Missao:
                r = proxima_acao(evento=evento, missao=missao, orcamento=_COM_ESPACO)
                assert isinstance(r, AcaoPermitida)


class TestPrecedencia:
    """A ordem das cláusulas É a política."""

    def test_tese_desperta_vem_antes_do_evento_novo(self) -> None:
        """O que já se decidiu esperar tem precedência sobre o que acabou de
        acontecer — senão o Cérebro reage ao mais recente e nunca fecha nada.
        É a diferença entre ter missão e ter reflexo."""
        r = proxima_acao(
            evento=Evento.BARRA_1S,
            missao=Missao.EXECUCAO,
            orcamento=_COM_ESPACO,
            teses_despertas=[_tese(amostra_atual=15)],
            validade=Validade("x", sobrevive_fdr=True),
        )
        assert r.acao is Acao.AVALIAR_TESE

    def test_tese_desperta_sem_amostra_renova_a_espera(self) -> None:
        r = proxima_acao(
            evento=Evento.PRAZO_DE_TESE,
            missao=Missao.PESQUISA,
            orcamento=_COM_ESPACO,
            teses_despertas=[_tese(amostra_atual=9)],
        )
        assert r.acao is Acao.AGUARDAR_AMOSTRA
        assert "9/15" in r.motivo

    def test_desfecho_atualiza_o_estado(self) -> None:
        """O laço de retorno do ADR 0036 — hoje o Ledger produz mapa e o motor
        nunca lê."""
        r = proxima_acao(
            evento=Evento.DESFECHO_NO_LEDGER,
            missao=Missao.EXECUCAO,
            orcamento=_COM_ESPACO,
            validade=Validade("x", sobrevive_fdr=True),
        )
        assert r.acao is Acao.ATUALIZAR_ESTADO

    def test_medir_metrica_faltante_vem_antes_do_que_custa(self) -> None:
        """`pesquisa.avaliar_necessidade` já distingue *falta métrica* (medir já,
        custo zero) de *falta amostra* (esperar, custa FDR)."""
        r = proxima_acao(
            evento=Evento.BARRA_DIARIA,
            missao=Missao.PESQUISA,
            orcamento=_COM_ESPACO,
            ha_metrica_faltando=True,
        )
        assert r.acao is Acao.MEDIR_METRICA
        assert "não custa multiplicidade" in r.motivo


class TestOrcamentoEsgotado:
    def test_orcamento_esgotado_impede_acao_de_pesquisa(self) -> None:
        r = proxima_acao(
            evento=Evento.BARRA_DIARIA, missao=Missao.PESQUISA, orcamento=_ESGOTADO
        )
        assert r.acao is Acao.NENHUMA_ACAO
        assert "orçamento estatístico esgotado" in r.motivo

    def test_medir_metrica_continua_permitido_com_orcamento_zerado(self) -> None:
        """Medir sobre dado existente não é tentativa nova. Bloquear isso seria
        confundir *terminar de olhar* com *olhar de novo*."""
        r = proxima_acao(
            evento=Evento.BARRA_DIARIA,
            missao=Missao.PESQUISA,
            orcamento=_ESGOTADO,
            ha_metrica_faltando=True,
        )
        assert r.acao is Acao.MEDIR_METRICA


class TestPoliticaNaoTemEstado:
    def test_o_modulo_nao_escreve_em_lugar_nenhum(self) -> None:
        """Guardrail com efeito colateral deixa de ser auditável por leitura.

        Lê a árvore sintática: nenhuma chamada a `open`, e nenhum import de
        módulo de I/O ou de banco.
        """
        import ast
        import inspect

        from radar.cerebro import politica

        arvore = ast.parse(inspect.getsource(politica))
        chamadas = {
            no.func.id
            for no in ast.walk(arvore)
            if isinstance(no, ast.Call) and isinstance(no.func, ast.Name)
        }
        assert "open" not in chamadas
        importados = {
            alias.name.split(".")[0]
            for no in ast.walk(arvore)
            if isinstance(no, ast.Import)
            for alias in no.names
        } | {
            (no.module or "").split(".")[0]
            for no in ast.walk(arvore)
            if isinstance(no, ast.ImportFrom)
        }
        assert not importados & {"sqlite3", "sqlalchemy", "pathlib", "os", "requests"}

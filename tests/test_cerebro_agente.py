"""
Testes do ciclo cognitivo (`radar.cerebro.agente` + `estado`) — ADR 0037.

A pergunta que estes testes respondem é a que separa agente de pipeline:
**o Cérebro lembra do ciclo anterior?** Se `test_o_cerebro_nao_recomeca_do_zero`
falhar, voltou a ser função pura chamada repetidamente.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from radar.cerebro.agente import Percepcao, ciclo
from radar.cerebro.estado import Estado
from radar.cerebro.orcamento import HipoteseNaGrade
from radar.cerebro.politica import Acao, Evento, Missao, Validade
from radar.cerebro.teses import EstadoDaTese, Tese, TesesEmMemoria

_T0 = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)
_GRADE = (
    HipoteseNaGrade("forte", p_valor=0.0001, t=+4.0),
    HipoteseNaGrade("media", p_valor=0.001389, t=+3.2),
    *[HipoteseNaGrade(f"r{i}", p_valor=0.5, t=0.1) for i in range(20)],
)


def _tese(**ajustes) -> Tese:
    base = {
        "identificador": "t1",
        "assunto": "estrategia_alfa",
        "motivo": "amostra insuficiente",
        "condicao_de_despertar": "15 dias independentes",
        "criada_em": _T0,
        "acordar_em": _T0 + timedelta(days=30),
        "amostra_exigida": 15,
        "amostra_atual": 9,
    }
    return Tese(**{**base, **ajustes})


def _percepcao(**ajustes) -> Percepcao:
    base = {"evento": Evento.BARRA_DIARIA, "momento": _T0}
    return Percepcao(**{**base, **ajustes})


class TestOCicloFecha:
    """A diferença entre agente e pipeline."""

    def test_o_cerebro_nao_recomeca_do_zero(self) -> None:
        """🔴 O teste central. Três ciclos, e o diário acumula.

        Um pipeline devolveria o mesmo estado toda vez; um agente carrega o que
        já viveu.
        """
        estado = Estado(missao=Missao.PESQUISA, grade=_GRADE)
        for i in range(3):
            estado, _ = ciclo(_percepcao(momento=_T0 + timedelta(hours=i)), estado)
        assert len(estado.diario) == 3
        assert estado.momento == _T0 + timedelta(hours=2)

    def test_todo_ciclo_produz_registro_inclusive_o_que_nao_faz_nada(self) -> None:
        """Emenda 7. Se ignorar fosse grátis E invisível, o agente ignoraria em
        todo lugar e pareceria seletivo — a mudez do emissor v1."""
        estado, registro = ciclo(_percepcao(), Estado(missao=Missao.PESQUISA, grade=_GRADE))
        assert registro.decisao.acao is Acao.NENHUMA_ACAO
        assert registro.executada is False
        assert len(estado.diario) == 1
        assert "nenhuma_acao" in estado.diario[0]

    def test_o_estado_e_imutavel_e_o_ciclo_devolve_copia(self) -> None:
        """Estado mutável deixaria metade da memória atualizada quando um erro
        acontecesse no meio do ciclo."""
        original = Estado(missao=Missao.PESQUISA, grade=_GRADE)
        novo, _ = ciclo(_percepcao(), original)
        assert original.diario == ()
        assert novo is not original
        assert len(novo.diario) == 1


class TestTesesNoCiclo:
    def test_tese_vencida_desperta_dentro_do_ciclo(self) -> None:
        repo = TesesEmMemoria([_tese()])
        estado = Estado(missao=Missao.PESQUISA, grade=_GRADE, teses=repo)
        _, registro = ciclo(
            _percepcao(momento=_T0 + timedelta(days=31)), estado
        )
        assert registro.decisao.acao is Acao.AGUARDAR_AMOSTRA
        assert repo.todas()[0].estado is EstadoDaTese.DESPERTA

    def test_tese_ja_desperta_em_ciclo_anterior_nao_fica_orfa(self) -> None:
        """A correção de um defeito meu: olhar só a transição AGUARDANDO →
        DESPERTA deixaria para sempre de fora a tese que despertou antes e não
        pôde ser resolvida."""
        repo = TesesEmMemoria([_tese(estado=EstadoDaTese.DESPERTA, amostra_atual=15)])
        estado = Estado(missao=Missao.PESQUISA, grade=_GRADE, teses=repo)
        _, registro = ciclo(_percepcao(momento=_T0 + timedelta(days=1)), estado)
        assert registro.decisao.acao is Acao.AVALIAR_TESE

    def test_tese_com_amostra_suficiente_vira_avaliar(self) -> None:
        repo = TesesEmMemoria([_tese(amostra_atual=15)])
        estado = Estado(missao=Missao.PESQUISA, grade=_GRADE, teses=repo)
        _, registro = ciclo(_percepcao(momento=_T0 + timedelta(days=31)), estado)
        assert registro.decisao.acao is Acao.AVALIAR_TESE


class TestCapacidades:
    def test_a_capacidade_registrada_e_invocada(self) -> None:
        chamadas: list[str] = []
        estado = Estado(missao=Missao.EXECUCAO, grade=_GRADE)
        _, registro = ciclo(
            _percepcao(evento=Evento.DESFECHO_NO_LEDGER, assunto="btc"),
            estado,
            capacidades={Acao.ATUALIZAR_ESTADO: lambda p, e: chamadas.append("ok")},
        )
        assert registro.decisao.acao is Acao.ATUALIZAR_ESTADO
        assert registro.executada is True
        assert chamadas == ["ok"]

    def test_acao_sem_capacidade_e_decidida_e_nao_executada(self) -> None:
        """Um Cérebro incompleto precisa poder rodar e MOSTRAR o que lhe falta,
        em vez de explodir — senão não dá para construí-lo por partes."""
        _, registro = ciclo(
            _percepcao(evento=Evento.DESFECHO_NO_LEDGER, assunto="btc"),
            Estado(missao=Missao.EXECUCAO, grade=_GRADE),
        )
        assert registro.decisao.acao is Acao.ATUALIZAR_ESTADO
        assert registro.executada is False
        assert registro.erro is None

    def test_capacidade_que_levanta_nao_derruba_o_ciclo(self) -> None:
        """Agente que morre na primeira ferramenta com defeito perde também a
        memória do que estava fazendo."""
        def _quebra(p, e):
            raise RuntimeError("corretora fora do ar")

        estado, registro = ciclo(
            _percepcao(evento=Evento.DESFECHO_NO_LEDGER, assunto="btc"),
            Estado(missao=Missao.EXECUCAO, grade=_GRADE),
            capacidades={Acao.ATUALIZAR_ESTADO: _quebra},
        )
        assert registro.executada is False
        assert "corretora fora do ar" in (registro.erro or "")
        assert len(estado.diario) == 1  # o ciclo avançou mesmo assim
        assert "FALHOU" in estado.diario[0]


class TestNaoHaAtalho:
    def test_nao_existe_caminho_que_produza_acao_sem_a_politica(self) -> None:
        """O agente decide *o que fazer*; não decide que a régua era outra.

        Lê a árvore sintática: `agente.py` só pode obter uma `AcaoPermitida`
        chamando `proxima_acao`. Construir uma à mão em outro ponto seria
        contornar o guardrail.
        """
        import ast
        import inspect

        from radar.cerebro import agente

        arvore = ast.parse(inspect.getsource(agente))
        construcoes = [
            no
            for no in ast.walk(arvore)
            if isinstance(no, ast.Call)
            and isinstance(no.func, ast.Name)
            and no.func.id == "AcaoPermitida"
        ]
        assert not construcoes, "agente construiu ação sem passar pela política"

    def test_emissao_exige_fdr_tambem_pelo_ciclo(self) -> None:
        """A trava da política vale ponta a ponta, não só na chamada direta."""
        _, registro = ciclo(
            _percepcao(evento=Evento.BARRA_1S, assunto="btc"),
            Estado(missao=Missao.EXECUCAO, grade=_GRADE),
            consultar_validade=lambda _a: Validade("x", sobrevive_fdr=False),
        )
        assert registro.decisao.acao is Acao.NENHUMA_ACAO


class TestEstado:
    def test_o_orcamento_e_recalculado_e_nao_guardado(self) -> None:
        """Guardar o número o transformaria em contador com outro nome, e ele
        ficaria velho no instante em que a grade mudasse."""
        estado = Estado(missao=Missao.PESQUISA, grade=_GRADE)
        antes = estado.orcamento.restante
        maior = estado.com(grade=(*_GRADE, HipoteseNaGrade("nova", p_valor=1e-9, t=+7.0)))
        assert maior.orcamento.restante > antes

    def test_instantaneo_publica_o_estado_inteiro(self) -> None:
        repo = TesesEmMemoria([_tese()])
        estado = Estado(missao=Missao.PESQUISA, grade=_GRADE, teses=repo, momento=_T0)
        i = estado.instantaneo()
        assert i["missao"] == "pesquisa"
        assert i["grade"] == len(_GRADE)
        assert i["orcamento"]["alvo"] == "forte"  # type: ignore[index]
        assert i["teses"]["aguardando"] == 1  # type: ignore[index]

    def test_lote_em_execucao_e_visivel_no_estado(self) -> None:
        estado = Estado(missao=Missao.PESQUISA, lote_em_execucao=("h1", "h2"))
        assert estado.lote_rodando is True
        assert estado.instantaneo()["lote_em_execucao"] == ["h1", "h2"]

    def test_sem_teses_o_estado_nao_explode(self) -> None:
        estado = Estado(missao=Missao.PESQUISA, grade=_GRADE)
        assert estado.teses_abertas == ()
        assert estado.instantaneo()["teses"] == {}


def test_percepcao_nao_carrega_juizo() -> None:
    """Percepção com score tornaria impossível separar, na autópsia, 'o dado
    estava errado' de 'a leitura do dado estava errada'."""
    campos = set(Percepcao.__dataclass_fields__)
    assert not campos & {"score", "probabilidade", "direcao", "confianca"}
    with pytest.raises(TypeError):
        Percepcao(evento=Evento.BARRA_1S, momento=_T0, score=0.9)  # type: ignore[call-arg]

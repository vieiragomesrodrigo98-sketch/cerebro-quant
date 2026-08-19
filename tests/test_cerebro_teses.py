"""
Testes das teses vivas (`radar.cerebro.teses`) — a emenda 4 do ADR 0037.

A trava central: **uma tese sobrevive ao ciclo**. O teste que prova isso é
`test_a_tese_sobrevive_ao_ciclo_e_reaparece_quando_o_prazo_vence` — se ele
falhar, o `AGUARDAR` voltou a ser status e o dia 2 não sabe o que o dia 1
esperava.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from radar.cerebro.teses import (
    EstadoDaTese,
    Tese,
    TesesEmMemoria,
    despertar_vencidas,
    resumo_de_mudez,
)

_T0 = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)


def _tese(**ajustes) -> Tese:
    base = {
        "identificador": "tese-184",
        "assunto": "estrategia_alfa × bull_lateral",
        "motivo": "amostra insuficiente: 9 dias independentes de 15 exigidos",
        "condicao_de_despertar": "atingir 15 dias independentes no regime bull_lateral",
        "criada_em": _T0,
        "acordar_em": _T0 + timedelta(days=30),
        "amostra_exigida": 15,
        "amostra_atual": 9,
    }
    return Tese(**{**base, **ajustes})


class TestATeseSobreviveAoCiclo:
    """A razão de o módulo existir."""

    def test_a_tese_sobrevive_ao_ciclo_e_reaparece_quando_o_prazo_vence(self) -> None:
        """Dia 1: "não há amostra, AGUARDAR". Dia 2 tem de saber o que era.

        Sem isto o Cérebro recomeça do zero a cada execução, que é a definição
        de pipeline sem memória operacional.
        """
        repo = TesesEmMemoria([_tese()])

        # dia 2, antes do prazo: nada acorda
        assert list(despertar_vencidas(repo, _T0 + timedelta(days=1))) == []

        # depois do prazo: reaparece, com o motivo intacto
        despertas = list(despertar_vencidas(repo, _T0 + timedelta(days=31)))
        assert len(despertas) == 1
        assert despertas[0].identificador == "tese-184"
        assert "9 dias independentes" in despertas[0].motivo
        assert despertas[0].estado is EstadoDaTese.DESPERTA

    def test_despertar_e_persistido_e_nao_so_devolvido(self) -> None:
        """Devolver sem salvar faria a mesma tese despertar em todo ciclo, para
        sempre — e o agente reprocessaria a mesma espera sem fim."""
        repo = TesesEmMemoria([_tese()])
        list(despertar_vencidas(repo, _T0 + timedelta(days=31)))
        assert list(despertar_vencidas(repo, _T0 + timedelta(days=32))) == []
        assert repo.todas()[0].estado is EstadoDaTese.DESPERTA

    def test_o_historico_registra_o_estado_da_amostra_no_despertar(self) -> None:
        repo = TesesEmMemoria([_tese()])
        desperta = next(iter(despertar_vencidas(repo, _T0 + timedelta(days=31))))
        assert "9/15" in desperta.historico[-1]

    def test_alterar_uma_tese_nao_apaga_o_que_ela_dizia_antes(self) -> None:
        """`Tese` é imutável e `com()` devolve cópia: é o "o que ela dizia
        antes" que permite auditar se a espera fazia sentido."""
        original = _tese()
        nova = original.com(nota="amostra subiu", amostra_atual=12)
        assert original.amostra_atual == 9
        assert nova.amostra_atual == 12
        assert nova.historico == ("amostra subiu",)


class TestCamposObrigatorios:
    """Cada obrigatoriedade existe porque a ausência produziu estado inauditável."""

    def test_tese_sem_motivo_e_recusada(self) -> None:
        with pytest.raises(ValueError, match="mudez com nome melhor"):
            _tese(motivo="   ")

    def test_tese_sem_condicao_de_despertar_e_recusada(self) -> None:
        """Condição vazia vira tese imortal: nada nunca a reavalia."""
        with pytest.raises(ValueError, match="espera eterna"):
            _tese(condicao_de_despertar="")

    def test_prazo_antes_da_criacao_e_recusado(self) -> None:
        with pytest.raises(ValueError, match="depois da criação"):
            _tese(acordar_em=_T0 - timedelta(days=1))

    def test_nao_existe_tese_sem_prazo(self) -> None:
        """`acordar_em` não aceita `None` — é o prazo que transforma a espera em
        EVENTO, e sem ele o Cérebro nunca acorda sozinho."""
        with pytest.raises(TypeError):
            Tese(  # type: ignore[call-arg]
                identificador="x",
                assunto="y",
                motivo="m",
                condicao_de_despertar="c",
                criada_em=_T0,
            )

    def test_amostra_exigida_zero_e_recusada(self) -> None:
        with pytest.raises(ValueError, match="contagem positiva"):
            _tese(amostra_exigida=0)


class TestAmostra:
    def test_amostra_insuficiente(self) -> None:
        assert _tese(amostra_atual=9, amostra_exigida=15).amostra_suficiente is False

    def test_amostra_atingida(self) -> None:
        assert _tese(amostra_atual=15, amostra_exigida=15).amostra_suficiente is True

    def test_espera_por_evento_nao_depende_de_amostra(self) -> None:
        """Nem toda espera é por amostra: aguardar um regime de mercado mudar
        não tem contagem, e exigir uma travaria a tese para sempre."""
        t = _tese(amostra_exigida=None, amostra_atual=0)
        assert t.amostra_suficiente is True


class TestResumoDeMudez:
    def test_publica_todos_os_estados_inclusive_zerados(self) -> None:
        """Chave ausente e valor zero precisam ser a mesma coisa para quem lê o
        painel — senão "não houve" e "não medi" voltam a se confundir."""
        r = resumo_de_mudez([_tese()])
        assert set(r) == {e.value for e in EstadoDaTese}
        assert r["aguardando"] == 1
        assert r["resolvida"] == 0

    def test_a_contagem_e_o_que_torna_a_mudez_visivel(self) -> None:
        """Um Cérebro com centenas de esperas e nada resolvido não está sendo
        seletivo — está travado. O emissor v1 ficou mudo um mês inteiro sendo
        lido como criterioso; o número é o que distingue os dois casos.
        """
        travado = [
            _tese(identificador=f"t{i}") for i in range(300)
        ]
        r = resumo_de_mudez(travado)
        assert r["aguardando"] == 300
        assert r["resolvida"] == 0

    def test_estado_expirada_e_distinto_de_resolvida(self) -> None:
        """Tese que morreu de velhice não é tese respondida — colapsar as duas
        destruiria a contagem de mudez."""
        r = resumo_de_mudez(
            [
                _tese(identificador="a", estado=EstadoDaTese.EXPIRADA),
                _tese(identificador="b", estado=EstadoDaTese.RESOLVIDA),
            ]
        )
        assert r["expirada"] == 1
        assert r["resolvida"] == 1


class TestRepositorio:
    def test_abertas_exclui_as_fechadas(self) -> None:
        repo = TesesEmMemoria(
            [
                _tese(identificador="viva"),
                _tese(identificador="morta", estado=EstadoDaTese.RESOLVIDA),
            ]
        )
        assert [t.identificador for t in repo.abertas()] == ["viva"]

    def test_desperta_continua_aberta(self) -> None:
        """Despertar não fecha a tese: ela volta para avaliação, e quem decide o
        que fazer é a política."""
        repo = TesesEmMemoria([_tese(estado=EstadoDaTese.DESPERTA)])
        assert len(repo.abertas()) == 1

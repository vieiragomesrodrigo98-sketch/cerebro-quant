"""
Testes da Fase 2 do Cérebro vivo — detecção, expectativa e decisão (ADR 0036).

Cada estágio responde UMA pergunta, e os testes de decomposição abaixo falham se
algum deles começar a responder a do vizinho.
"""

from __future__ import annotations

import ast
import math
import pathlib

import pandas as pd
import pytest

from radar.cerebro.contexto import ContextoDeMercado, Mercado
from radar.cerebro.decisao import (
    NAO_OPERA,
    Decisao,
    DecisaoRegistrada,
    decidir,
    resumo_de_mudez,
)
from radar.cerebro.deteccao import Candidato, Situacao, detectar_situacoes
from radar.cerebro.expectativa import Expectativa, avaliar
from radar.cerebro.mapa_validade import Direcao
from radar.cerebro.monitor import Observacao
from radar.cerebro.universo import AtivoRef


def _observacao(elegivel: bool = True) -> Observacao:
    return Observacao(
        ativo=AtivoRef.de("PETR4", Mercado.B3),
        contexto=ContextoDeMercado(
            mercado=Mercado.B3, regime="bear" if elegivel else "desconhecido"
        ),
        elegivel_no_ciclo=elegivel,
    )


def _candidato() -> Candidato:
    return Candidato(observacao=_observacao(), situacoes=frozenset({Situacao.BREAKOUT}))


_PRECOS = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})


class TestDeteccao:
    def test_detecta_uniao_das_situacoes(self) -> None:
        d1 = lambda o, p: [Situacao.BREAKOUT]  # noqa: E731
        d2 = lambda o, p: [Situacao.MOMENTUM, Situacao.BREAKOUT]  # noqa: E731
        c = detectar_situacoes(_observacao(), _PRECOS, [d1, d2])
        assert c is not None
        assert c.situacoes == frozenset({Situacao.BREAKOUT, Situacao.MOMENTUM})

    def test_nada_detectado_devolve_none(self) -> None:
        """"Nada acontecendo" é o estado NORMAL da varredura total — criar
        objeto para cada ativo do universo a cada ciclo seria ruído com custo."""
        assert detectar_situacoes(_observacao(), _PRECOS, [lambda o, p: []]) is None

    def test_observacao_inelegivel_nao_vira_candidata(self) -> None:
        """Sem contexto legível não há partição, e sem partição a situação não
        pode ser interpretada (G-P1). O ativo NÃO sai do universo."""
        assert detectar_situacoes(
            _observacao(elegivel=False), _PRECOS, [lambda o, p: [Situacao.BREAKOUT]]
        ) is None

    def test_detector_que_levanta_nao_cala_os_outros(self) -> None:
        def quebrado(o, p):
            raise RuntimeError("bug no detector")

        c = detectar_situacoes(
            _observacao(), _PRECOS, [quebrado, lambda o, p: [Situacao.REVERSAO]]
        )
        assert c is not None and c.situacoes == frozenset({Situacao.REVERSAO})

    def test_candidato_sem_situacao_e_recusado(self) -> None:
        with pytest.raises(ValueError, match="ao menos uma situação"):
            Candidato(observacao=_observacao(), situacoes=frozenset())

    def test_vocabulario_de_situacao_e_descritivo_nao_avaliativo(self) -> None:
        """Guardrail *resultado da estratégia ≠ propriedade do ativo*: rotular a
        situação com o resultado esperado contamina a detecção com a conclusão."""
        proibidos = ("bom", "ruim", "forte", "fraco", "provavel", "otimo", "alta_chance")
        for s in Situacao:
            assert not any(p in s.value for p in proibidos), f"{s.value} tem juízo no nome"


class TestExpectativa:
    def test_ev_desconta_o_custo(self) -> None:
        e = avaliar(p_ganho=0.5, payoff_ganho=0.10, payoff_perda=0.05, custo_roundtrip=0.01)
        assert e.bruto == pytest.approx(0.5 * 0.10 - 0.5 * 0.05)
        assert e.ev == pytest.approx(e.bruto - 0.01)

    def test_probabilidade_alta_com_payoff_contra_perde_dinheiro(self) -> None:
        """O erro mais comum de quem vem de classificação: 70% de acerto e
        payoff assimétrico contra tem EV negativo."""
        e = avaliar(p_ganho=0.70, payoff_ganho=0.01, payoff_perda=0.05, custo_roundtrip=0.0)
        assert e.p_ganho > 0.5
        assert e.ev < 0

    def test_probabilidade_baixa_com_payoff_a_favor_ganha(self) -> None:
        e = avaliar(p_ganho=0.35, payoff_ganho=0.20, payoff_perda=0.03, custo_roundtrip=0.0)
        assert e.p_ganho < 0.5
        assert e.ev > 0

    def test_custo_comeu_o_edge_distingue_de_sem_edge(self) -> None:
        """Distinguir importa: o primeiro caso melhora com execução e ativo mais
        líquido; o segundo não melhora com nada."""
        comeu = avaliar(p_ganho=0.6, payoff_ganho=0.02, payoff_perda=0.01, custo_roundtrip=0.02)
        sem = avaliar(p_ganho=0.3, payoff_ganho=0.01, payoff_perda=0.05, custo_roundtrip=0.0)
        assert comeu.custo_comeu_o_edge is True
        assert sem.custo_comeu_o_edge is False

    def test_custo_zero_devolve_margem_infinita_e_nao_numero_arbitrario(self) -> None:
        e = avaliar(p_ganho=0.6, payoff_ganho=0.1, payoff_perda=0.01, custo_roundtrip=0.0)
        assert math.isinf(e.margem_sobre_custo)

    @pytest.mark.parametrize("p", [-0.1, 1.1])
    def test_probabilidade_fora_de_0_1_e_erro(self, p: float) -> None:
        with pytest.raises(ValueError, match="fora de"):
            avaliar(p_ganho=p, payoff_ganho=0.1, payoff_perda=0.1, custo_roundtrip=0.0)

    def test_payoff_negativo_e_erro(self) -> None:
        """A direção já está na fórmula; payoff negativo inverteria o sinal duas
        vezes — a classe de bug que travou o ADR 0034."""
        with pytest.raises(ValueError, match="MAGNITUDES"):
            avaliar(p_ganho=0.5, payoff_ganho=-0.1, payoff_perda=0.1, custo_roundtrip=0.0)


class TestDecisao:
    def _exp(self, ev_positivo: bool = True) -> Expectativa:
        return (
            Expectativa(p_ganho=0.6, payoff_ganho=0.10, payoff_perda=0.02, custo=0.001)
            if ev_positivo
            else Expectativa(p_ganho=0.3, payoff_ganho=0.01, payoff_perda=0.10, custo=0.001)
        )

    def test_tese_morta_e_invalidado_antes_de_qualquer_conta(self) -> None:
        r = decidir(_candidato(), self._exp(), direcao=Direcao.LONG, momento_favoravel=True, tese_ainda_vale=False)
        assert r.decisao is Decisao.INVALIDADO

    def test_sem_vantagem_liquida_e_sem_sinal(self) -> None:
        r = decidir(_candidato(), self._exp(ev_positivo=False), direcao=Direcao.LONG, momento_favoravel=True)
        assert r.decisao is Decisao.SEM_SINAL

    def test_vantagem_sem_gatilho_e_aguardar_e_nao_sem_sinal(self) -> None:
        """A distinção que o ADR 0036 exige: *tese válida* não é *momento
        adequado*. Colapsar os dois perderia a oportunidade nos ciclos
        seguintes."""
        r = decidir(_candidato(), self._exp(), direcao=Direcao.LONG, momento_favoravel=False)
        assert r.decisao is Decisao.AGUARDAR

    def test_vantagem_com_gatilho_opera(self) -> None:
        r = decidir(_candidato(), self._exp(), direcao=Direcao.LONG, momento_favoravel=True)
        assert r.opera is True
        assert r.decisao in (Decisao.COMPRAR, Decisao.VENDER)


class TestWaitNaoESilencio:
    """G-P2: mudez é FALHA, nunca abstenção. O emissor v1 ficou um mês sem
    emitir e isso foi lido como seletividade."""

    def test_toda_decisao_exige_motivo(self) -> None:
        with pytest.raises(ValueError, match="silêncio com nome de decisão"):
            DecisaoRegistrada(
                ativo_id="b3:X", decisao=Decisao.AGUARDAR, motivo="  ",
                expectativa=None, chave_de_particao="b3:bear",
            )

    def test_aguardar_carrega_a_expectativa_que_o_sustentou(self) -> None:
        """`WAIT` sem o número que o justificou é opção grátis: o Cérebro deixa
        de errar porque deixa de decidir."""
        r = decidir(_candidato(), self._expectativa(), direcao=Direcao.LONG, momento_favoravel=False)
        assert r.expectativa is not None
        assert r.motivo.strip()

    def _expectativa(self) -> Expectativa:
        return Expectativa(p_ganho=0.6, payoff_ganho=0.10, payoff_perda=0.02, custo=0.001)

    def test_resumo_de_mudez_conta_todas_as_saidas(self) -> None:
        decisoes = [
            decidir(_candidato(), self._expectativa(), direcao=Direcao.LONG, momento_favoravel=False),
            decidir(_candidato(), self._expectativa(), direcao=Direcao.LONG, momento_favoravel=True),
        ]
        resumo = resumo_de_mudez(decisoes)
        assert resumo["_total"] == 2
        assert resumo["_opera"] == 1
        assert resumo[Decisao.AGUARDAR.value] == 1

    def test_nao_opera_e_conjunto_explicito(self) -> None:
        assert set(NAO_OPERA) == {Decisao.AGUARDAR, Decisao.SEM_SINAL, Decisao.INVALIDADO}


class TestDecomposicaoDosEstagios:
    """Cada módulo responde UMA pergunta. Se um começar a responder a do
    vizinho, o motor volta a ser monolítico e a autópsia volta a ser
    impossível — foram 4 defeitos encadeados no v1."""

    def _imports(self, arquivo: str) -> set[str]:
        fonte = pathlib.Path(f"src/radar/cerebro/{arquivo}").read_text(encoding="utf-8")
        modulos: set[str] = set()
        for no in ast.walk(ast.parse(fonte)):
            if isinstance(no, ast.Import):
                modulos.update(a.name for a in no.names)
            elif isinstance(no, ast.ImportFrom) and no.module:
                modulos.add(no.module)
        return modulos

    def test_deteccao_nao_conhece_expectativa_nem_decisao(self) -> None:
        m = self._imports("deteccao.py")
        assert not [x for x in m if "expectativa" in x or "decisao" in x or "ranking" in x]

    def test_expectativa_nao_decide(self) -> None:
        m = self._imports("expectativa.py")
        assert not [x for x in m if "decisao" in x or "ranking" in x or "alocacao" in x]

    def test_nenhum_estagio_importa_motor_ou_lab(self) -> None:
        for arquivo in ("deteccao.py", "expectativa.py", "decisao.py"):
            proibidos = [
                x
                for x in self._imports(arquivo)
                if x.startswith(("radar.mie", "radar.engines", "radar.scalp", "radar.lab"))
            ]
            assert not proibidos, f"{arquivo} importou implementação concreta: {proibidos}"

    def test_candidato_nao_carrega_julgamento(self) -> None:
        campos = set(Candidato.__dataclass_fields__)
        assert not (campos & {"score", "probabilidade", "direcao", "ev", "ranking"})


class TestRevisaoDaFase2:
    """Os quatro defeitos achados na revisão da Fase 2, cada um com a trava que
    impede o retorno. Nenhum deles fazia teste falhar antes — foi leitura
    crítica que os achou, e é por isso que viram teste agora."""

    def _exp(self, ganho: float = 0.10, perda: float = 0.02) -> Expectativa:
        return Expectativa(p_ganho=0.6, payoff_ganho=ganho, payoff_perda=perda, custo=0.001)

    # --- 1) A DIREÇÃO SAÍA DA ARITMÉTICA (bug grave) ------------------------
    def test_long_compra_e_short_vende(self) -> None:
        for direcao, esperado in ((Direcao.LONG, Decisao.COMPRAR), (Direcao.SHORT, Decisao.VENDER)):
            r = decidir(_candidato(), self._exp(), direcao=direcao, momento_favoravel=True)
            assert r.decisao is esperado

    def test_short_com_payoff_ganho_maior_ainda_vende(self) -> None:
        """A regressão exata. A 1ª versão fazia
        `COMPRAR if payoff_ganho >= payoff_perda else VENDER` — então um SHORT
        com payoff_ganho=0,10 e payoff_perda=0,02 saía como COMPRAR.

        Assimetria de payoff não tem relação nenhuma com direção; direção é
        propriedade da HIPÓTESE. É a classe de bug que o ADR 0034 bloqueou, e a
        1ª versão a reintroduziu com um comentário afirmando que a evitava."""
        r = decidir(_candidato(), self._exp(ganho=0.10, perda=0.02), direcao=Direcao.SHORT, momento_favoravel=True)
        assert r.decisao is Decisao.VENDER

    def test_long_com_payoff_perda_maior_ainda_compra(self) -> None:
        """Espelho do anterior. Precisa de `p_ganho` alto para o EV ficar
        POSITIVO com perda maior que ganho — senão o `SEM_SINAL` vem antes e o
        teste não chega a exercer a direção. (A 1ª versão deste teste usava
        p=0,6 e falhou por isso: o teste estava errado, não o código.)"""
        exp = Expectativa(p_ganho=0.90, payoff_ganho=0.05, payoff_perda=0.10, custo=0.001)
        assert exp.ev > 0
        r = decidir(_candidato(), exp, direcao=Direcao.LONG, momento_favoravel=True)
        assert r.decisao is Decisao.COMPRAR

    @pytest.mark.parametrize("direcao", [Direcao.NEUTRO, Direcao.DESCONHECIDA])
    def test_direcao_indefinida_nao_vira_ordem(self, direcao: Direcao) -> None:
        """Emitir com direção indefinida é o bloqueio metodológico do ADR 0034 —
        custou uma trava de projeto inteira até a semântica ser declarada."""
        r = decidir(_candidato(), self._exp(), direcao=direcao, momento_favoravel=True)
        assert r.decisao is Decisao.SEM_SINAL
        assert "direção" in r.motivo

    def test_direcao_aparece_no_motivo(self) -> None:
        r = decidir(_candidato(), self._exp(), direcao=Direcao.SHORT, momento_favoravel=True)
        assert "short" in r.motivo

    # --- 2) STRING MÁGICA no monitor ----------------------------------------
    def test_monitor_usa_a_constante_e_nao_string_literal(self) -> None:
        """`"desconhecido"` literal duplicava a constante de `contexto.py`: se
        ela mudasse, o monitor quebraria em silêncio."""
        fonte = pathlib.Path("src/radar/cerebro/monitor.py").read_text(encoding="utf-8")
        assert '"desconhecido"' not in fonte
        assert "DESCONHECIDO" in fonte

    # --- 3) PARÂMETRO DESCARTADO em expectativa -----------------------------
    def test_avaliar_nao_recebe_candidato(self) -> None:
        """A 1ª versão recebia `Candidato` e o descartava com `_ = candidato`.
        Parâmetro não usado sugere acoplamento que não existe."""
        import inspect

        assert "candidato" not in inspect.signature(avaliar).parameters

    def test_expectativa_nao_importa_deteccao(self) -> None:
        fonte = pathlib.Path("src/radar/cerebro/expectativa.py").read_text(encoding="utf-8")
        assert "from radar.cerebro.deteccao" not in fonte

    # --- 4) FALHA DE DETECTOR engolida em silêncio --------------------------
    def test_falha_de_detector_e_contada_e_nao_engolida(self) -> None:
        """Contradizia o princípio que eu mesmo apliquei ao monitor: lá o
        inelegível APARECE. Um detector com bug que sempre levanta produziria
        zero situações para sempre, indistinguível de "mercado calmo"."""

        def quebrado(o, p):
            raise RuntimeError("bug")

        c = detectar_situacoes(
            _observacao(), _PRECOS, [quebrado, quebrado, lambda o, p: [Situacao.MOMENTUM]]
        )
        assert c is not None
        assert c.evidencias["falhas_de_detector"] == 2

    def test_sem_falha_nao_polui_as_evidencias(self) -> None:
        c = detectar_situacoes(_observacao(), _PRECOS, [lambda o, p: [Situacao.MOMENTUM]])
        assert c is not None
        assert "falhas_de_detector" not in c.evidencias

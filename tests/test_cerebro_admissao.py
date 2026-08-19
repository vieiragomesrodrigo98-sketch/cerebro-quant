"""
O portão de admissão — Fase 2 do plano de 13/08.

O que ele existe para impedir: gastar célula do orçamento de FDR medindo uma
hipótese que a economia já reprova. Medido: o edge do `scalp_cripto_v2` foi 5%
do pedágio, e no scalp de cripto o custo é 296% do movimento mediano de 60 s.

⚠️ Os dois testes que mais importam aqui são os de **não-degeneração**: um portão
que aprova tudo e um portão que reprova tudo são igualmente inúteis, e ambos
passam despercebidos se o teste só exercita o caminho feliz.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from radar.cerebro.admissao import (
    RAZAO_MAXIMA_CUSTO_MOVIMENTO,
    avaliar_familia,
    avaliar_horizonte,
    celulas_admitidas,
    pool_minimo_para,
)
from radar.cerebro.operabilidade import (
    Operabilidade,
    classificar_operabilidade,
)


def _op(ticker: str, custo: float, movimentos: dict[int, float], operaveis: tuple[int, ...]):
    return Operabilidade(
        ticker=ticker,
        custo_roundtrip=custo,
        movimento_por_horizonte=movimentos,
        horizontes_operaveis=operaveis,
    )


def _universo(n: int, *, custo: float, movimento: float, horizonte: int = 5):
    """`n` ativos idênticos — isola o eixo sob teste."""
    return {
        f"T{i}": _op(f"T{i}", custo, {horizonte: movimento}, (horizonte,))
        for i in range(n)
    }


class TestPoolMinimo:
    @pytest.mark.parametrize(
        ("fracao", "esperado"),
        [(0.10, 10), (0.05, 20), (0.20, 5), (0.25, 4), (1.0, 1)],
    )
    def test_derivado_da_fracao_do_gate(self, fracao, esperado):
        """Não é número escolhido: é `ceil(1/fração)` — o menor pool em que o
        topo ainda seleciona em vez de carimbar."""
        assert pool_minimo_para(fracao) == esperado

    @pytest.mark.parametrize("ruim", [0.0, -0.1, 1.5])
    def test_fracao_invalida_levanta(self, ruim):
        with pytest.raises(ValueError, match="fracao_gate"):
            pool_minimo_para(ruim)


class TestAsDuasPernas:
    """`admissivel` exige pool suficiente E economia que paga — falha-fechado."""

    def test_pool_grande_e_economia_boa_admite(self):
        v = avaliar_horizonte(_universo(20, custo=0.001, movimento=0.01), 5, fracao_gate=0.10)
        assert v.admissivel
        assert v.motivo == ""

    def test_economia_boa_mas_pool_pequeno_reprova(self):
        """4 operáveis com gate de 10%: o "topo de 10%" seria 1 de 4 = 25%."""
        v = avaliar_horizonte(_universo(4, custo=0.001, movimento=0.01), 5, fracao_gate=0.10)
        assert not v.admissivel
        assert v.pool_suficiente is False
        assert v.economia_paga is True
        assert "carimbaria" in v.motivo

    def test_pool_grande_mas_economia_ruim_reprova(self):
        """O caso do scalp: custo 296% do movimento."""
        v = avaliar_horizonte(_universo(50, custo=0.0296, movimento=0.01), 5, fracao_gate=0.10)
        assert not v.admissivel
        assert v.pool_suficiente is True
        assert v.economia_paga is False
        assert "296%" in v.motivo

    def test_universo_sem_nenhum_operavel_reprova_pelas_duas(self):
        """Ninguém opera E o universo típico não paga — as duas pernas caem.

        ⚠️ Este teste exigia `inf` até 13/08, e a expectativa estava presa à
        semântica ERRADA: a razão era medida só sobre os operáveis, então sem
        operável não havia o que medir. Com a medida sobre o universo inteiro,
        *"ninguém opera"* deixa de significar *"ninguém foi medido"* — e aqui a
        razão é 0,01/0,001 = 10, dez vezes o limiar. É informação melhor que
        `inf`: diz o quanto falta.
        """
        vazio = {f"T{i}": _op(f"T{i}", 0.01, {5: 0.001}, ()) for i in range(50)}
        v = avaliar_horizonte(vazio, 5, fracao_gate=0.10)
        assert not v.admissivel
        assert not v.pool_suficiente
        assert not v.economia_paga
        assert v.razao_custo_movimento == pytest.approx(10.0)

    def test_sem_medida_nenhuma_a_razao_e_infinita(self):
        """O caso genuinamente imensurável: custo desconhecido em todo mundo.

        Ausência de medida nunca vira benefício da dúvida — é a mesma doutrina
        do custo em `operabilidade`.
        """
        sem_medida = {
            f"T{i}": _op(f"T{i}", float("nan"), {5: 0.001}, ()) for i in range(50)
        }
        v = avaliar_horizonte(sem_medida, 5, fracao_gate=0.10)
        assert math.isinf(v.razao_custo_movimento)
        assert not v.admissivel
        assert "sem movimento medido" in v.motivo


class TestNaoDegenera:
    """Um portão que aprova tudo e um que reprova tudo são igualmente inúteis.

    ⚠️ Estes testes foram REESCRITOS em 13/08, depois de uma auditoria
    adversarial mostrar que a versão anterior os alimentava com `Operabilidade`
    **que o produtor jamais emitiria** — por exemplo `custo=0,0296` e
    `movimento=0,01` com `horizontes_operaveis=(5,)`, quando
    `classificar_operabilidade` exige `movimento > 2 × custo` para marcar
    operável.

    Um teste que fabrica entrada impossível não protege nada: ele passa
    qualquer que seja a implementação, e foi o que escondeu a perna econômica
    morta. Agora o dado entra pelo PRODUTOR REAL.
    """

    @staticmethod
    def _pelo_produtor(n: int, *, custo: float, movimento: float, horizonte: int = 5):
        """Constrói via `classificar_operabilidade` — nada de objeto à mão.

        `ret_fwd_<h>` alternando ±movimento dá |retorno| mediano = movimento,
        que é o que `movimento_tipico` mede.
        """
        linhas = []
        for i in range(n):
            for k in range(10):
                linhas.append(
                    {
                        "ticker": f"T{i}",
                        f"ret_fwd_{horizonte}": movimento if k % 2 else -movimento,
                    }
                )
        quadro = pd.DataFrame(linhas)
        custos = {f"T{i}": custo for i in range(n)}
        return classificar_operabilidade(quadro, custos, horizontes=(horizonte,))

    def test_nao_aprova_tudo(self):
        """Custo 100× o movimento: ninguém opera, e a economia reprova."""
        ruim = avaliar_horizonte(
            self._pelo_produtor(50, custo=0.10, movimento=0.001), 5, fracao_gate=0.10
        )
        assert not ruim.admissivel
        assert not ruim.economia_paga, (
            "a perna econômica precisa DISPARAR aqui — se só o pool reprova, o "
            "portão tem uma perna decorativa"
        )

    def test_nao_reprova_tudo(self):
        bom = avaliar_horizonte(
            self._pelo_produtor(50, custo=0.0005, movimento=0.05), 5, fracao_gate=0.10
        )
        assert bom.admissivel

    def test_a_perna_economica_reprova_com_pool_cheio(self):
        """O caso que a versão anterior era incapaz de produzir.

        Universo em que TODO ativo é operável (movimento 3× o custo, acima da
        margem de 2×) — pool cheio — e ainda assim a razão mediana do universo
        fica acima do limiar? Não: com todos operáveis a razão é 1/3.

        Este teste existe para fixar a fronteira REAL do portão: com o produtor
        honesto, `pool_suficiente` e `economia_paga` andam juntos, e quem
        discrimina de fato é a mistura — universo em que poucos operam.
        """
        classif = self._pelo_produtor(50, custo=0.01, movimento=0.03)
        v = avaliar_horizonte(classif, 5, fracao_gate=0.10)
        assert v.pool_suficiente
        assert v.economia_paga
        assert v.razao_custo_movimento == pytest.approx(1 / 3, rel=0.01)

    def test_universo_misto_e_onde_a_economia_decide(self):
        """5 ativos ótimos e 95 péssimos: pool passa, universo típico não paga."""
        bons = self._pelo_produtor(5, custo=0.0001, movimento=0.05)
        ruins = self._pelo_produtor(95, custo=0.10, movimento=0.001)
        misto = {f"B{k}": v for k, v in bons.items()} | {f"R{k}": v for k, v in ruins.items()}
        v = avaliar_horizonte(misto, 5, fracao_gate=0.05)
        assert v.n_operaveis == 5
        assert not v.economia_paga, (
            "o ativo mediano deste universo não paga o pedágio — a economia tem "
            "de reprovar mesmo com 5 ativos ótimos no topo"
        )

    def test_a_fronteira_esta_onde_foi_declarada(self):
        """Exatamente em `custo == movimento` ainda passa; acima, não."""
        na_linha = avaliar_horizonte(_universo(50, custo=0.01, movimento=0.01), 5, fracao_gate=0.10)
        acima = avaliar_horizonte(_universo(50, custo=0.0101, movimento=0.01), 5, fracao_gate=0.10)
        assert na_linha.razao_custo_movimento == pytest.approx(RAZAO_MAXIMA_CUSTO_MOVIMENTO)
        assert na_linha.economia_paga
        assert not acima.economia_paga


class TestFamiliaInteira:
    def test_horizonte_reprovado_nao_reprova_a_familia(self):
        """ADR 0038: ausência de oportunidade ≠ ausência do modelo. Uma família
        com 3 horizontes em que 1 reprova pré-registra 2 células, não zero."""
        classif = {}
        for i in range(30):
            classif[f"T{i}"] = _op(
                f"T{i}",
                0.002,
                {5: 0.0005, 21: 0.02, 250: 0.20},   # h=5 não paga; 21 e 250 pagam
                (21, 250),
            )
        vereditos = avaliar_familia(classif, (5, 21, 250), fracao_gate=0.10)
        assert not vereditos[5].admissivel
        assert vereditos[21].admissivel
        assert vereditos[250].admissivel
        assert celulas_admitidas(vereditos) == 2

    def test_o_custo_em_fdr_e_o_numero_de_admitidos(self):
        """É este número que entra na conta do orçamento — não o de horizontes
        declarados."""
        classif = _universo(30, custo=0.10, movimento=0.001, horizonte=5)
        vereditos = avaliar_familia(classif, (5,), fracao_gate=0.10)
        assert celulas_admitidas(vereditos) == 0

    def test_fracao_do_universo_e_reportada(self):
        classif = {}
        for i in range(100):
            oper = (5,) if i < 40 else ()
            classif[f"T{i}"] = _op(f"T{i}", 0.001, {5: 0.01}, oper)
        v = avaliar_horizonte(classif, 5, fracao_gate=0.10)
        assert v.n_universo == 100
        assert v.n_operaveis == 40
        assert v.fracao_do_universo == pytest.approx(0.40)

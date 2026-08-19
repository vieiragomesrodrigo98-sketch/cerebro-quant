"""
Testes do orçamento estatístico (`radar.cerebro.orcamento`).

A trava central é a **emenda 2 do DEV**: orçamento é derivado do estado
estatístico, nunca um contador. O teste que prova isso é
`test_hipotese_forte_AUMENTA_o_espaco` — se ele falhar, alguém trocou a
derivação por `restante -= 1` e o módulo perdeu a razão de existir.
"""

from __future__ import annotations

import pytest

from radar.cerebro.orcamento import (
    HipoteseNaGrade,
    Orcamento,
    OrcamentoEsgotadoError,
    escolher_alvo,
    espaco_restante,
    exigir_espaco,
)

# A grade REAL do projeto em 2026-08-12, copiada de `data/mapa_validade.json`
# em PRECISÃO CHEIA. Os quatro primeiros são cripto fortemente negativo; o
# quinto é o único positivo.
#
# ⚠️ A precisão não é preciosismo, e custou uma falha para ficar claro: a
# primeira versão deste fixture usou o `p` arredondado que eu tinha impresso na
# tela (`0.001389` em vez de `0.0013885837810332419`). Isso move `5·q/p` de
# 180,04 para 179,98, o teto de 180 para 179, e reprovou um código correto. O
# limiar do BH é uma divisão pelo `p` — arredondar o denominador de uma régua é
# mudar a régua.
_GRADE_REAL = [
    HipoteseNaGrade("scalp_cripto_v2_h60b_a", p_valor=1.4985524752826146e-15, t=-7.9771),
    HipoteseNaGrade("scalp_cripto_v2_h60b_b", p_valor=7.127585677450586e-07, t=-4.9579),
    HipoteseNaGrade("day_cripto_v1_h24_B", p_valor=1.3298695055071355e-05, t=-4.3551),
    HipoteseNaGrade("day_cripto_v1_h72_B", p_valor=0.00020390654169693563, t=-3.7141),
    HipoteseNaGrade("estrategia_alfa", p_valor=0.0013885837810332419, t=+3.1970),
    HipoteseNaGrade("day_cripto_v1_h24_A", p_valor=0.00466445294912235, t=-2.8293),
    HipoteseNaGrade("day_cripto_v1_h24_Bb", p_valor=0.005021798568904535, t=-2.8056),
    HipoteseNaGrade("mie_cripto_1h_h72", p_valor=0.005854530335404694, t=-2.7558),
    HipoteseNaGrade("scalp_reversao_esticada", p_valor=0.006025768413256151, t=-2.7464),
    HipoteseNaGrade("swing_b3_h10", p_valor=0.02062890618140063, t=+2.3147),
] + [
    # os postos 11 a 57 da grade real, sem efeito relevante
    HipoteseNaGrade(f"resto_{i}", p_valor=0.30 + i * 0.01, t=0.5)
    for i in range(47)
]


class TestOrcamentoEDerivado:
    """A emenda 2 — a razão de este módulo não ser um inteiro numa variável."""

    def test_o_numero_do_projeto_e_180_57_123(self) -> None:
        """Derivado, não arbitrado: `estrategia_alfa` no posto 5 com p=0,001389
        sobrevive enquanto `p₅ ≤ (5/m)·0,05`, ou seja `m ≤ 180`."""
        o = espaco_restante(_GRADE_REAL)
        assert o.alvo == "estrategia_alfa"
        assert (o.teto, o.gasto, o.restante) == (180, 57, 123)

    def test_hipotese_forte_aumenta_o_espaco(self) -> None:
        """🔴 O teste que separa derivação de contador.

        Um contador só sabe descer. A escada do BH sobe quando entra achado
        forte, e o orçamento tem de subir junto — senão o sistema pune quem
        descobre coisas.
        """
        antes = espaco_restante(_GRADE_REAL).restante
        com_achado = espaco_restante(
            [*_GRADE_REAL, HipoteseNaGrade("achado_novo", p_valor=1e-9, t=+6.0)]
        ).restante
        assert com_achado > antes

    def test_hipotese_nula_diminui_o_espaco(self) -> None:
        antes = espaco_restante(_GRADE_REAL).restante
        com_nula = espaco_restante(
            [*_GRADE_REAL, HipoteseNaGrade("tentativa_vazia", p_valor=0.94, t=0.1)]
        ).restante
        assert com_nula == antes - 1

    def test_nao_existe_operacao_de_decremento_no_modulo(self) -> None:
        """Guarda contra a regressão exata que a emenda 2 proíbe.

        Lê a ÁRVORE SINTÁTICA, não o texto. A primeira versão procurava a
        string `-= 1` e casou com a prosa do próprio docstring que **proíbe** a
        operação — o teste que existe para impedir "regra que persegue a
        grafia" era, ele mesmo, uma. `ast.AugAssign` com `ast.Sub` é o que a
        emenda proíbe, e é isso que se procura.
        """
        import ast
        import inspect

        from radar.cerebro import orcamento

        arvore = ast.parse(inspect.getsource(orcamento))
        decrementos = [
            no for no in ast.walk(arvore)
            if isinstance(no, ast.AugAssign) and isinstance(no.op, ast.Sub)
        ]
        assert not decrementos, (
            "orçamento voltou a ser contador: "
            f"{[ast.unparse(no) for no in decrementos]}"
        )


class TestAlvo:
    def test_o_alvo_e_a_melhor_positiva_sobrevivente(self) -> None:
        """Não se protege poder estatístico para continuar podendo afirmar que
        algo NÃO funciona — refutação já é conhecimento fechado."""
        alvo = escolher_alvo(_GRADE_REAL)
        assert alvo is not None
        assert alvo.nome == "estrategia_alfa"
        assert alvo.t == pytest.approx(3.1970, abs=1e-4)

    def test_refutacao_forte_nunca_vira_alvo(self) -> None:
        so_negativas = [h for h in _GRADE_REAL if not h.positiva]
        assert escolher_alvo(so_negativas) is None

    def test_sem_positiva_o_orcamento_e_zero_e_nao_infinito(self) -> None:
        """Ausência de conhecimento não é licença para gastar. Devolver
        'ilimitado' seria exatamente o movimento que o ADR 0031 impede."""
        o = espaco_restante([h for h in _GRADE_REAL if not h.positiva])
        assert o.alvo is None
        assert o.teto == 0
        assert o.esgotado

    def test_as_refutacoes_de_cripto_sustentam_o_achado_de_b3(self) -> None:
        """O resultado mais contraintuitivo do projeto, e a razão de o cálculo
        rodar sobre a grade INTEIRA.

        Os postos 1–4 são cripto fortemente negativo. São eles que erguem a
        escada até o posto 5. Removê-los promove `estrategia_alfa` a posto 1, onde
        o limiar vira `q/m` — e ela morre.
        """
        sem_as_refutacoes = [h for h in _GRADE_REAL if (h.t or 0) > -3.0]
        assert escolher_alvo(_GRADE_REAL) is not None
        assert escolher_alvo(sem_as_refutacoes) is None

    def test_pode_proteger_um_alvo_declarado_em_vez_do_automatico(self) -> None:
        o = espaco_restante(_GRADE_REAL, alvo="swing_b3_h10")
        assert o.alvo == "swing_b3_h10"
        assert o.teto < 180  # alvo mais fraco, menos espaço


class TestNaoEstimavel:
    def test_nao_estimavel_conta_no_denominador(self) -> None:
        """ADR 0032: motor novo não zera o contador. Hipótese que não pôde ser
        medida consumiu tentativa do mesmo jeito."""
        o = espaco_restante([*_GRADE_REAL, HipoteseNaGrade("sem_amostra", p_valor=None)])
        assert o.gasto == 58

    def test_nao_estimavel_nunca_e_alvo(self) -> None:
        """*Não-estimável não é refutado* — mas também não é evidência
        favorável. Sem `t` medido, não há o que proteger."""
        assert HipoteseNaGrade("x", p_valor=None, t=None).positiva is False


class TestExigirEspaco:
    def test_recusa_e_visivel_e_nao_um_false_engolivel(self) -> None:
        """A emenda 2 exige recusa visível. Um `if` ignorado em silêncio
        reproduziria a defesa que existe, não dispara, e é lida como saúde."""
        with pytest.raises(OrcamentoEsgotadoError, match="Corte escopo ANTES de medir"):
            exigir_espaco(_GRADE_REAL, quantas=124)

    def test_autoriza_dentro_do_espaco(self) -> None:
        o = exigir_espaco(_GRADE_REAL, quantas=123)
        assert isinstance(o, Orcamento)
        assert o.restante == 123

    def test_a_mensagem_nomeia_quem_morre(self) -> None:
        """Erro que diz 'orçamento esgotado' e não diz o que se perde não deixa
        ninguém decidir se vale a pena."""
        with pytest.raises(OrcamentoEsgotadoError, match="estrategia_alfa"):
            exigir_espaco(_GRADE_REAL, quantas=999)


class TestGradeVazia:
    def test_grade_vazia_nao_explode(self) -> None:
        o = espaco_restante([])
        assert (o.teto, o.gasto, o.restante) == (0, 0, 0)

    def test_p_zero_por_subfluxo_nao_vira_orcamento_infinito(self) -> None:
        """`p` exatamente 0 é artefato de ponto flutuante num `t` enorme, não
        medição. O orçamento não pode depender disso."""
        grade = [
            HipoteseNaGrade("subfluxo", p_valor=0.0, t=+40.0),
            HipoteseNaGrade("real", p_valor=0.001, t=+3.3),
        ]
        assert espaco_restante(grade).teto < 10_000

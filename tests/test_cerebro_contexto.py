"""
Testes do estágio MARKET CONTEXT (`radar.cerebro.contexto`) — ADR 0036.

Cobrem o que o módulo promete e, principalmente, as travas que ele existe para
segurar: vocabulário comum entre mercados (G-P4), contexto como partição e não
feature (G-P1), e ausência de leitura que não vira afirmação sobre o mercado.
"""

from __future__ import annotations

import pandas as pd
import pytest

from radar.cerebro.contexto import (
    DESCONHECIDO,
    REGIMES,
    ContextoDeMercado,
    Mercado,
    detectar,
)


def _detector_vazio(precos: pd.DataFrame) -> pd.DataFrame:
    """Detector que nao acha nada — o caso comum na varredura total."""
    return pd.DataFrame(columns=["regime", "confianca"])


def _detector_que_levanta(precos: pd.DataFrame) -> pd.DataFrame:
    """Detector que quebra com entrada estruturalmente errada — e o Cerebro
    tem de absorver, porque um ativo ruim nao pode derrubar a varredura."""
    raise ValueError("colunas obrigatorias ausentes")


def _detector_fixo(regime: str, confianca: float | None = None):
    def _d(precos: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame([{"regime": regime, "confianca": confianca}])
    return _d


class TestVocabularioComum:
    """G-P4 do Contrato de Pensamento: os dois mercados falam o mesmo
    vocabulário de contexto. Sem isso, uma zona de validade medida em B3 não é
    comparável com uma de cripto, e o Router do ADR 0032 fica sem sobre o que
    rotear."""

    def test_regime_fora_do_vocabulario_e_recusado(self) -> None:
        with pytest.raises(ValueError, match="fora do vocabulário comum"):
            ContextoDeMercado(mercado=Mercado.B3, regime="lateral_com_vies")

    @pytest.mark.parametrize("regime", REGIMES)
    def test_os_quatro_regimes_sao_aceitos_nos_dois_mercados(self, regime: str) -> None:
        for mercado in (Mercado.B3, Mercado.CRIPTO):
            assert ContextoDeMercado(mercado=mercado, regime=regime).regime == regime

    def test_b3_e_cripto_compartilham_o_mesmo_conjunto(self) -> None:
        """A trava contra a regressão mais provável: alguém acrescenta um regime
        só para cripto e o vocabulário deixa de ser comum sem ninguém notar."""
        from radar.historical.regime_cripto import REGIMES_MIE_CRIPTO
        from radar.mie.regime import REGIMES_MIE

        assert set(REGIMES) == set(REGIMES_MIE) == set(REGIMES_MIE_CRIPTO)


class TestParticaoNaoFeature:
    """G-P1: contexto SEPARA populações, não entra no vetor de entrada. O
    Cérebro v1 morreu disso — 7 de 45 colunas constantes dentro do dia, e a de
    maior ganho era uma delas: o modelo aprendeu o DIA, não o ativo."""

    def test_chave_de_particao_identifica_mercado_e_regime(self) -> None:
        c = ContextoDeMercado(mercado=Mercado.CRIPTO, regime="bear")
        assert c.chave_de_particao() == "cripto:bear"

    def test_mercados_diferentes_no_mesmo_regime_particionam_separado(self) -> None:
        """Duas populações distintas não podem colidir na mesma chave: `bear` de
        B3 e `bear` de cripto são amostras diferentes, e agregá-las produziria
        exatamente o pooling que o ADR 0036 alerta."""
        b3 = ContextoDeMercado(mercado=Mercado.B3, regime="bear").chave_de_particao()
        cripto = ContextoDeMercado(mercado=Mercado.CRIPTO, regime="bear").chave_de_particao()
        assert b3 != cripto

    def test_contexto_e_imutavel(self) -> None:
        """Observação que o consumidor altera deixa de servir como chave: duas
        hipóteses poderiam declarar a mesma partição falando de coisas
        diferentes."""
        c = ContextoDeMercado(mercado=Mercado.B3, regime="bull_lateral")
        with pytest.raises((AttributeError, TypeError)):
            c.regime = "panico"  # type: ignore[misc]


class TestAusenciaDeLeituraNaoEAfirmacao:
    """`DESCONHECIDO` não é um 5º regime. Cair no regime mais comum por default
    transformaria falta de dado em afirmação sobre o mercado — o mesmo modo de
    falha da sentinela que reportou verde sobre um antibot desligado."""

    def test_desconhecido_e_aceito_mas_marcado_como_ilegivel(self) -> None:
        c = ContextoDeMercado(mercado=Mercado.B3, regime=DESCONHECIDO)
        assert c.legivel is False

    def test_regime_valido_e_legivel(self) -> None:
        assert ContextoDeMercado(mercado=Mercado.B3, regime="panico").legivel is True

    def test_desconhecido_nao_esta_no_vocabulario(self) -> None:
        assert DESCONHECIDO not in REGIMES

    def test_quadro_vazio_vira_desconhecido_e_nao_excecao(self) -> None:
        """O Cérebro vivo varre o universo INTEIRO: ativo sem histórico é
        esperado, não excepcional. Levantar aqui mataria a varredura — foi o que
        o `Decimal("NaN")` fez com o ciclo overnight de 08/08."""
        vazio = pd.DataFrame(columns=["ticker", "date", "Close", "regime"])
        c = detectar(vazio, Mercado.B3, _detector_vazio)
        assert c.regime == DESCONHECIDO
        assert c.legivel is False

    def test_detector_que_levanta_nao_derruba_a_varredura(self) -> None:
        """Entrada estruturalmente errada faz o detector real levantar; o
        contexto tem de absorver e devolver ausência de leitura."""
        lixo = pd.DataFrame({"coluna_que_nao_existe": [1, 2, 3]})
        for mercado in (Mercado.B3, Mercado.CRIPTO):
            assert detectar(lixo, mercado, _detector_que_levanta).regime == DESCONHECIDO

    def test_confianca_ausente_e_none_nao_zero(self) -> None:
        """`0.0` afirmaria 'confiança medida em zero'; `None` diz 'não medida'."""
        c = ContextoDeMercado(mercado=Mercado.B3, regime="bear")
        assert c.confianca is None


class TestRoteamento:
    def test_aceita_mercado_como_string(self) -> None:
        """Quem varre o universo tem o mercado como dado, normalmente string —
        exigir o enum empurraria a conversão para cada chamador, que é
        exatamente o `if mercado == ...` espalhado que este módulo remove."""
        vazio = pd.DataFrame()
        assert detectar(vazio, "cripto", _detector_vazio).mercado is Mercado.CRIPTO
        assert detectar(vazio, "b3", _detector_vazio).mercado is Mercado.B3

    def test_mercado_desconhecido_e_erro_e_nao_silencio(self) -> None:
        """Mercado inválido é defeito de programação, não dado ruim: aqui
        levantar é o certo, ao contrário do regime ilegível."""
        with pytest.raises(ValueError):
            detectar(pd.DataFrame(), "forex", _detector_vazio)


class TestInjecaoDoDetector:
    """A inversão de dependência que o ADR 0032 exige: o Cérebro define o
    contrato, o motor é injetado na borda.

    A 1ª versão deste módulo importava `radar.mie.regime` direto e o teste de
    camadas reprovou — corretamente. A saída fácil seria acrescentar a violação
    à allowlist; não foi feita, porque allowlist é para dívida HERDADA, e dívida
    nova acomodada nela é o mecanismo virando decoração.
    """

    def test_regime_do_detector_chega_ao_contexto(self) -> None:
        c = detectar(pd.DataFrame(), Mercado.CRIPTO, _detector_fixo("bull_trending", 0.82))
        assert c.regime == "bull_trending"
        assert c.confianca == pytest.approx(0.82)
        assert c.legivel is True
        assert c.chave_de_particao() == "cripto:bull_trending"

    def test_regime_fora_do_vocabulario_vindo_do_detector_vira_desconhecido(self) -> None:
        """Detector que devolve rótulo desconhecido não pode contaminar o
        vocabulário comum nem derrubar a varredura: vira ausência de leitura."""
        c = detectar(pd.DataFrame(), Mercado.B3, _detector_fixo("lateral_com_vies"))
        assert c.regime == DESCONHECIDO
        assert c.legivel is False

    def test_cerebro_nao_importa_motor(self) -> None:
        """A trava explícita, no arquivo do próprio módulo: se alguém voltar a
        importar `radar.mie` aqui, este teste falha junto com o de camadas — e
        falhar em dois lugares torna a regressão difícil de ignorar."""
        import ast
        import pathlib

        fonte = pathlib.Path("src/radar/cerebro/contexto.py").read_text(encoding="utf-8")
        modulos = set()
        for no in ast.walk(ast.parse(fonte)):
            if isinstance(no, ast.Import):
                modulos.update(a.name for a in no.names)
            elif isinstance(no, ast.ImportFrom) and no.module:
                modulos.add(no.module)
        proibidos = [m for m in modulos if m.startswith(("radar.mie", "radar.engines", "radar.scalp"))]
        assert not proibidos, f"contexto.py voltou a importar motor: {proibidos}"

class TestLeOQueOsDetectoresReaisPublicam:
    """A guarda que faltava, e a razão de o defeito ter sobrevivido meses.

    Até 13/08, `_ultima_linha_regime` procurava a coluna `"regime"` — nome que
    **detector nenhum deste projeto produz**. Os dois publicam `regime_mie`
    (`radar.mie.regime.COLUNAS_SAIDA`, `radar.historical.regime_cripto.
    COLUNAS_SAIDA`). O efeito: `detectar()` devolvia `DESCONHECIDO` para todo
    detector real, sempre.

    Passou porque **todos** os testes acima injetam um quadro com `"regime"` —
    uma fixture que a produção não gera. Testar a fixture em vez do contrato é
    o que transforma cobertura em falsa tranquilidade, e o módulo tinha zero
    importadores fora do pacote para desmentir.
    """

    def test_o_nome_procurado_existe_nos_detectores_de_verdade(self) -> None:
        """Acopla ao CONTRATO dos detectores, não a uma fixture: se qualquer um
        renomear a coluna, este teste cai — que é o aviso que não existia."""
        from radar.cerebro.contexto import NOMES_DA_COLUNA_DE_REGIME
        from radar.historical.regime_cripto import COLUNAS_SAIDA as SAIDA_CRIPTO
        from radar.mie.regime import COLUNAS_SAIDA as SAIDA_B3

        for nome_do_detector, saida in (("b3", SAIDA_B3), ("cripto", SAIDA_CRIPTO)):
            assert set(NOMES_DA_COLUNA_DE_REGIME) & set(saida), (
                f"o detector de {nome_do_detector} publica {saida} e "
                f"`contexto` procura {NOMES_DA_COLUNA_DE_REGIME} — sem "
                "interseção, `detectar()` devolve DESCONHECIDO para sempre"
            )

    def test_le_regime_mie_que_e_o_que_a_producao_gera(self) -> None:
        quadro = pd.DataFrame([
            {"date": "2026-07-28", "regime_mie": "bear"},
            {"date": "2026-07-29", "regime_mie": "bull_lateral"},
        ])
        ctx = detectar(quadro, Mercado.CRIPTO, lambda _: quadro)
        assert ctx.regime == "bull_lateral", "não leu `regime_mie`"
        assert ctx.legivel

    def test_continua_lendo_regime_para_nao_quebrar_quem_ja_usava(self) -> None:
        quadro = pd.DataFrame([{"regime": "bear"}])
        assert detectar(quadro, Mercado.B3, lambda _: quadro).regime == "bear"

    def test_coluna_ausente_vira_desconhecido_e_nao_excecao(self) -> None:
        quadro = pd.DataFrame([{"date": "2026-07-29", "adx_14": 17.3}])
        ctx = detectar(quadro, Mercado.CRIPTO, lambda _: quadro)
        assert ctx.regime == DESCONHECIDO and not ctx.legivel

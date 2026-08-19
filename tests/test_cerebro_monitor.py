"""
Testes do contrato de universo e do MARKET MONITOR — ADR 0036, Fase 1.

A trava central aqui não é funcional, é **arquitetural**: o monitor observa e
para. Se algum dia ele pontuar, decidir ou emitir, os testes de decomposição
abaixo falham — e falham por design, não por acidente.
"""

from __future__ import annotations

import ast
import pathlib

import pandas as pd
import pytest

from radar.cerebro.contexto import Mercado
from radar.cerebro.monitor import (
    INTERVALO_PADRAO_S,
    Observacao,
    ciclos,
    elegiveis,
    varrer,
)
from radar.cerebro.universo import AtivoRef, TipoDeInstrumento, unir


class _ProvedorFake:
    """Provedor de teste — a prova de que o Cérebro só conhece o CONTRATO."""

    def __init__(self, mercado: Mercado, simbolos: list[str]) -> None:
        self._mercado = mercado
        self._simbolos = simbolos

    @property
    def mercado(self) -> Mercado:
        return self._mercado

    def ativos(self):
        return [AtivoRef.de(s, self._mercado) for s in self._simbolos]


def _detector(regime: str):
    def _d(precos: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame([{"regime": regime, "confianca": 0.7}])

    return _d


def _precos_ok(_ativo: AtivoRef) -> pd.DataFrame:
    return pd.DataFrame({"Close": [1.0, 2.0, 3.0]})


def _precos_que_quebram(_ativo: AtivoRef) -> pd.DataFrame:
    raise OSError("parquet corrompido")


_DETECTORES = {Mercado.B3: _detector("bear"), Mercado.CRIPTO: _detector("bull_trending")}


class TestContratoDeUniverso:
    def test_ativo_id_deriva_e_nao_colide_entre_mercados(self) -> None:
        """Mesmo símbolo em mercados diferentes precisa de identidade distinta —
        colisão de identidade já produziu, neste projeto, duas hipóteses
        homônimas exibindo os números da primeira."""
        a = AtivoRef.de("XPTO", Mercado.B3)
        b = AtivoRef.de("XPTO", Mercado.CRIPTO)
        assert a.ativo_id != b.ativo_id
        assert a.ativo_id == "b3:XPTO"

    def test_ativo_ref_e_imutavel(self) -> None:
        a = AtivoRef.de("PETR4", Mercado.B3)
        with pytest.raises((AttributeError, TypeError)):
            a.simbolo = "VALE3"  # type: ignore[misc]

    def test_simbolo_vazio_e_recusado(self) -> None:
        with pytest.raises(ValueError, match="não vazios"):
            AtivoRef(ativo_id="b3:", simbolo="", mercado=Mercado.B3)

    def test_unir_e_ordenado_e_sem_duplicata(self) -> None:
        """Varredura em ordem não-determinística produz relatório que muda sem o
        mercado ter mudado."""
        p1 = _ProvedorFake(Mercado.B3, ["VALE3", "PETR4"])
        p2 = _ProvedorFake(Mercado.CRIPTO, ["BTCUSDT"])
        p3 = _ProvedorFake(Mercado.B3, ["PETR4"])  # duplicata proposital
        ids = [a.ativo_id for a in unir(p1, p2, p3)]
        assert ids == sorted(ids)
        assert len(ids) == len(set(ids)) == 3

    def test_tipo_de_instrumento_diz_o_que_o_ativo_e(self) -> None:
        """A lente do corretor: instrumento decide spread, horário e liquidação.
        Não é juízo de qualidade."""
        a = AtivoRef.de("BTCUSDT", Mercado.CRIPTO, TipoDeInstrumento.PAR_SPOT)
        assert a.tipo is TipoDeInstrumento.PAR_SPOT


class TestMonitorObserva:
    def test_varre_o_universo_inteiro(self) -> None:
        obs = varrer(
            [_ProvedorFake(Mercado.B3, ["A", "B"]), _ProvedorFake(Mercado.CRIPTO, ["C"])],
            _precos_ok,
            _DETECTORES,
        )
        assert [o.ativo.ativo_id for o in obs] == ["b3:A", "b3:B", "cripto:C"]
        assert all(o.elegivel_no_ciclo for o in obs)

    def test_contexto_vem_do_detector_do_mercado_certo(self) -> None:
        obs = varrer(
            [_ProvedorFake(Mercado.B3, ["A"]), _ProvedorFake(Mercado.CRIPTO, ["C"])],
            _precos_ok,
            _DETECTORES,
        )
        por_id = {o.ativo.ativo_id: o for o in obs}
        assert por_id["b3:A"].contexto.regime == "bear"
        assert por_id["cripto:C"].contexto.regime == "bull_trending"

    def test_fonte_que_quebra_nao_derruba_o_ciclo(self) -> None:
        """Um ativo ruim não pode matar a varredura — foi o que o
        `Decimal("NaN")` fez com o ciclo overnight de 08/08."""
        obs = varrer([_ProvedorFake(Mercado.B3, ["A", "B"])], _precos_que_quebram, _DETECTORES)
        assert len(obs) == 2
        assert not any(o.elegivel_no_ciclo for o in obs)

    def test_inelegivel_aparece_no_resultado(self) -> None:
        """Omitir o que falhou faria "não observei" e "observei e estava limpo"
        ficarem indistinguíveis — o defeito que o canário do scan existe para
        impedir. O ativo continua no universo (ADR 0036)."""
        obs = varrer([_ProvedorFake(Mercado.B3, ["A"])], _precos_que_quebram, _DETECTORES)
        assert len(obs) == 1
        assert obs[0].ativo.ativo_id == "b3:A"
        assert elegiveis(obs) == []

    def test_mercado_sem_detector_vira_inelegivel_e_nao_excecao(self) -> None:
        obs = varrer([_ProvedorFake(Mercado.CRIPTO, ["C"])], _precos_ok, {Mercado.B3: _detector("bear")})
        assert obs[0].elegivel_no_ciclo is False

    def test_ciclos_respeita_maximo_e_nao_dorme_de_verdade(self) -> None:
        dormidas: list[float] = []
        saida = list(
            ciclos(
                [_ProvedorFake(Mercado.B3, ["A"])],
                _precos_ok,
                _DETECTORES,
                intervalo_s=1.5,
                maximo=3,
                dormir=dormidas.append,
            )
        )
        assert len(saida) == 3
        # Dorme ENTRE ciclos, não depois do último.
        assert dormidas == [1.5, 1.5]

    def test_cadencia_padrao_e_explicita(self) -> None:
        """`intervalo_s` mexe no denominador do FDR (ADR 0031, penalização por
        tentativas): tem de ser constante nomeada, não número solto."""
        assert INTERVALO_PADRAO_S == 300.0


class TestMonitorNaoDecide:
    """A trava arquitetural do DEV: `UNIVERSO → … → CANDIDATOS`, nunca
    `→ SCORE → SINAL`. O motor anterior era monolítico e, quando adoeceu, exigiu
    medir 4 defeitos encadeados um a um para descobrir onde estava quebrado."""

    def test_observacao_nao_carrega_score_probabilidade_nem_direcao(self) -> None:
        campos = set(Observacao.__dataclass_fields__)
        proibidos = {"score", "probabilidade", "direcao", "sinal", "confianca_sinal", "ranking"}
        assert not (campos & proibidos), (
            f"Observacao ganhou campo de julgamento: {campos & proibidos}. "
            "O monitor observa; quem julga é deteccao/expectativa/decisao."
        )

    def test_modulo_nao_importa_estagios_posteriores(self) -> None:
        """Se o monitor importar ranking, expectativa ou decisão, a decomposição
        acabou — e a próxima autópsia volta a ser impossível."""
        fonte = pathlib.Path("src/radar/cerebro/monitor.py").read_text(encoding="utf-8")
        modulos = set()
        for no in ast.walk(ast.parse(fonte)):
            if isinstance(no, ast.Import):
                modulos.update(a.name for a in no.names)
            elif isinstance(no, ast.ImportFrom) and no.module:
                modulos.add(no.module)
        posteriores = {
            m
            for m in modulos
            if any(
                p in m
                for p in ("ranking", "expectativa", "decisao", "deteccao", "alvo", "pool")
            )
        }
        assert not posteriores, f"monitor importou estágio posterior: {posteriores}"

    def test_cerebro_nao_importa_motor_nem_lab(self) -> None:
        """A mesma inversão que a catraca de camadas cobrou em `contexto.py`:
        o universo entra por contrato, não por import de `radar.lab`."""
        for arquivo in ("monitor.py", "universo.py"):
            fonte = pathlib.Path(f"src/radar/cerebro/{arquivo}").read_text(encoding="utf-8")
            modulos = set()
            for no in ast.walk(ast.parse(fonte)):
                if isinstance(no, ast.Import):
                    modulos.update(a.name for a in no.names)
                elif isinstance(no, ast.ImportFrom) and no.module:
                    modulos.add(no.module)
            proibidos = [
                m
                for m in modulos
                if m.startswith(("radar.mie", "radar.engines", "radar.scalp", "radar.lab"))
            ]
            assert not proibidos, f"{arquivo} importou implementação concreta: {proibidos}"

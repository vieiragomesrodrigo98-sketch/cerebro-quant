"""O Cérebro lê o mapa de validade (`CEREBRO_LACO_AUTONOMO01`, P0).

O elo que faltava, e por que faltava
--------------------------------------
O card descreve a cadeia que não fecha: `execução → outcome → mapa de validade →
re-treino → novo ranking → volta`. O diagnóstico mediu o sintoma — *"nenhum motor,
emissor ou scanner lê `mapa_validade.json`"*.

A causa não era desatenção: **não havia chave de junção**. O mapa fala
`familia:celula#ponta`; o sinal falava `trilha` (três valores herdados) e uma `origem`
enterrada dentro de `justificativa_estruturada` — o engine de day trade refutado nem
isso tinha. Perguntar "a hipótese por trás deste sinal foi validada?" era impossível,
então nenhum leitor teria o que fazer. `Signal.hipotese` é a chave; `radar.cerebro.pool`
é o leitor; o Guard 0-ter de `save_signal` é quem o obedece.

O que se guarda aqui
---------------------
Que a pergunta acionável é a NEGATIVA (hoje o mapa tem zero `VALIDADO`, e exigir
validação calaria o produto), que ausência de hipótese não vira permissão nem bloqueio,
e a assimetria ausente × corrompido — que é a mesma de
`COST_TRACKER_LEDGER_CORROMPIDO01`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from radar.cerebro.pool import (
    MapaDeValidadeIlegivelError,
    estado_da_hipotese,
    foi_refutada,
    pool,
)


def _mapa(tmp_path: Path, hipoteses: list[tuple[str, str]]) -> Path:
    alvo = tmp_path / "mapa_validade.json"
    alvo.write_text(
        json.dumps({"hipoteses": [{"hipotese": h, "estado": e} for h, e in hipoteses]}),
        encoding="utf-8",
    )
    return alvo


class TestLeituraDoMapa:
    def test_refutada_e_reconhecida(self, tmp_path: Path) -> None:
        m = _mapa(tmp_path, [("scalp_cripto_v2:h60b#topo", "refutado")])

        assert foi_refutada("scalp_cripto_v2:h60b#topo", m) is True
        assert estado_da_hipotese("scalp_cripto_v2:h60b#topo", m) == "refutado"

    def test_nao_comprovada_nao_e_refutada(self, tmp_path: Path) -> None:
        """O par negativo. "Ainda não provei" e "provei que não" são estados
        diferentes, e confundi-los enterraria 45 hipóteses vivas."""
        m = _mapa(tmp_path, [("swing_v1:x#topo", "nao_comprovado")])

        assert foi_refutada("swing_v1:x#topo", m) is False

    def test_hipotese_desconhecida_e_none_e_nao_aprovacao(self, tmp_path: Path) -> None:
        m = _mapa(tmp_path, [("swing_v1:x#topo", "validado")])

        assert estado_da_hipotese("nunca_medida", m) is None
        assert foi_refutada("nunca_medida", m) is False

    def test_pool_traz_so_o_validado(self, tmp_path: Path) -> None:
        m = _mapa(
            tmp_path,
            [("a:1#topo", "validado"), ("b:2#topo", "nao_comprovado"),
             ("c:3#topo", "refutado"), ("d:4#topo", "validado")],
        )

        assert pool(m) == ("a:1#topo", "d:4#topo")

    def test_pool_vazio_e_o_retrato_correto_de_hoje(self, tmp_path: Path) -> None:
        """Zero validadas não é falha do leitor — é o estado do projeto."""
        m = _mapa(tmp_path, [("a:1#topo", "nao_comprovado")])

        assert pool(m) == ()


class TestHipoteseAusente:
    """Sinal legado não tem hipótese, e isso não é permissão nem bloqueio."""

    @pytest.mark.parametrize("vazio", [None, ""])
    def test_sem_hipotese_nao_bloqueia(self, tmp_path: Path, vazio: str | None) -> None:
        m = _mapa(tmp_path, [("x:1#topo", "refutado")])

        assert foi_refutada(vazio, m) is False


class TestAusenteVersusCorrompido:
    """A assimetria deliberada — mesma lição de COST_TRACKER_LEDGER_CORROMPIDO01."""

    def test_mapa_ausente_degrada_e_nao_trava(self, tmp_path: Path) -> None:
        """Checkout sem o artefato gerado não pode virar 'emissão proibida'."""
        inexistente = tmp_path / "nao_existe.json"

        assert pool(inexistente) == ()
        assert estado_da_hipotese("qualquer", inexistente) is None
        assert foi_refutada("qualquer", inexistente) is False

    def test_mapa_corrompido_levanta(self, tmp_path: Path) -> None:
        """Arquivo presente e ilegível não vira 'nenhuma hipótese refutada'.

        Sem isto, um mapa danificado faria a emissão rodar ACREDITANDO que
        consultou o instrumento — pior que não consultar.
        """
        alvo = tmp_path / "mapa_validade.json"
        alvo.write_text('{"hipoteses": [nao e json', encoding="utf-8")

        with pytest.raises(MapaDeValidadeIlegivelError):
            foi_refutada("x:1#topo", alvo)
        with pytest.raises(MapaDeValidadeIlegivelError):
            pool(alvo)


class TestOMapaREALDoProjeto:
    """Contra o artefato de verdade — o leitor tem de entender o formato que existe."""

    def test_le_o_mapa_versionado_sem_explodir(self) -> None:
        real = Path(__file__).resolve().parents[2] / "data" / "mapa_validade.json"
        if not real.is_file():
            pytest.skip("mapa não gerado neste checkout")

        validadas = pool(real)
        dados = json.loads(real.read_text(encoding="utf-8"))
        refutadas = [
            h["hipotese"] for h in dados["hipoteses"] if h.get("estado") == "refutado"
        ]

        assert len(validadas) == dados["resumo"]["validado"]
        assert all(foi_refutada(h, real) for h in refutadas), (
            "o leitor discorda do próprio arquivo — formato mudou sem o leitor saber"
        )
        assert refutadas, (
            "se nada está refutado, este teste não exerce o caminho que importa"
        )

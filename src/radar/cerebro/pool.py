"""O Strategy Pool — o único lugar que LÊ `data/mapa_validade.json`.

O elo que faltava, e por que ele faltava
-----------------------------------------
`CEREBRO_LACO_AUTONOMO01` (P0) descreve a cadeia que não fecha:
`execução → outcome → mapa de validade → re-treino → novo ranking → volta`. O
diagnóstico da S170 mediu o sintoma: *"nenhum motor, emissor ou scanner lê
`mapa_validade.json`"* — 8 ocorrências no repo, nenhuma em `src/`, `api/` ou
scanner.

A causa não era desatenção. **Não havia chave de junção.** O mapa fala
`familia:celula#ponta`; o sinal falava `trilha` (três valores herdados) e uma
`origem` enterrada dentro de `justificativa_estruturada` — o engine de day trade
refutado nem isso tinha, nascia com origem nula. Perguntar *"a hipótese por trás
deste sinal foi validada?"* era impossível, então nenhum leitor teria o que fazer.

`Signal.hipotese` é a chave; este módulo é o leitor.

O que este módulo NÃO faz
--------------------------
Não promove nada. A decisão do DEV de 04/08 é explícita: o Cérebro é autônomo no
ciclo de APRENDIZADO e a escada `shadow → paper → champion/challenger → produção`
continua sendo decisão humana. Aqui só se responde *em que estado o instrumento
colocou esta hipótese*; o que fazer com a resposta é de quem chama.

E não inverte o default. Hoje o mapa tem **zero** hipóteses validadas — um
consumidor que exigisse `VALIDADO` para emitir calaria o produto inteiro. Por isso
a pergunta útil hoje é a negativa (`foi refutada?`), que é acionável e não depende
de nada ter sido validado ainda.

Ausência × corrupção, e a assimetria é deliberada
--------------------------------------------------
Mapa **ausente** devolve pool vazio e `None` por hipótese: o projeto pode estar num
checkout sem o artefato gerado, e travar a emissão por isso seria transformar
"ainda não medi" em "está proibido". Mapa **corrompido levanta**: arquivo presente e
ilegível é estado danificado, e responder `None` faria a emissão rodar achando que
consultou o instrumento. É a mesma lição de `COST_TRACKER_LEDGER_CORROMPIDO01` e a
mesma escolha de `radar.universe.cripto_saneamento.excluidos_do_artefato`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

MAPA_PADRAO: Final[Path] = (
    Path(__file__).resolve().parents[3] / "data" / "mapa_validade.json"
)
"""Onde `scripts/build_mapa_validade.py` grava o Evidence Map."""

ESTADO_REFUTADO: Final[str] = "refutado"
ESTADO_VALIDADO: Final[str] = "validado"


class MapaDeValidadeIlegivelError(RuntimeError):
    """O mapa existe e não pôde ser lido. Nunca degradado para "não sei"."""


def _carregar(caminho: Path | None) -> dict[str, str]:
    alvo = caminho or MAPA_PADRAO
    if not alvo.is_file():
        return {}
    try:
        dados = json.loads(alvo.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MapaDeValidadeIlegivelError(
            f"{alvo} existe e não pôde ser lido ({exc}). Um mapa corrompido não vira "
            "'nenhuma hipótese refutada' — regere com "
            "`python scripts/build_mapa_validade.py`."
        ) from exc
    return {
        str(h["hipotese"]): str(h.get("estado", ""))
        for h in dados.get("hipoteses", [])
        if h.get("hipotese")
    }


def estado_da_hipotese(hipotese: str, caminho: Path | None = None) -> str | None:
    """O estado que o instrumento atribuiu, ou `None` se ele não a conhece.

    `None` significa **não medida**, jamais "aprovada": hipótese fora do mapa é
    hipótese que nunca passou pelo Evidence Map.
    """
    return _carregar(caminho).get(hipotese)


def foi_refutada(hipotese: str | None, caminho: Path | None = None) -> bool:
    """A pergunta acionável hoje.

    `None` (sinal legado, sem hipótese declarada) devolve `False` — não é
    permissão, é a constatação de que não há o que consultar. Quem chama deve
    CONTAR esses casos: a fração não-atribuível mede quanto da emissão ainda vive
    fora do alcance do instrumento.
    """
    if not hipotese:
        return False
    return estado_da_hipotese(hipotese, caminho) == ESTADO_REFUTADO


def pool(caminho: Path | None = None) -> tuple[str, ...]:
    """As hipóteses `VALIDADO` — as únicas que podem receber capital.

    Hoje devolve vazio, e isso é o retrato correto do projeto: nenhuma hipótese
    passou nos seis critérios. Um pool vazio não é falha do leitor.
    """
    return tuple(
        sorted(h for h, e in _carregar(caminho).items() if e == ESTADO_VALIDADO)
    )

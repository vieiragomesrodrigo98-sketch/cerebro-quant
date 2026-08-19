"""
src/radar/cerebro/pesquisa.py — o Evidence Research Loop, com trava de p-hacking.

O problema que este módulo resolve
-----------------------------------
O Evidence Map classifica hipóteses, mas nada as reavalia. Dado novo chega todo
dia (`persist_binance_bars` roda de minuto em minuto), é gravado em disco, e
**ninguém volta para medir com ele** — uma `CANDIDATA` bloqueada por falta de
amostra fica onde está indefinidamente, mesmo depois de o dado que ela precisa
já ter chegado.

Mas a correção ingênua — um cron que remede as 30 toda semana — é pior que o
problema. Nas palavras do DEV:

    Testa → não passou → testa de novo → não passou → testa de novo → até passar.
    Porque isso seria p-hacking automatizado.

A distinção que torna o loop honesto: `data_cutoff`
-----------------------------------------------------
Uma rodada só é **evidência nova** se enxergou dado que a anterior não enxergou.
Re-medir com o mesmo `data_cutoff` não é uma segunda observação — é a **mesma**
observação contada duas vezes, e tratá-la como nova é exatamente o mecanismo que
transforma ruído em descoberta.

Por isso toda rodada é registrada com o corte de dado que a produziu, e
`avaliar_necessidade` **recusa** re-medição sem avanço material de corte.

A bifurcação que muda o custo estatístico — e é o coração do módulo
--------------------------------------------------------------------
"Falta alguma coisa" não é uma categoria só. As duas metades têm custos de
multiplicidade **opostos**:

* **FALTA AMOSTRA** (`Dimensao.NAO` em evidência, com dias insuficientes) — a
  hipótese foi testada e não alcançou a régua. Re-testá-la com mais dado é um
  **teste novo da mesma hipótese**, e cada um conta no denominador do FDR.
  ⇒ **ESPERAR** o corte avançar. Custo: multiplicidade.

* **FALTA MÉTRICA** (`Dimensao.NAO_MEDIDA`) — capacidade, estabilidade e
  drawdown que ninguém calculou. Isso **não é um teste**: é a conclusão de uma
  medição inacabada sobre o dado que já existe. Medir agora não gasta
  multiplicidade nenhuma, porque não há hipótese sendo testada de novo — há uma
  pergunta que nunca foi feita.
  ⇒ **EXECUTAR JÁ**. Custo: zero.

Confundir as duas trava o projeto nos dois sentidos: faz esperar por dado o que
podia ser medido hoje, e faz remedir sem parar o que precisa de tempo.

O que este módulo NÃO faz
--------------------------
Ele não mede nada e não decide nada sobre o mercado. Ele responde *"vale a pena
gastar uma rodada nesta hipótese agora, e por quê?"* — e a resposta é auditável
porque a decisão e o motivo ficam no registro append-only junto com o corte de
dado que a justificou.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final

from radar.cerebro.mapa_validade import Dimensao, Estado, Veredito

MINIMO_DIAS_NOVOS: Final[int] = 60
"""
Quantos dias independentes NOVOS o corte precisa ter ganhado para justificar
re-medir uma hipótese travada por amostra.

60 é ~um trimestre de B3 e ~dois meses de cripto. O número não é mágico e o
critério importante é o outro: **ele existe**. Sem um piso, "esperar dado" vira
"remedir toda semana com 5 dias a mais", que é o mesmo teste repetido com ruído
diferente — a definição operacional de p-hacking.

Este piso NÃO se aplica a `FALTA_METRICA`: lá não há teste sendo repetido.
"""

REGISTRO_PADRAO: Final[str] = "data/cerebro_rodadas_pesquisa.jsonl"


class Necessidade(str, Enum):
    """O que esta hipótese precisa, e portanto o que fazer com ela agora."""

    FALTA_METRICA = "falta_metrica"
    FALTA_AMOSTRA = "falta_amostra"
    NENHUMA = "nenhuma"
    ENCERRADA = "encerrada"
    """`REFUTADA` — não se gasta rodada nela sem informação NOVA declarada."""

    @property
    def acao(self) -> str:
        return _ACAO[self]

    @property
    def custa_multiplicidade(self) -> bool:
        """
        Só `FALTA_AMOSTRA` gasta FDR. Medir uma métrica que nunca foi calculada
        não testa hipótese nenhuma de novo — completa uma medição inacabada.
        """
        return self is Necessidade.FALTA_AMOSTRA


_ACAO: Final[dict[Necessidade, str]] = {
    Necessidade.FALTA_METRICA: "EXECUTAR JÁ — a medição não depende de dado novo",
    Necessidade.FALTA_AMOSTRA: "ESPERAR o corte avançar; re-teste conta no FDR",
    Necessidade.NENHUMA: "nada a fazer — a hipótese passou em tudo que se mede",
    Necessidade.ENCERRADA: "não reabrir sem informação NOVA declarada",
}


@dataclass(frozen=True)
class Rodada:
    """
    Uma medição registrada. `data_cutoff` é o campo que distingue evidência nova
    de re-contagem do mesmo histórico — sem ele o registro não serve para nada.
    """

    research_run_id: str
    hipotese: str
    data_cutoff: str
    """Data máxima do dado que entrou. ISO `YYYY-MM-DD`."""
    dias_independentes: int
    rodada_n: int
    estado: str
    t_nw: float | None = None
    necessidade: str | None = None
    motivo: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Decisao:
    hipotese: str
    necessidade: Necessidade
    acao: str
    motivo: str
    executar_agora: bool
    metricas_faltando: tuple[str, ...] = ()
    dias_novos_desde_ultima: int | None = None
    rodadas_gastas: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hipotese": self.hipotese,
            "necessidade": self.necessidade.value,
            "acao": self.acao,
            "motivo": self.motivo,
            "executar_agora": self.executar_agora,
            "metricas_faltando": list(self.metricas_faltando),
            "dias_novos_desde_ultima": self.dias_novos_desde_ultima,
            "rodadas_gastas": self.rodadas_gastas,
            "custa_multiplicidade": self.necessidade.custa_multiplicidade,
        }


def avaliar_necessidade(
    v: Veredito,
    *,
    dias_independentes: int,
    ultima: Rodada | None = None,
    minimo_dias_novos: int = MINIMO_DIAS_NOVOS,
) -> Decisao:
    """
    Decide o que fazer com UMA hipótese agora.

    A ordem é fixa e não é arbitrária: **métrica antes de amostra**. Uma
    hipótese que tem métricas não medidas E amostra curta deve medir as
    métricas primeiro — é de graça em multiplicidade, e o resultado pode
    reprovar por outro caminho, poupando a espera inteira. Inverter a ordem
    gastaria meses esperando dado para depois descobrir que a capacidade era
    R$ 300.
    """
    if v.estado is Estado.REFUTADO:
        return Decisao(
            v.hipotese, Necessidade.ENCERRADA, Necessidade.ENCERRADA.acao,
            "refutada com poder estatístico — reabrir exige informação NOVA",
            executar_agora=False,
            rodadas_gastas=(ultima.rodada_n if ultima else 0),
        )

    if v.estado is Estado.NAO_ESTIMAVEL:
        # Sem dimensões computadas, os testes abaixo veriam "nada faltando" e
        # devolveriam NENHUMA — "está tudo certo" sobre a hipótese que MENOS foi
        # medida. É a terceira vez nesta sessão que `nao_estimavel` cai num
        # default de sucesso por ausência de dados; o padrão é claro o bastante
        # para virar teste.
        return Decisao(
            v.hipotese, Necessidade.FALTA_AMOSTRA, Necessidade.FALTA_AMOSTRA.acao,
            f"não estimável: {v.justificativa}. Precisa de DADO, não de nova "
            "análise — e por isso conta no FDR quando for re-medida",
            executar_agora=False,
            dias_novos_desde_ultima=(
                None if ultima is None else dias_independentes - ultima.dias_independentes
            ),
            rodadas_gastas=(ultima.rodada_n if ultima else 0),
        )

    nao_medidas = tuple(
        criterio for criterio, dim in v.dimensoes.items() if dim is Dimensao.NAO_MEDIDA
    )
    if nao_medidas:
        return Decisao(
            v.hipotese, Necessidade.FALTA_METRICA, Necessidade.FALTA_METRICA.acao,
            f"{len(nao_medidas)} métrica(s) nunca calculada(s) sobre o dado que já "
            "existe — medir não gasta multiplicidade porque não repete teste nenhum",
            executar_agora=True,
            metricas_faltando=nao_medidas,
            rodadas_gastas=(ultima.rodada_n if ultima else 0),
        )

    if not v.criterios_faltantes:
        return Decisao(
            v.hipotese, Necessidade.NENHUMA, Necessidade.NENHUMA.acao,
            "todos os critérios medidos e satisfeitos", executar_agora=False,
            rodadas_gastas=(ultima.rodada_n if ultima else 0),
        )

    if ultima is None:
        return Decisao(
            v.hipotese, Necessidade.FALTA_AMOSTRA, Necessidade.FALTA_AMOSTRA.acao,
            "primeira rodada registrada para esta hipótese", executar_agora=True,
            rodadas_gastas=0,
        )

    novos = dias_independentes - ultima.dias_independentes
    if novos < minimo_dias_novos:
        return Decisao(
            v.hipotese, Necessidade.FALTA_AMOSTRA, Necessidade.FALTA_AMOSTRA.acao,
            f"só {novos} dia(s) independente(s) novo(s) desde o corte "
            f"{ultima.data_cutoff} (mínimo {minimo_dias_novos}) — re-medir agora "
            "seria contar o MESMO histórico outra vez",
            executar_agora=False, dias_novos_desde_ultima=novos,
            rodadas_gastas=ultima.rodada_n,
        )

    return Decisao(
        v.hipotese, Necessidade.FALTA_AMOSTRA, Necessidade.FALTA_AMOSTRA.acao,
        f"{novos} dias independentes novos desde {ultima.data_cutoff} — evidência "
        f"genuinamente nova. Esta será a rodada {ultima.rodada_n + 1} e CONTA no "
        "denominador do FDR",
        executar_agora=True, dias_novos_desde_ultima=novos,
        rodadas_gastas=ultima.rodada_n,
    )


# ── registro append-only ──────────────────────────────────────────────────────
@dataclass
class RegistroDeRodadas:
    """
    Histórico append-only. Nunca reescreve linha: uma rodada registrada é um
    fato, e apagá-la esconderia uma tentativa do denominador do FDR.
    """

    caminho: Path
    _rodadas: list[Rodada] = field(default_factory=list)

    @classmethod
    def carregar(cls, caminho: str | Path = REGISTRO_PADRAO) -> RegistroDeRodadas:
        p = Path(caminho)
        reg = cls(caminho=p)
        if p.exists():
            for linha in p.read_text(encoding="utf-8").splitlines():
                if linha.strip():
                    reg._rodadas.append(Rodada(**json.loads(linha)))
        return reg

    @property
    def rodadas(self) -> tuple[Rodada, ...]:
        return tuple(self._rodadas)

    def ultima_de(self, hipotese: str) -> Rodada | None:
        anteriores = [r for r in self._rodadas if r.hipotese == hipotese]
        return max(anteriores, key=lambda r: r.rodada_n) if anteriores else None

    def tentativas_de(self, hipotese: str) -> int:
        """Quantas rodadas esta hipótese já gastou. É o que o FDR precisa saber."""
        return sum(1 for r in self._rodadas if r.hipotese == hipotese)

    def registrar(self, r: Rodada) -> None:
        anterior = self.ultima_de(r.hipotese)
        if anterior is not None and r.data_cutoff == anterior.data_cutoff:
            raise ValueError(
                f"rodada de '{r.hipotese}' com o MESMO data_cutoff "
                f"({r.data_cutoff}) da rodada {anterior.rodada_n}: isso não é "
                "evidência nova, é a mesma observação contada duas vezes"
            )
        self._rodadas.append(r)
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        with self.caminho.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    def denominador_fdr(self) -> int:
        """
        Total de testes já gastos — TODAS as rodadas de TODAS as hipóteses.

        Não é o número de hipóteses: uma hipótese re-medida 3× gastou 3 testes,
        e o FDR cobra por tentativa, não por ideia. Usar a contagem de hipóteses
        aqui tornaria o corte de Benjamini-Hochberg artificialmente generoso
        exatamente na direção que favorece o pesquisador.
        """
        return len(self._rodadas)

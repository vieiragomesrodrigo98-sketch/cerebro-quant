"""
src/radar/cerebro/historico.py — a memória de leituras do Cérebro.

O buraco que este módulo fecha
-------------------------------
Até aqui, cada leitura do projeto era um evento isolado: media, publicava um
veredito, e o próximo experimento começava do zero. Ninguém perguntava **"em
relação à tentativa anterior, isto melhorou ou piorou?"** — e sem essa
pergunta não existe auto-calibração, só uma sequência de opiniões
independentes sobre o mesmo mercado.

Consequência concreta e já paga: a Fase 1 mediu Rank IC OOS de −0,0211 na
célula cobaia do mercado piloto e concluiu NO-GO. A família `swing_v1` mediu −0,0268
na mesma célula, com outro alvo e outro instrumento — e a leitura foi escrita
sem confrontar os dois números. Piorou. Ninguém teria notado sem comparar.

O que este módulo NÃO faz
--------------------------
Não julga. `comparar_com_historico` devolve o DELTA e a direção; decidir o que
fazer com "piorou" é do humano (ADR 0032: autonomia no aprendizado, escada na
promoção). E não inventa comparabilidade: leituras em eixos diferentes ficam
explicitamente marcadas como não-comparáveis, em vez de virarem um número
enganoso.

O eixo comparável
------------------
**Rank IC OOS** (média diária + `t` de Newey-West), por
`(mercado, horizonte)`. É a única métrica que sobrevive às mudanças de
desenho que o projeto já fez — alvo binário por limiar, alvo por decil,
gate por probabilidade, gate por Top-N — porque mede a mesma coisa em todas:
*a ordenação do modelo bate com a ordenação do futuro?* Excesso por trade
NÃO serve de eixo: muda de significado quando o gate muda.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

DIRECAO_MELHOROU: Final[str] = "melhorou"
DIRECAO_PIOROU: Final[str] = "piorou"
DIRECAO_IGUAL: Final[str] = "igual"
DIRECAO_PRIMEIRA: Final[str] = "primeira_leitura"
DIRECAO_INCOMPARAVEL: Final[str] = "incomparavel"

DELTA_IRRELEVANTE: Final[float] = 0.005
"""Abaixo disto em Rank IC, a diferença entre duas leituras é declarada IGUAL.

Não é gosto: o erro padrão do Rank IC médio diário nas amostras deste projeto
(1.000–5.000 dias, desvio diário ~0,15–0,25) fica na casa de 0,003–0,008.
Chamar de "melhora" um delta menor que o próprio ruído da medida é como o
projeto já se enganou antes com `t` inflado — só que na direção do otimismo."""


@dataclass(frozen=True)
class RegistroLeitura:
    """Uma leitura, no formato mínimo que permite comparar com as outras.

    `desenho` é o campo que dá sentido ao delta: sem saber O QUE mudou entre
    duas leituras, saber que piorou 0,006 não ensina nada. É texto livre de
    propósito — é a hipótese que o experimento testou, escrita por quem o
    rodou."""

    familia: str
    mercado: str
    horizonte: int
    medido_em: str
    rank_ic_medio: float
    rank_ic_t: float
    n_dias: int
    desenho: str
    veredito: str
    braco: str = ""
    excesso_por_trade: float | None = None
    n_trades: int | None = None
    observacoes: dict[str, Any] = field(default_factory=dict)

    @property
    def eixo(self) -> tuple[str, int]:
        """A chave de comparabilidade: mesmo mercado, mesmo horizonte."""
        return (self.mercado, self.horizonte)


@dataclass(frozen=True)
class Comparacao:
    """O veredito de uma leitura CONTRA a melhor anterior do mesmo eixo."""

    registro: RegistroLeitura
    anterior: RegistroLeitura | None
    delta_rank_ic: float | None
    direcao: str
    licao: str


def registrar(caminho: Path, registro: RegistroLeitura) -> None:
    """Acrescenta a leitura ao histórico (JSONL append-only).

    Append-only por decisão, mesma doutrina do ledger de custo de LLM: uma
    leitura ruim não pode ser apagada para o histórico ficar bonito. O contador
    de tentativas do ADR 0032 §9 depende de o histórico ser completo — se ele
    puder ser podado, o denominador do FDR vira ficção."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(registro), ensure_ascii=False) + "\n")


def carregar(caminho: Path) -> list[RegistroLeitura]:
    """Lê o histórico. Arquivo ausente devolve `[]`; linha ilegível LEVANTA —
    mesma doutrina de `COST_TRACKER_LEDGER_CORROMPIDO01`, porque um histórico
    parcial lido como completo faria uma piora parecer primeira leitura."""
    if not caminho.exists():
        return []
    registros: list[RegistroLeitura] = []
    for numero, linha in enumerate(caminho.read_text(encoding="utf-8").splitlines(), start=1):
        if not linha.strip():
            continue
        try:
            registros.append(RegistroLeitura(**json.loads(linha)))
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"{caminho}:{numero} — registro de leitura ilegível: {exc}") from exc
    return registros


def comparar_com_historico(
    novo: RegistroLeitura,
    historico: list[RegistroLeitura],
    *,
    delta_irrelevante: float = DELTA_IRRELEVANTE,
) -> Comparacao:
    """
    Confronta `novo` com a MELHOR leitura anterior do mesmo eixo
    `(mercado, horizonte)`.

    Melhor = maior `rank_ic_medio`. Comparar contra a melhor, e não contra a
    última, é deliberado: comparar com a última permitiria uma sequência de
    pioras pequenas ser lida como "estável", enquanto a distância para o melhor
    já conhecido cresce. É o mesmo motivo pelo qual champion/challenger compara
    com o campeão, nunca com o experimento anterior.

    Uma leitura só é comparável a outra do MESMO eixo. Eixos diferentes
    devolvem `DIRECAO_INCOMPARAVEL` com delta `None` — nunca um número.
    """
    anteriores = [r for r in historico if r.eixo == novo.eixo and r.medido_em < novo.medido_em]
    if not anteriores:
        return Comparacao(
            novo, None, None, DIRECAO_PRIMEIRA,
            f"primeira leitura do eixo {novo.mercado}/h{novo.horizonte} — "
            "nada a comparar, o número entra como linha de base",
        )

    melhor = max(anteriores, key=lambda r: r.rank_ic_medio)
    delta = novo.rank_ic_medio - melhor.rank_ic_medio

    if abs(delta) < delta_irrelevante:
        direcao = DIRECAO_IGUAL
        licao = (
            f"delta {delta:+.4f} está dentro do ruído da própria medida "
            f"({delta_irrelevante:.3f}) — a mudança de desenho não moveu o eixo. "
            f"Antes: {melhor.desenho}. Agora: {novo.desenho}."
        )
    elif delta > 0:
        direcao = DIRECAO_MELHOROU
        licao = (
            f"Rank IC subiu {delta:+.4f} (de {melhor.rank_ic_medio:+.4f} para "
            f"{novo.rank_ic_medio:+.4f}). O que mudou: {novo.desenho} (antes: {melhor.desenho})."
        )
    else:
        direcao = DIRECAO_PIOROU
        licao = (
            f"Rank IC CAIU {delta:+.4f} (de {melhor.rank_ic_medio:+.4f} para "
            f"{novo.rank_ic_medio:+.4f}). O que mudou: {novo.desenho} (antes: {melhor.desenho}). "
            "Piorar com desenho novo é informação: descarta a hipótese que motivou a mudança."
        )
    return Comparacao(novo, melhor, delta, direcao, licao)


def resumo_de_aprendizado(historico: list[RegistroLeitura]) -> dict[str, Any]:
    """
    O que o projeto aprendeu, por eixo: quantas tentativas, qual o melhor Rank
    IC já visto, e se a tendência das tentativas é de melhora ou de piora.

    `tentativas` é o número que o ADR 0032 §9 manda não zerar — ele aparece
    aqui para que qualquer leitura futura seja lida com o denominador certo à
    vista, em vez de o contador viver só na cabeça de quem lembra.
    """
    por_eixo: dict[str, dict[str, Any]] = {}
    for r in historico:
        chave = f"{r.mercado}/h{r.horizonte}"
        bloco = por_eixo.setdefault(
            chave, {"tentativas": 0, "melhor_rank_ic": float("-inf"), "melhor_desenho": "",
                    "ultimo_rank_ic": float("nan"), "ultimo_desenho": "", "algum_positivo": False}
        )
        bloco["tentativas"] += 1
        if r.rank_ic_medio > bloco["melhor_rank_ic"]:
            bloco["melhor_rank_ic"] = r.rank_ic_medio
            bloco["melhor_desenho"] = r.desenho
        bloco["ultimo_rank_ic"] = r.rank_ic_medio
        bloco["ultimo_desenho"] = r.desenho
        bloco["algum_positivo"] = bloco["algum_positivo"] or r.rank_ic_medio > 0

    return {
        "n_leituras": len(historico),
        "n_eixos": len(por_eixo),
        "tentativas_totais": sum(b["tentativas"] for b in por_eixo.values()),
        "eixos": por_eixo,
    }

"""
src/radar/cerebro/estado.py — o que sobrevive entre ciclos.

Pergunta única: **"o que eu sei, o que estou esperando, e quanto ainda posso
gastar?"**

⚠️ SEM ESTADO NÃO HÁ AGENTE
---------------------------
É a tabela do ADR 0037 §2 virando código. Sem isto o Cérebro é uma **função pura
chamada repetidamente** — que é exatamente o que existia antes, e o motivo de
cada execução recomeçar do zero.

| componente | onde vive |
|---|---|
| conhecimento | mapa de validade (injetado) |
| evidência | Ledger (injetado) |
| teses vivas | `teses.RepositorioDeTeses` |
| orçamento | derivado da grade a cada leitura |
| percepção corrente | o último ciclo |
| missão | `politica.Missao` |

⚠️ O ORÇAMENTO NÃO É ARMAZENADO
-------------------------------
Ele é **recalculado** toda vez que se lê o estado, e essa escolha é a emenda 2
aplicada aqui: guardar o número o transformaria em contador com outro nome, e ele
ficaria velho no instante em que a grade mudasse. O que se persiste é a grade; o
orçamento é uma leitura dela.

⚠️ ESTADO PERSISTIDO É SUPERFÍCIE DE BUG NOVA
---------------------------------------------
Está declarado nas Consequências negativas do ADR 0037: *estado corrompido
produz decisão errada que parece deliberada*. É por isso que `Estado` é
**imutável**, que toda transição devolve cópia, e que `instantaneo()` existe —
um estado que não se consegue inspecionar num painel é um estado que ninguém
audita.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime

from radar.cerebro.orcamento import HipoteseNaGrade, Orcamento, espaco_restante
from radar.cerebro.politica import Missao
from radar.cerebro.teses import RepositorioDeTeses, Tese, resumo_de_mudez


@dataclass(frozen=True, slots=True)
class Estado:
    """A memória operacional do Cérebro.

    `grade` e `teses` são as duas coisas que **precisam** sobreviver ao processo.
    O resto é derivado ou efêmero, e derivar em vez de guardar é o que impede
    duas verdades sobre o mesmo fato.
    """

    missao: Missao
    #: A grade de tentativas já registradas — o denominador do FDR. É o que se
    #: persiste; o orçamento sai dela.
    grade: tuple[HipoteseNaGrade, ...] = ()
    #: Onde as teses vivem. Injetado: `cerebro` não escolhe formato em disco.
    teses: RepositorioDeTeses | None = None
    #: Hipóteses do lote que está rodando AGORA. Enquanto não estiver vazio, as
    #: definições dessas hipóteses são imutáveis (emenda 3).
    lote_em_execucao: tuple[str, ...] = ()
    #: Quando o estado foi montado. `None` em teste puro.
    momento: datetime | None = None
    #: Trilha do que o Cérebro fez. Append-only, e inclui `NENHUMA_ACAO`.
    diario: tuple[str, ...] = field(default_factory=tuple)

    @property
    def orcamento(self) -> Orcamento:
        """Recalculado, nunca lido de campo. Ver o aviso no cabeçalho."""
        return espaco_restante(self.grade)

    @property
    def teses_abertas(self) -> Sequence[Tese]:
        return self.teses.abertas() if self.teses is not None else ()

    @property
    def lote_rodando(self) -> bool:
        return bool(self.lote_em_execucao)

    def com(self, *, nota: str | None = None, **campos: object) -> Estado:
        """Nova versão do estado. Imutável de propósito — ver o cabeçalho."""
        diario = (*self.diario, nota) if nota else self.diario
        return replace(self, diario=diario, **campos)  # type: ignore[arg-type]

    def instantaneo(self) -> dict[str, object]:
        """O estado inteiro em forma legível — para painel, log e autópsia.

        Publica as contagens de mudez **sempre**, inclusive zeradas: um Cérebro
        com centenas de `aguardando` e zero `resolvida` não está sendo seletivo,
        está travado, e sem este número os dois casos são indistinguíveis
        (G-P2).
        """
        orcamento = self.orcamento
        return {
            "missao": self.missao.value,
            "momento": self.momento.isoformat() if self.momento else None,
            "grade": len(self.grade),
            "orcamento": {
                "teto": orcamento.teto,
                "gasto": orcamento.gasto,
                "restante": orcamento.restante,
                "alvo": orcamento.alvo,
                "q": orcamento.q,
            },
            "teses": resumo_de_mudez(self.teses.abertas()) if self.teses else {},
            "lote_em_execucao": list(self.lote_em_execucao),
            "ciclos_registrados": len(self.diario),
        }

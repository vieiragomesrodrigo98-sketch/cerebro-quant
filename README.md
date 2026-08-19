# Cérebro — motor de descoberta de oportunidades

**Um sistema quantitativo construído para não acreditar em si mesmo.**
*A quantitative engine built to distrust its own results.*

🇧🇷 [Português](#português) · 🇺🇸 [English](#english)

<sub>597 testes · 57 módulos · 11,5k linhas · Python 3.11+ · sem banco, sem rede, sem configuração</sub>

---

## Português

Este repositório é um **extrato executável** do núcleo de decisão de um sistema privado
de pesquisa quantitativa em mercados financeiros. O produto completo não é público; o
que está aqui roda sozinho, com testes verdes, e mostra a parte que interessa: **como o
sistema decide em quem acreditar.**

### O problema

Procurar padrão em série de preços acha padrão. Sempre. Com dados suficientes e
liberdade suficiente, qualquer pessoa produz um backtest bonito — e a maior parte deles
é ruído com aparência de descoberta. O trabalho difícil não é encontrar sinal; é
**construir o que impede você de se enganar**.

### A resposta: cinco portões mecânicos

Nenhuma estratégia é levada a sério antes de passar por `cerebro/contrato.py`. São
portões verificados por `pytest`, nunca por revisão humana:

| Portão | O que exige |
|---|---|
| **G-P0** | O instrumento aprendeu de fato — não é um modelo degenerado |
| **G-P1** | Nenhuma coluna constante dentro do instante no espaço de seleção |
| **G-P2** | A saída **discrimina** — silêncio é FALHA, nunca abstenção |
| **G-P3** | O limiar é alcançável na escala em que se expressa |
| **G-P4** | Vocabulário de contexto comum entre mercados |

O **G-P2** é o mais contraintuitivo e o mais caro de aprender. Um modelo que não emite
nada parece prudente; ele apenas não está sendo medido. Num episódio real, um emissor
ficou mudo por um mês inteiro em produção e foi lido como *seletivo* — o limiar de corte
estava acima do teto que o calibrador conseguia alcançar. Nenhum teste apontava para
isso, porque não existia teste que tratasse mudez como defeito. O **G-P3** nasceu do
mesmo episódio.

### Orçamento de tentativas: derivado, nunca contado

`cerebro/orcamento.py` resolve multiplicidade sem contador. Testar muitas hipóteses
garante achar "resultado significante" por acaso; a defesa usual é contar tentativas e
parar. Isso é errado — o espaço admissível sob Benjamini-Hochberg depende da
**distribuição** dos p-valores, não da contagem.

Então `remaining -= 1` não existe aqui. Uma hipótese nula **consome** espaço
estatístico; uma hipótese forte **devolve** espaço. O orçamento é recalculado a partir
da grade inteira toda vez.

### O guardrail epistemológico

Dois erros são fáceis e caros, e o código os trata como bugs:

- **Resultado da estratégia ≠ propriedade do ativo.** Uma estratégia que ganhou em PETR4
  não torna PETR4 um bom ativo.
- **Hipótese não separável fica inconclusiva** — não se fabrica um filtro para salvá-la.
  `cerebro/historico.py` recusa comparar leituras em eixos diferentes em vez de produzir
  um número enganoso.

### Mapa

```
src/radar/cerebro/
  contrato.py     os 5 portões — o coração deste repositório
  orcamento.py    espaço estatístico derivado da distribuição de p-valores
  ranking.py      seleção cross-sectional por percentil do instante
  politica.py     guardrail sem estado: decide o que é PERMITIDO, nunca o que FAZER
  teses.py        AGUARDAR como decisão persistida, com condição de despertar
  risco.py        drawdown, curva de patrimônio, guardas de sanidade de preço
  diagnostico.py  por que erramos — separado de quanto ganhamos
  ...             34 módulos no total
tests/            597 testes, um arquivo por módulo do Cérebro
```

### Rodar

```bash
python -m venv .venv
pip install -e ".[dev]"
pytest -q
```

Sem banco, sem rede, sem chave de API, sem arquivo de configuração.

### Honestidade sobre o escopo

Isto é um **extrato**, não o produto. Ficaram de fora: ingestão de dados, feature stores,
camada de execução, API, frontend e o histórico de pesquisa. Identificadores de
estratégia foram anonimizados. Os `__init__.py` dos pacotes vizinhos foram esvaziados de
propósito — no repositório de origem eles reexportam o subpacote inteiro e arrastariam a
camada de configuração junto.

---

## English

This repository is an **executable extract** from the decision core of a private
quantitative research system for financial markets. The full product is not public; what
is here runs standalone, with a green test suite, and shows the part that matters: **how
the system decides what to believe.**

### The problem

Searching for patterns in price series finds patterns. Always. With enough data and
enough freedom, anyone produces a beautiful backtest — and most of them are noise that
looks like discovery. The hard work is not finding signal; it is **building the thing
that stops you from fooling yourself**.

### The answer: five mechanical gates

No strategy is taken seriously before passing `cerebro/contrato.py` — gates verified by
`pytest`, never by human review:

| Gate | What it requires |
|---|---|
| **G-P0** | The instrument actually learned — not a degenerate model |
| **G-P1** | No within-instant constant column in the selection space |
| **G-P2** | The output **discriminates** — silence is FAILURE, never abstention |
| **G-P3** | The threshold is reachable on the scale it is expressed in |
| **G-P4** | Shared context vocabulary across markets |

**G-P2** is the least intuitive and the most expensive lesson. A model that emits nothing
looks prudent; it is simply not being measured. In a real incident, an emitter went
silent in production for a full month and was read as *selective* — its cutoff sat above
the ceiling its calibrator could reach. No test caught it, because no test treated
muteness as a defect. **G-P3** was born from the same incident.

### Trial budget: derived, never counted

`cerebro/orcamento.py` handles multiplicity without a counter. Testing many hypotheses
guarantees finding "significant results" by chance; the usual defense is to count trials
and stop. That is wrong — the admissible space under Benjamini-Hochberg depends on the
**distribution** of p-values, not on the count.

So `remaining -= 1` does not exist here. A null hypothesis **consumes** statistical
space; a strong one **gives space back**. The budget is recomputed from the full grid
every time.

### The epistemological guardrail

Two mistakes are easy and expensive, and the code treats them as bugs:

- **A strategy's result is not a property of the asset.** A strategy that profited on
  PETR4 does not make PETR4 a good asset.
- **A non-separable hypothesis stays inconclusive** — you do not fabricate a filter to
  rescue it. `cerebro/historico.py` refuses to compare readings on different axes rather
  than emitting a misleading number.

### Running it

```bash
python -m venv .venv
pip install -e ".[dev]"
pytest -q
```

No database, no network, no API key, no config file.

### Scope, honestly

This is an **extract**, not the product. Left out: data ingestion, feature stores,
execution layer, API, frontend, and the research history. Strategy identifiers are
anonymized. Neighboring packages' `__init__.py` files were emptied deliberately — in the
source repository they re-export entire subpackages and would drag the configuration
layer along.

---

<sub>MIT © 2026 Rodrigo Gomes Vieira</sub>

# Cérebro — opportunity discovery engine

[![CI](https://github.com/rodrigogvieira98/cerebro-quant/actions/workflows/ci.yml/badge.svg)](https://github.com/rodrigogvieira98/cerebro-quant/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) ![Python](https://img.shields.io/badge/python-3.11%2B-blue)

**A quantitative engine built to distrust its own results.**

🇧🇷 [Português](README.md) · 🇪🇸 [Español](README.es.md) · 🇺🇸 **English**

<sub>597 tests · 57 modules · 11.5k lines · Python 3.11+ · no database, no network, no config</sub>

---


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

> Code and internal documentation are in Portuguese — the language the system
> was designed in. Identifiers (classes, functions) are in English.

---

<sub>MIT © 2026 Rodrigo Gomes Vieira</sub>

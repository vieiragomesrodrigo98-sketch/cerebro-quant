# Cérebro — motor de descoberta de oportunidades

[![CI](https://github.com/vieiragomesrodrigo98-sketch/cerebro-quant/actions/workflows/ci.yml/badge.svg)](https://github.com/vieiragomesrodrigo98-sketch/cerebro-quant/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) ![Python](https://img.shields.io/badge/python-3.11%2B-blue)

**Um sistema quantitativo construído para não acreditar em si mesmo.**

🇧🇷 **Português** · 🇪🇸 [Español](README.es.md) · 🇺🇸 [English](README.en.md)

<sub>597 testes · 57 módulos · 11,5k linhas · Python 3.11+ · sem banco, sem rede, sem configuração</sub>

---


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

> O código e a documentação interna estão em português — é a língua em que o
> sistema foi desenhado. Identificadores (classes, funções) em inglês.

---

<sub>MIT © 2026 Rodrigo Gomes Vieira</sub>

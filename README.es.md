# Cerebro — motor de descubrimiento de oportunidades

[![CI](https://github.com/rodrigogvieira98/cerebro-quant/actions/workflows/ci.yml/badge.svg)](https://github.com/rodrigogvieira98/cerebro-quant/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) ![Python](https://img.shields.io/badge/python-3.11%2B-blue)

**Un sistema cuantitativo construido para no creerse a sí mismo.**

🇧🇷 [Português](README.md) · 🇪🇸 **Español** · 🇺🇸 [English](README.en.md)

<sub>597 pruebas · 57 módulos · 11,5k líneas · Python 3.11+ · sin base de datos, sin red, sin configuración</sub>

---

Este repositorio es un **extracto ejecutable** del núcleo de decisión de un sistema
privado de investigación cuantitativa en mercados financieros. El producto completo no
es público; lo que está aquí funciona por sí solo, con las pruebas en verde, y muestra
la parte que importa: **cómo el sistema decide en quién creer.**

## El problema

Buscar patrones en series de precios encuentra patrones. Siempre. Con suficientes datos
y suficiente libertad, cualquiera produce un backtest bonito — y la mayoría son ruido
con apariencia de descubrimiento. El trabajo difícil no es encontrar la señal; es
**construir aquello que te impide engañarte a ti mismo**.

## La respuesta: cinco compuertas mecánicas

Ninguna estrategia se toma en serio antes de pasar por `cerebro/contrato.py`. Son
compuertas verificadas por `pytest`, nunca por revisión humana:

| Compuerta | Qué exige |
|---|---|
| **G-P0** | El instrumento realmente aprendió — no es un modelo degenerado |
| **G-P1** | Ninguna columna constante dentro del instante en el espacio de selección |
| **G-P2** | La salida **discrimina** — el silencio es FALLO, nunca abstención |
| **G-P3** | El umbral es alcanzable en la escala en que se expresa |
| **G-P4** | Vocabulario de contexto común entre mercados |

**G-P2** es la menos intuitiva y la lección más cara. Un modelo que no emite nada parece
prudente; simplemente no está siendo medido. En un incidente real, un emisor quedó mudo
durante un mes entero en producción y se leyó como *selectivo* — su umbral de corte
estaba por encima del techo que su calibrador podía alcanzar. Ninguna prueba lo detectó,
porque no existía prueba que tratara la mudez como defecto. **G-P3** nació del mismo
incidente.

## Presupuesto de intentos: derivado, nunca contado

`cerebro/orcamento.py` resuelve la multiplicidad sin contador. Probar muchas hipótesis
garantiza encontrar "resultados significativos" por azar; la defensa habitual es contar
intentos y parar. Eso es incorrecto — el espacio admisible bajo Benjamini-Hochberg
depende de la **distribución** de los p-valores, no del conteo.

Por eso `remaining -= 1` no existe aquí. Una hipótesis nula **consume** espacio
estadístico; una hipótesis fuerte **devuelve** espacio. El presupuesto se recalcula a
partir de la rejilla completa cada vez.

## La barrera epistemológica

Dos errores son fáciles y caros, y el código los trata como bugs:

- **El resultado de la estrategia no es una propiedad del activo.** Una estrategia que
  ganó en PETR4 no convierte a PETR4 en un buen activo.
- **Una hipótesis no separable queda inconclusa** — no se fabrica un filtro para
  rescatarla. `cerebro/historico.py` se niega a comparar lecturas en ejes distintos en
  lugar de producir un número engañoso.

## Mapa

```
src/radar/cerebro/
  contrato.py     las 5 compuertas — el corazón de este repositorio
  orcamento.py    espacio estadístico derivado de la distribución de p-valores
  ranking.py      selección transversal por percentil del instante
  politica.py     barrera sin estado: decide qué está PERMITIDO, nunca qué HACER
  teses.py        ESPERAR como decisión persistida, con condición de despertar
  risco.py        drawdown, curva de patrimonio, guardas de sanidad de precio
  diagnostico.py  por qué fallamos — separado de cuánto ganamos
  ...             34 módulos en total
tests/            597 pruebas, un archivo por módulo del Cerebro
```

## Ejecutar

```bash
python -m venv .venv
pip install -e ".[dev]"
pytest -q
```

Sin base de datos, sin red, sin clave de API, sin archivo de configuración.

## Honestidad sobre el alcance

Esto es un **extracto**, no el producto. Quedaron fuera: ingesta de datos, feature
stores, capa de ejecución, API, frontend y el historial de investigación. Los
identificadores de estrategia están anonimizados. Los `__init__.py` de los paquetes
vecinos fueron vaciados a propósito — en el repositorio de origen reexportan el
subpaquete entero y arrastrarían la capa de configuración consigo.

> El código y la documentación interna están en portugués — es el idioma en que el
> sistema fue diseñado. Los identificadores (clases, funciones) están en inglés.

---

<sub>MIT © 2026 Rodrigo Gomes Vieira</sub>

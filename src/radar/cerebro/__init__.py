"""
`radar.cerebro` — o que é COMUM a todos os motores do Cérebro v2.

O Cérebro v2 tem especialistas separados por família de horizonte (Scalp,
Day, Swing) porque as premissas econômicas deles são incompatíveis — um
scalp não pode dividir espaço de features com um swing. Mas eles não são
quatro cérebros isolados: o que faz deles UM Cérebro é o Router que decide
qual especialista está ativo em cada momento, e o Router só é construível se
todos os especialistas falarem o MESMO vocabulário de contexto e publicarem
mapa de validade no MESMO formato.

Este pacote é esse denominador comum. Nasceu com uma peça
(`radar.cerebro.contrato`, o Contrato de Pensamento do ADR 0032) e hoje tem
19 módulos — contrato, família, catálogo, ranking, risco, custos, alocação,
oportunidade, trava por ativo, mapa de validade, diagnóstico, autópsia,
capacidade, estabilidade, operabilidade, alvo, pesquisa e histórico.

Nada aqui conhece um motor específico: zero import de `radar.mie`,
`radar.scalp` ou `radar.engines`. A dependência é sempre no sentido
motor → contrato, nunca o inverso.

Esta invariante é **verificada por `tests/unit/test_arquitetura_camadas.py`**,
não apenas declarada aqui. A diferença não é acadêmica: entre 2026-08-04 e
2026-08-07 esta docstring afirmava "zero import de `radar.mie`" enquanto
`catalogo.py` importava `radar.mie.dataset` para declarar uma família — a
invariante mais estrutural do ADR 0032, violada dentro do pacote que a
declara, porque nada a checava. O vocabulário de colunas mudou para
`radar.features.colunas` (folha sem dependências, ver o docstring de lá) e a
regra virou teste.
"""

# Parte 2 — Curva de orçamento

Seeds 42, 43 e 44 no orçamento 5; seed 42 nos demais. Desvio amostral (ddof=1).
As seeds alteram o treino, mantendo os CSVs oficiais e o checkpoint SimCLR fixos.
Os dois métodos usam o mesmo pré-processamento, otimizador, taxa de aprendizado e seleção pelo val.
Teto fornecido pelo monitor: 0,686; referência, não limite teórico.

| Método | Rótulos/classe | F1 macro (média ± desvio) | Acurácia (média ± desvio) |
|---|---:|---:|---:|
| piso | 1 | 0.2522 ± 0.0000 | 0.2762 ± 0.0000 |
| piso | 5 | 0.4074 ± 0.0127 | 0.4390 ± 0.0093 |
| piso | 20 | 0.5651 ± 0.0000 | 0.6079 ± 0.0000 |
| simclr | 1 | 0.1986 ± 0.0000 | 0.2158 ± 0.0000 |
| simclr | 5 | 0.3667 ± 0.0417 | 0.4254 ± 0.0448 |
| simclr | 20 | 0.4662 ± 0.0000 | 0.5438 ± 0.0000 |

Maiores ganhos médios positivos: white_tailed_deer (+0.1549), eastern_cottontail (+0.0928), american_crow (+0.0773), domestic_cat (+0.0750), northern_raccoon (+0.0628).

Sem ganho médio positivo: bobcat, eastern_gray_squirrel, eastern_chipmunk, striped_skunk, grey_fox, bird, chicken, coyote, red_fox, wild_turkey, eastern_fox_squirrel, american_black_bear, horse.

O ganho é calculado entre seeds pareadas. Três seeds não demonstram significância estatística; classes com poucos exemplos no teste podem variar bastante.
As predições individuais e os F1 de todas as classes ficam nos CSVs/JSONs para auditoria.
# Validação por array — Human Randomwalk

Dataset: `espargos-0007-human-helmet-randomwalk-1.tfrecords`

Split: validação, 111 clusters

Métrica: MAE em graus


| Método | Energia | Componentes | Compressão | A0 | A1 | A2 | A3 | Global |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uRoot-MUSIC + CRAP | - | - | 1.00x | 2,67 | 5,47 | 7,03 | 3,54 | 4,68 |
| PCA complexa + CRAP | 95% | 72 | 23.56x | 2,67 | 5,56 | 7,08 | 3,57 | 4,72 |
| PCA real/imaginária + CRAP | 95% | 144 | 23.56x | 2,68 | 5,56 | 7,08 | 3,58 | 4,72 |
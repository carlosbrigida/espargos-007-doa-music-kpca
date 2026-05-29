Relatório Técnico – Fase 2
Análise dos Datasets Completos e Comparação PCA × KPCA
Projeto

ESPARGOS-0007 – Análise de Subespaços em Dados CSI para Aplicações MUSIC

1. Objetivo da Fase

Após a análise preliminar utilizando subconjuntos de 1000 snapshots, o objetivo desta fase foi:

Repetir todas as análises utilizando datasets completos.
Verificar se as conclusões obtidas anteriormente permaneciam válidas.
Comparar cenários simples e dinâmicos.
Avaliar a preservação do subespaço dominante por PCA e KPCA.
Investigar o potencial do KPCA como etapa de pré-processamento para MUSIC.
2. Datasets Utilizados
Standing Center

Arquivo:

espargos-0007-human-helmet-standing-center-1.tfrecords

Características:

Snapshots: 5041
Duração: 82.9 s
Taxa aproximada: 60.8 Hz

Faixa espacial:

X: 0.8101 → 0.8314 m
Y: 3.4578 → 3.5333 m
Z: -0.5056 → -0.5033 m

Representa um cenário praticamente estático.

Circle

Arquivo:

espargos-0007-circle-1.tfrecords

Características:

Snapshots: 77076
Duração: 1290.2 s
Taxa aproximada: 59.7 Hz

Faixa espacial:

X: -0.8925 → 2.2925 m
Y: 1.7724 → 4.7457 m
Z: -1.4100 → -1.4003 m

Representa um cenário altamente dinâmico com grande exploração espacial.

3. Eigenspectrum dos Datasets Completos
Standing

Autovalores dominantes:

λ1 = 5.99e6
λ2 = 4.76e6
λ3 = 4.09e6
λ4 = 3.02e6
λ5 = 2.04e4

Energia acumulada:

90% -> 4 componentes
95% -> 4 componentes
99% -> 11 componentes

Eigengap:

λ4 / λ5 = 147.8
Circle

Autovalores dominantes:

λ1 = 5.77e6
λ2 = 4.33e6
λ3 = 4.08e6
λ4 = 3.08e6
λ5 = 6.84e4

Energia acumulada:

90% -> 4 componentes
95% -> 4 componentes
99% -> 41 componentes

Eigengap:

λ4 / λ5 = 45.1
4. Principais Descobertas do Eigenspectrum
Descoberta 1

Mesmo no cenário Circle completo:

95% da energia permanece concentrada em apenas 4 componentes.

Isso indica uma estrutura dominante extremamente robusta do canal.

Descoberta 2

O movimento circular aumenta significativamente a energia dos modos secundários.

Comparação:

Standing:
99% -> 11 componentes

Circle:
99% -> 41 componentes
Descoberta 3

O eigengap reduz-se significativamente:

Standing:
147.8

Circle:
45.1

Indicando maior complexidade estrutural do canal.

5. Estabilidade Temporal dos Subespaços

Análise realizada utilizando janelas temporais.

Standing

Resultado anterior:

Max angle médio ≈ 4.27°
Mean angle médio ≈ 3.04°
Circle Completo

Resultados:

Janela 100
Max médio = 11.90°
Mean médio = 7.76°
Janela 250
Max médio = 11.18°
Mean médio = 6.96°
Janela 500
Max médio = 12.09°
Mean médio = 7.17°
Janela 1000
Max médio = 12.94°
Mean médio = 7.51°
6. Interpretação Física

O subespaço dominante sofre rotação significativa ao longo da trajetória circular.

Entretanto:

A rotação permanece limitada.

O subespaço não se desorganiza completamente.

Isso sugere que:

A dinâmica do canal ocorre dentro de uma variedade de baixa dimensão.
7. Benchmark Computacional
Standing
RAM CSI: 0.064 GB

Leitura:      0.29 s
Covariância:  0.08 s
Autovalores:  0.70 s

Tempo total:  1.07 s
Circle
RAM CSI: 0.974 GB

Leitura:      4.00 s
Covariância:  1.05 s
Autovalores:  0.74 s

Tempo total:  5.94 s
8. PCA – Dataset Completo
Standing
95% da variância
8 componentes

MSE = 80.59

Max angle = 0.000094°
Mean angle = 0.000061°
Circle
95% da variância
8 componentes

MSE = 265.71

Max angle = 0.000087°
Mean angle = 0.000058°
9. Principal Resultado da PCA

Apesar do aumento da complexidade do cenário:

O PCA preserva o subespaço dominante praticamente sem erro angular.

Na prática:

Erro angular ≈ 10⁻⁴ graus

Portanto:

PCA é extremamente compatível com algoritmos baseados em subespaços.
10. Otimização do KPCA

Foi realizada uma varredura de parâmetros:

γ ∈ [1e-5, 1e-2]

utilizando amostragem uniforme do dataset Circle.

Melhor resultado:

γ = 3e-4
11. Comparação Direta PCA × KPCA

Mesmo número de componentes:

8 componentes
PCA
MSE = 260.81

Max angle = 0.063°
Mean angle = 0.046°
KPCA (γ = 3e-4)
MSE = 4956.14

Max angle = 18.06°
Mean angle = 4.72°
12. Conclusão sobre KPCA Reconstrutivo

O KPCA reconstrutivo apresentou:

Erro de reconstrução significativamente maior
+
Grande distorção do subespaço dominante

Consequentemente:

KPCA não deve ser utilizado como substituto direto da PCA para reconstrução do CSI antes do MUSIC.
13. Conclusão Geral da Fase

As análises realizadas com datasets completos confirmaram que:

O canal possui um subespaço dominante de aproximadamente quatro dimensões.
Essa estrutura permanece presente mesmo em trajetórias longas e altamente dinâmicas.
O movimento aumenta a energia dos modos secundários, mas não destrói a estrutura principal.
PCA preserva o subespaço dominante com precisão praticamente perfeita.
KPCA reconstrutivo não apresentou vantagens para preservação de subespaços.
14. Próxima Fase

A hipótese mais promissora para a inovação da tese não é utilizar KPCA para reconstruir o CSI.

As próximas investigações deverão explorar KPCA como:

• gerador de pesos para snapshots
• detector de regimes de propagação
• segmentador de estados do canal
• mecanismo de seleção de snapshots
• pré-processamento adaptativo para MUSIC

Ou seja:

KPCA não necessariamente substituindo MUSIC,
mas auxiliando MUSIC.
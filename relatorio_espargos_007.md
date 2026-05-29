Relatório Técnico – Exploração Inicial do Dataset ESPARGOS-0007 e Avaliação de PCA/KPCA
1. Objetivo da etapa

O objetivo desta fase foi:

Configurar o ambiente de desenvolvimento.
Inspecionar e compreender a estrutura do dataset ESPARGOS-0007.
Decodificar os arquivos TFRecord.
Caracterizar estatisticamente o CSI.
Investigar a estrutura de subespaços do cenário.
Avaliar a estabilidade temporal dos subespaços.
Comparar PCA e KPCA sob a perspectiva de preservação de subespaços, em vez de apenas erro de reconstrução.
2. Ambiente computacional

Foi configurado:

VS Code
Python 3.12.10
Ambiente virtual (venv)
TensorFlow
NumPy
SciPy
Scikit-Learn

Durante os testes o TensorFlow identificou suporte a:

AVX
AVX2
AVX512F
AVX512_VNNI
AVX512_BF16
FMA

indicando que o novo computador está utilizando otimizações modernas de CPU.

3. Estrutura do dataset ESPARGOS-0007

Foram identificados diversos arquivos TFRecord:

espargos-0007-empty-room.tfrecords
espargos-0007-human-helmet-standing-center-1.tfrecords
espargos-0007-randomwalk-1.tfrecords
espargos-0007-circle-1.tfrecords
espargos-0007-spiral-ccw-1-part1.tfrecords
...

Tamanhos variando de:

66 MB
até
2.7 GB
4. Estrutura interna dos registros

Cada registro contém:

time
rssi
mac
pos
csi

Após desserialização com TensorFlow:

CSI
shape = (4, 2, 4, 53)
dtype = complex64
RSSI
shape = (4, 2, 4)
dtype = float32
POS
shape = (3,)
dtype = float64
TIME
shape = ()
dtype = float64
5. Interpretação física do cenário

A documentação analisada mostrou:

Infraestrutura
4 arrays receptores
North
South
East
West
4 transmissores
TX1
TX2
TX3
TX4
Alvos
ambiente vazio
pessoa com refletor
trajetória circular
random walk
espirais
trajetórias meandrantes
6. Análise do cenário escolhido

Foi utilizado inicialmente:

human-helmet-standing-center-1

por ser um cenário praticamente estacionário.

7. Contagem de snapshots

Número total de registros:

5041

Duração:

82.9 s

Taxa de aquisição:

≈ 60.8 snapshots/s
8. Análise da trajetória

Posição inicial:

[0.8298, 3.5223, -0.5050]

Posição final:

[0.8150, 3.5139, -0.5044]

Variações observadas:

X
≈ 2.1 cm
Y
≈ 7.5 cm
Z
≈ 2.3 mm

Conclusão:

Alvo praticamente estacionário.

Esse cenário é adequado como baseline.

9. Estatísticas globais do CSI

Tensor completo:

(5041, 4, 2, 4, 53)

Magnitude:

Mean = 99.42
Std  = 28.51
Min  = 0
Max  = 181

Fase:

Mean = 0.0067
Std  = 1.8147
Range = [-π, π]

Conclusões:

canal não é perfeitamente estático;
existem variações temporais;
fase bruta não está alinhada;
comportamento compatível com ambiente indoor real.
10. Construção do subconjunto para análise

Foi criado:

standing_center_1000.npz

contendo:

1000 snapshots

para acelerar experimentos.

11. Análise espectral do subespaço

Covariância construída sobre:

1000 snapshots
1696 variáveis complexas

Autovalores dominantes:

λ1 = 5.98e6
λ2 = 4.74e6
λ3 = 4.09e6
λ4 = 2.96e6

Quinto autovalor:

λ5 = 2.34e4

Eigengap:

λ4 / λ5 ≈ 126
12. Descoberta principal

A energia acumulada mostrou:

90% -> 4 componentes
95% -> 4 componentes
99% -> 10 componentes

Resultado extremamente relevante.

Apesar de existirem:

1696 dimensões complexas

apenas:

4 componentes

explicam mais de:

95%

da energia observada.

13. Interpretação científica

O cenário parece viver em:

subespaço dominante de baixa dimensão

embutido em um espaço de observação muito maior.

Consequências:

PCA faz sentido.
MUSIC tem forte separação sinal/ruído.
Existe estrutura dominante muito clara.
14. Estabilidade temporal dos autovalores

Foi realizada análise em janelas:

100 snapshots

Os quatro maiores autovalores permaneceram estáveis ao longo do tempo.

Não foram observadas:

mudanças abruptas;
transições LOS/NLOS;
mudanças de regime evidentes.
15. Estabilidade dos subespaços

Foram calculados ângulos principais entre subespaços consecutivos.

Resultados:

2° – 6°

tipicamente.

Exemplos:

0→1 : 2.50°
6→7 : 2.09°
8→9 : 4.96°
Conclusão

O cenário:

human-helmet-standing-center

apresenta:

alta estacionariedade;
subespaços estáveis;
eigengap elevado;
baixa dimensionalidade efetiva.

É um excelente cenário baseline para MUSIC.

16. Eigenspectrum

O gráfico mostrou:

Região dominante
4 autovalores principais

claramente separados.

Região intermediária

Cauda longa associada a:

multipath;
variações temporais;
ruído correlacionado.
Piso numérico

Queda brusca próxima do rank máximo imposto pelo número de snapshots.

17. PCA – preservação de subespaços

Foi realizada:

redução
→ reconstrução
→ comparação do subespaço

Resultados:

PCA 90%
7 componentes
Max angle = 0.62°
Mean = 0.15°
PCA 95%
8 componentes
Max angle = 0.00045°
PCA 99%
19 componentes
Max angle = 0.00029°
Conclusão sobre PCA

O PCA preserva praticamente perfeitamente o subespaço dominante.

Em especial:

95% de energia
↓
8 componentes
↓
erro angular ≈ 0°

Resultado extremamente forte.

18. KPCA – análise inicial

Foi utilizado:

KernelPCA(
    kernel="rbf"
)

com reconstrução via:

inverse_transform()
Resultados observados
Erro de reconstrução

PCA:

MSE = 53 – 495

KPCA:

MSE ≈ 5313

para praticamente todas as configurações testadas.

Preservação de subespaços

Resultados variando entre:

15°
e
89°

dependendo de:

gamma

e

n_components
Descoberta importante

O comportamento do KPCA mostrou-se altamente sensível ao parâmetro:

gamma

Enquanto o PCA permaneceu extremamente robusto.

19. Interpretação crítica

Os resultados atuais NÃO permitem concluir que:

KPCA é melhor que PCA

para este cenário.

Pelo contrário.

Neste experimento:

PCA
menor erro;
maior estabilidade;
preservação quase perfeita dos subespaços.
KPCA
reconstrução significativamente pior;
forte distorção dos subespaços;
elevada dependência dos hiperparâmetros.
20. Limitação importante da análise

O teste utilizou:

kpca.inverse_transform()

que depende do problema de:

pre-image

conhecidamente difícil em Kernel PCA.

Portanto:

não se pode concluir que o espaço kernel seja inadequado.

Pode-se apenas concluir que:

a reconstrução para o espaço original
não preservou adequadamente a estrutura observada.
21. Principais descobertas até o momento
Dataset

✓ Estrutura compreendida e decodificada.

CSI

✓ Tensor complexo validado.

Cenário Standing Center

✓ Quase estacionário.

Subespaço

✓ Apenas 4 modos dominam >95% da energia.

MUSIC

✓ Forte separação entre subespaço de sinal e ruído.

PCA

✓ Preservação quase perfeita dos subespaços.

KPCA

✓ Extremamente sensível ao kernel e ao parâmetro γ.

✓ Reconstrução via pre-image produziu forte distorção dos subespaços.

22. Conclusão geral da etapa

O resultado mais importante desta fase não foi sobre KPCA.

Foi a descoberta de que o cenário:

human-helmet-standing-center

possui:

baixa dimensionalidade efetiva;
subespaços extremamente estáveis;
eigengap muito forte;
excelente adequação como baseline para MUSIC.

Além disso, os experimentos mostraram que:

Preservação de subespaços é uma métrica muito mais informativa do que apenas erro de reconstrução ou MAE angular.

Essa observação pode se tornar um dos pilares metodológicos do trabalho, permitindo avaliar PCA, KPCA e futuros métodos de redução de dimensionalidade sob uma perspectiva diretamente ligada à física dos algoritmos de subespaços como o MUSIC.
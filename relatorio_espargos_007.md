Análise Exploratória do Dataset ESPARGOS-0007 para Estudo de Métodos Baseados em Subespaços
Resumo

Este relatório apresenta uma análise exploratória inicial do dataset ESPARGOS-0007, contendo medições de Channel State Information (CSI) obtidas por meio da plataforma ESPARGOS. O objetivo desta etapa foi compreender a estrutura dos dados, caracterizar estatisticamente o cenário experimental selecionado, investigar a dimensionalidade efetiva do canal e avaliar o comportamento de técnicas de redução de dimensionalidade, com ênfase em PCA e Kernel PCA (KPCA). Os resultados indicam que o cenário analisado apresenta forte estrutura de baixa dimensão, elevada estabilidade temporal dos subespaços dominantes e excelente adequação como cenário baseline para futuros estudos envolvendo MUSIC e métodos de preservação de subespaços.

1. Introdução

Métodos de estimação baseados em subespaços, como MUSIC, dependem fortemente da correta separação entre os subespaços de sinal e ruído. Em aplicações envolvendo CSI, técnicas de redução de dimensionalidade são frequentemente utilizadas para reduzir complexidade computacional, remover ruído e melhorar a robustez das estimativas.

Entretanto, a simples redução de erro de reconstrução não garante que a estrutura física relevante para algoritmos baseados em subespaços seja preservada. Assim, torna-se necessário avaliar não apenas a capacidade de reconstrução dos dados, mas também a preservação dos subespaços dominantes do canal.

O presente estudo tem como objetivo caracterizar o dataset ESPARGOS-0007 e investigar preliminarmente o comportamento de PCA e KPCA sob a perspectiva de preservação de subespaços.

2. Dataset ESPARGOS-0007
2.1 Estrutura geral

O dataset ESPARGOS-0007 contém medições de CSI armazenadas em arquivos TFRecord correspondentes a diferentes configurações experimentais.

Entre os cenários disponíveis encontram-se:

Empty Room
Human Helmet Standing Center
Circle
Random Walk
Spiral
Meanders

Os arquivos possuem tamanhos variando entre aproximadamente 66 MB e 2,7 GB.

2.2 Estrutura dos registros

Cada snapshot contém os seguintes campos:

Campo	Descrição
time	Timestamp da medição
rssi	RSSI por enlace
mac	Identificação do transmissor
pos	Posição do alvo
csi	Channel State Information

Após desserialização, observou-se:

CSI
shape = (4, 2, 4, 53)
dtype = complex64
RSSI
shape = (4, 2, 4)
dtype = float32
POS
shape = (3,)
dtype = float64
3. Configuração Experimental

O ambiente experimental é composto por:

Arrays receptores
North
South
East
West
Transmissores
TX1
TX2
TX3
TX4
Alvos

Foram realizados experimentos com:

ambiente vazio;
alvo passivo refletor;
pessoa utilizando retrorefletor;
trajetórias controladas.
4. Cenário Selecionado
Human Helmet Standing Center

O cenário inicialmente selecionado foi:

human-helmet-standing-center-1

A escolha foi motivada pelo fato de representar um caso quase estacionário, adequado para construção de uma referência experimental.

5. Caracterização Temporal e Espacial
5.1 Quantidade de snapshots

Número total de medições:

5041 snapshots

Duração total:

82.9 segundos

Taxa aproximada:

60.8 snapshots/s
5.2 Estabilidade da posição

Posição inicial:

[0.8298, 3.5223, -0.5050]

Posição final:

[0.8150, 3.5139, -0.5044]

Faixas observadas:

Eixo	Variação
X	~2 cm
Y	~7.5 cm
Z	~2 mm

Esses resultados confirmam que o alvo permaneceu praticamente estacionário durante a aquisição.

6. Estatísticas do CSI

O tensor completo analisado possui dimensão:

(5041, 4, 2, 4, 53)
Magnitude
Métrica	Valor
Média	99.42
Desvio padrão	28.51
Mínimo	0
Máximo	181
Fase
Métrica	Valor
Média	0.0067
Desvio padrão	1.8147
Intervalo	[-π, π]

Os resultados são compatíveis com um ambiente indoor real sujeito a multipercurso e pequenas flutuações temporais.

7. Análise de Subespaços
7.1 Construção da matriz de covariância

Foi construído um subconjunto contendo:

1000 snapshots

armazenado em:

standing_center_1000.npz

A matriz de covariância foi estimada utilizando:

1696 variáveis complexas
7.2 Espectro de autovalores

Os quatro maiores autovalores observados foram:

λ1 = 5.98×10⁶
λ2 = 4.74×10⁶
λ3 = 4.09×10⁶
λ4 = 2.96×10⁶

O quinto autovalor apresentou magnitude significativamente inferior:

λ5 = 2.34×10⁴

Resultando em:

λ4 / λ5 ≈ 126

Esse elevado eigengap evidencia uma forte separação entre os modos dominantes e o restante do espectro.

7.3 Dimensionalidade efetiva

A energia acumulada revelou:

Energia preservada	Componentes
90%	4
95%	4
99%	10

Apesar do espaço de observação possuir 1696 dimensões complexas, apenas quatro modos dominantes explicam mais de 95% da energia observada.

8. Estabilidade Temporal dos Subespaços

A estabilidade temporal foi investigada utilizando janelas consecutivas de 100 snapshots.

Os ângulos principais observados entre subespaços consecutivos ficaram tipicamente entre:

2° e 6°

Exemplos:

0→1 : 2.50°
6→7 : 2.09°
8→9 : 4.96°

Esses resultados indicam que o canal apresenta elevada estacionariedade no período analisado.

9. Avaliação de PCA

A preservação dos subespaços foi analisada após:

Redução dimensional.
Reconstrução dos dados.
Reestimação dos subespaços dominantes.

Resultados:

Variância	Componentes	Ângulo Máximo
90%	7	0.62°
95%	8	0.00045°
99%	19	0.00029°

O PCA preservou praticamente de forma perfeita o subespaço dominante.

10. Avaliação de Kernel PCA

Foi utilizado Kernel PCA com kernel RBF.

Foram avaliadas diferentes combinações de:

número de componentes;
parâmetro γ.
10.1 Reconstrução

Os erros médios observados foram:

PCA
MSE ≈ 53–495
KPCA
MSE ≈ 5313
10.2 Preservação dos subespaços

Dependendo dos hiperparâmetros utilizados, os ângulos observados variaram aproximadamente entre:

15°
e
89°

indicando elevada sensibilidade à escolha de γ.

11. Discussão

Os resultados mostram que o cenário Human Helmet Standing Center apresenta uma estrutura extremamente favorável para métodos baseados em subespaços.

Foram observados simultaneamente:

forte eigengap;
baixa dimensionalidade efetiva;
estabilidade temporal;
preservação quase perfeita por PCA.

Por outro lado, o KPCA apresentou forte dependência dos hiperparâmetros e desempenho significativamente inferior quando avaliado através da reconstrução no espaço original.

Entretanto, deve-se ressaltar que a operação de reconstrução utilizada pelo KPCA depende da solução aproximada do problema de pre-image, o que limita a interpretação dos resultados obtidos.

12. Conclusões

As principais conclusões desta etapa são:

O dataset ESPARGOS-0007 foi corretamente decodificado e caracterizado.
O cenário Human Helmet Standing Center apresenta comportamento altamente estacionário.
Apenas quatro modos dominantes explicam mais de 95% da energia observada.
Os subespaços dominantes permanecem estáveis ao longo do tempo.
O PCA preserva praticamente de forma perfeita os subespaços relevantes.
O KPCA mostrou elevada sensibilidade aos hiperparâmetros e forte degradação quando avaliado via reconstrução no espaço original.
A preservação de subespaços mostrou-se uma métrica mais informativa do que o erro de reconstrução isoladamente.
13. Trabalhos Futuros

As próximas etapas previstas incluem:

análise dos demais cenários do dataset;
implementação de MUSIC sobre os dados originais;
comparação MUSIC + PCA;
comparação MUSIC + KPCA;
avaliação da preservação dos subespaços de sinal e ruído;
estudo da separação entre cenários utilizando representações reduzidas;
investigação da relação entre preservação de subespaços e erro angular de estimação
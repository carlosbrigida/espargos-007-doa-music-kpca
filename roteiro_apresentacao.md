ROTEIRO COMPLETO DA APRESENTAÇÃO
Slide 1 – Title

Bom dia.

Nesta reunião vou apresentar o trabalho desenvolvido até o momento utilizando o dataset ESPARGOS 007.

O foco principal foi compreender a estrutura dos dados CSI, caracterizar o subespaço associado a esses sinais e investigar PCA e Kernel PCA como possíveis ferramentas para uma futura integração com o algoritmo MUSIC.

Slide 2 – Overview

A apresentação está organizada em sete partes.

Primeiro apresento o problema de pesquisa.

Depois a caracterização do dataset.

Em seguida os trabalhos relacionados.

A metodologia utilizada.

Os resultados obtidos.

E por fim as conclusões e próximos passos.

INTRODUÇÃO
Slide 3 – Introduction

O objetivo deste trabalho ainda não foi melhorar o MUSIC.

O objetivo foi entender os dados CSI do ESPARGOS e investigar se PCA e KPCA conseguem representar esses dados preservando estruturas importantes.

A motivação é que o MUSIC depende fortemente da separação entre subespaço de sinal e subespaço de ruído.

Portanto, antes de modificar o MUSIC, preciso entender os dados e os subespaços presentes neles.

Slide 4 – Problem Statement

A pergunta principal desta etapa é:

PCA e KPCA conseguem preservar estruturas relevantes dos dados CSI para uma futura aplicação com MUSIC?

Até o momento:

a caracterização dos dados foi concluída;
PCA e KPCA foram avaliadas de forma exploratória;
a integração com MUSIC ainda não foi implementada.
MUSIC
Slide 5 – Classical MUSIC Algorithm

O MUSIC trabalha a partir de uma matriz de snapshots.

Y=[x
1
	​

x
2
	​

...x
N
	​

]

Cada coluna representa um snapshot.

Ou seja, uma observação do sistema.

A partir dessa matriz calculamos a matriz de covariância:

R=
N
1
	​

YY
H

onde:

Y
H
 é a transposta conjugada;
N é o número de snapshots.

A matriz de covariância mede a correlação existente entre os sinais.

Depois realizamos a decomposição espectral:

R=EΛE
H

Os autovalores indicam quanta energia existe em cada direção.

Os autovetores associados aos maiores autovalores formam o subespaço de sinal.

Os demais formam o subespaço de ruído.

Slide 6 – MUSIC Pseudospectrum

A ideia central do MUSIC é:

a(θ)⊥E
n
	​


O vetor diretor a(θ) representa como um sinal chegaria ao array para um determinado ângulo.

Quando o ângulo é o correto, esse vetor torna-se ortogonal ao subespaço de ruído.

Isso leva ao pseudoespectro:

P
MUSIC
	​

(θ)=
a
H
(θ)E
n
	​

E
n
H
	​

a(θ)
1
	​


O denominador mede o alinhamento entre o vetor diretor e o subespaço de ruído.

Quando existe ortogonalidade:

o denominador se aproxima de zero;
o espectro cresce;
surge um pico.

Os picos correspondem às direções estimadas de chegada.

DATASET
Slide 7 – ESPARGOS Dataset

O dataset utilizado foi o ESPARGOS 007.

Ele contém medições reais de CSI em ambiente indoor.

Cada snapshot contém:

4 arrays receptores;
2 linhas de antenas;
4 colunas de antenas;
53 subportadoras OFDM.

Para os experimentos iniciais utilizei principalmente os cenários Standing Center e Circle.

O QUE É OFDM?

OFDM significa:

Orthogonal Frequency Division Multiplexing.

Em vez de transmitir toda a informação em uma única frequência, o sinal é dividido em várias subportadoras.

É como uma rodovia:

sem OFDM → uma única faixa;
com OFDM → várias faixas paralelas.

Isso torna a comunicação mais robusta.

O QUE É CSI?

CSI significa:

Channel State Information.

O CSI mede como o ambiente afeta o sinal transmitido.

Para cada subportadora temos:

H(f
k
	​

)=∣H(f
k
	​

)∣e
jϕ
k
	​


onde:

∣H(f
k
	​

)∣ é a magnitude;
ϕ
k
	​

 é a fase.

A magnitude indica quanto o sinal foi atenuado.

A fase indica quanto ele foi atrasado.

Slide 8 – Understanding the CSI Tensor

Cada snapshot é armazenado como:

H
i
	​

∈C
4×2×4×53

As dimensões representam:

4 arrays;
2 linhas;
4 colunas;
53 subportadoras.

Cada elemento do tensor representa:

A resposta complexa do canal para uma antena específica e uma subportadora específica.

Pergunta provável do Paim

O que existe dentro daquele tensor?

Resposta:

Existem 1696 coeficientes complexos de CSI.

Cada coeficiente representa a resposta do canal medida por uma antena específica para uma subportadora específica.

Slide 9 – OFDM and CSI Measurements

O sinal Wi-Fi é transmitido usando OFDM.

O canal afeta cada subportadora de forma diferente.

O CSI mede exatamente essa resposta.

Por isso cada snapshot contém informações distribuídas pelas 53 subportadoras.

Slide 10 – From CSI Tensor to Snapshot Vector

O tensor é transformado em vetor.

x
i
	​

=vec(H
i
	​

)

A operação vec apenas reorganiza os dados.

Nenhuma informação é descartada.

Como:

4×2×4×53=1696

cada snapshot passa a ser representado por:

x
i
	​

∈C
1696
Slide 11 – From Snapshot Vectors to Data Matrix

Empilhamos todos os snapshots:

X=
	​

x
1
T
	​

x
2
T
	​

⋮
x
N
T
	​

	​

	​


Cada linha é um snapshot.

Cada coluna é uma característica.

Como PCA e KPCA trabalham com dados reais:

X
r
	​

=[ℜ(X)ℑ(X)]

Separando parte real e imaginária.

Obtendo:

X
r
	​

∈R
N×3392
Slide 12 – Dataset Characterization

Antes de aplicar PCA ou KPCA procurei entender a estrutura dos dados.

O gráfico superior mostra os autovalores.

Poucos autovalores concentram a maior parte da energia.

O gráfico inferior mostra a variância acumulada.

Observamos que aproximadamente quatro componentes explicam cerca de 95% da variância.

Isso sugere a existência de um subespaço dominante de baixa dimensão.

TRABALHOS RELACIONADOS
Slide 13 – MUSIC

Schmidt propôs o MUSIC em 1986.

Foi um dos primeiros métodos de alta resolução para estimação de direção de chegada baseado em subespaços.

Esse será o algoritmo utilizado como referência.

Slide 14 – PCA e KPCA

PCA:

Pearson (1901)
Hotelling (1933)

KPCA:

Schölkopf (1998)

A PCA realiza redução linear de dimensionalidade.

A KPCA estende essa ideia para relações não lineares através de kernels.

Slide 15 – What is PCA?

A PCA procura direções que expliquem a maior parte da variância.

Fluxo:

X
r
	​

→Covariance→Eigenvectors→Principal Components

As componentes principais não possuem significado físico direto.

São apenas direções matemáticas que concentram a maior parte da informação.

Slide 16 – What is Kernel PCA?

A KPCA segue a mesma ideia.

x→ϕ(x)→PCA

Primeiro os dados são mapeados para um espaço de características.

Depois a PCA é aplicada nesse novo espaço.

Isso permite capturar estruturas não lineares.

METODOLOGIA
Slide 17 – Proposed Methodology

Primeiro:

caracterização do tensor CSI.

Depois:

construção dos snapshots.

Em seguida:

análise exploratória usando PCA e KPCA.

Por fim:

integração futura com MUSIC.

A hipótese inicial é aplicar KPCA após a construção dos snapshots.

Mas outras possibilidades ainda serão investigadas.

Slide 18 – Covariance Matrix Estimation

Comparo duas covariâncias.

Original:

R=
N
1
	​

X
H
X

Reconstruída:

R
^
=
N
1
	​

X
^
H
X
^

Depois comparo os subespaços através dos ângulos principais.

O objetivo é verificar se PCA ou KPCA preservam o subespaço dominante.

RESULTADOS
Slide 19 – Current Findings

Principais observações:

estrutura de baixa dimensão;
quatro componentes explicam aproximadamente 95% da variância;
PCA e KPCA avaliadas de forma exploratória;
integração com MUSIC ainda não realizada.
Slide 20 – PCA Results

Utilizando apenas 8 componentes:

ângulos principais praticamente nulos.

Isso significa:

o subespaço reconstruído é praticamente igual ao original.

Portanto:

a PCA preservou muito bem o subespaço dominante.

Slide 21 – Kernel PCA Results

A KPCA produziu ângulos maiores.

Isso significa que o subespaço obtido é diferente.

Não significa necessariamente que seja pior.

Significa apenas que a representação mudou.

Ainda precisamos verificar o efeito dessa mudança no MUSIC.

CONCLUSÕES
Slide 22 – Conclusions

Consegui:

caracterizar o dataset;
compreender a estrutura do tensor CSI;
construir a matriz de snapshots;
identificar um subespaço dominante de baixa dimensão;
avaliar PCA e KPCA como ferramentas exploratórias.
Slide 23 – Future Work

Próximos passos:

compreender completamente a implementação do Luís;
reproduzir os resultados de referência do MUSIC;
analisar a estimação da covariância;
investigar possíveis pontos de inserção da KPCA.

Objetivo final:

Verificar se a KPCA pode melhorar a estimação de direção de chegada baseada em MUSIC.

RESPOSTA MAIS IMPORTANTE DA APRESENTAÇÃO

Pergunta:

"Por que você aplicou PCA e KPCA se ainda não integrou ao MUSIC?"

Resposta:

"Antes de modificar o pipeline do MUSIC eu precisava entender a estrutura dos dados CSI e verificar se PCA e KPCA preservavam ou modificavam o subespaço dominante dos sinais. A integração com o MUSIC será a próxima etapa da pesquisa."
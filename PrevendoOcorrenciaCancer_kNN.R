# Prevendo a Ocorrência de Câncer utilizando o algortimo de Machine Learning kNN

# Configurando o diretório de trabalho
# Coloque entre aspas o diretório de trabalho que você está usando no seu computador
setwd("{SET_YOUR_HOME_DIRECTORY_HERE}")
getwd()

# Definição do Problema de Negócio: Previsão de Ocorrência de Câncer de Mama
# http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

## Etapa 1 - Coletando os Dados

# Os dados do câncer da mama incluem 569 observações de biópsias de câncer, 
# cada um com 32 características (variáveis). Uma característica é um número de 
# identificação (ID), outro é o diagnóstico de câncer, e 30 são medidas laboratoriais 
# numéricas. O diagnóstico é codificado como "M" para indicar maligno ou "B" para 
# indicar benigno.

# stringsAsFactors = FALSE -> Informar para não colocar as variaveis numéricas como Factor (Categórica)
dados <- read.csv("dataset.csv", stringsAsFactors = FALSE) 
str(dados)
View(dados)

# Sempre ficar atento a formato de dados numéricos. Por exemplo:
# A coluna texture_mean está com Percentual (12.39) e a smoothness_mean também (0.10 = 10%), porém com medidas diferentes;
# Deve-se transformar e padronizar a medida; 

# Coluna Diagnis é nossa coluna target

## Etapa 2 - Pré-Processamento

# Excluindo a coluna ID
# Independentemente do método de aprendizagem de máquina, deve sempre ser excluídas 
# variáveis de ID. Caso contrário, isso pode levar a resultados errados porque o ID 
# pode ser usado para unicamente "prever" cada exemplo. Por conseguinte, um modelo 
# que inclui um identificador pode sofrer de superajuste (overfitting), 
# e será muito difícil usá-lo para generalizar outros dados.
dados$id = NULL

# Ajustando o label da variável alvo
# apply -> FOr sobre todos registros
dados$diagnosis = sapply(dados$diagnosis, function(x){ifelse(x=='M', 'Maligno', 'Benigno')})

# Muitos classificadores requerem que as variáveis sejam do tipo Fator
table(dados$diagnosis) #Tabela de Contingência

# transformando a variavel target diagnosis em factor
dados$diagnosis <- factor(dados$diagnosis, levels = c("Benigno", "Maligno"), labels = c("Benigno", "Maligno"))
str(dados$diagnosis)

# Verificando a proporção
round(prop.table(table(dados$diagnosis)) * 100, digits = 1) 

# Medidas de Tendência Central
# Detectamos um problema de escala entre os dados, que então precisam ser normalizados
# O cálculo de distância feito pelo kNN é dependente das medidas de escala nos dados de entrada.
summary(dados[c("radius_mean", "area_mean", "smoothness_mean")])

# Criando um função de normalização para colocar os dados na mesma escala
normalizar <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# Testando a função de normalização - os resultados devem ser idênticos
# Dados estão em escava diferente e estão sendo padronizados para uma unidade de escala
# onde a representatividade seja igual para todos
normalizar(c(1, 2, 3, 4, 5))
normalizar(c(10, 20, 30, 40, 50))
normalizar(c(100, 200, 300, 400, 500))

# Normalizando os dados
# lappy -> loop das colunas 2:31 sob a função normalizar
dados_norm <- as.data.frame(lapply(dados[2:31], normalizar))
View(dados_norm)


## Etapa 3: Treinando o modelo com KNN

# Carregando o pacote library
# install.packages("class")
library(class)
?knn #Modelo de classificação onde "considera registros mais próximos" 

# Criando dados de treino e dados de teste
# A melhor forma de fazer é de forma aleatória, poderia ter sido utilizado o Random
dados_treino <- dados_norm[1:469, ] # Linha 1-469 todas Colunas
dados_teste <- dados_norm[470:569, ]

# Criando os labels para os dados de treino e de teste
# Labels representa a informação(coluna) que se deseja classificar (Target),
# Pode ser qualquer coisa, 0,1 M,F, TRUE,FALSE...
dados_treino_labels <- dados[1:469, 1]
dados_teste_labels <- dados[470:569, 1]
length(dados_treino_labels)
length(dados_teste_labels)

# Criando o modelo
modelo_knn_v1 <- knn(train = dados_treino, 
                     test = dados_teste,
                     cl = dados_treino_labels, # a Classificação que o algoritmo deve realizar
                     k = 21) # Variavel mais importante do modelo knn
#k representa que irá olhar para os 21 pontos de dados mais proximos de cada ponto de dado;

# A função knn() retorna um objeto do tipo fator com as previsões para cada exemplo no dataset de teste
summary(modelo_knn_v1)

## Etapa 4: Avaliando e Interpretando o Modelo

# Carregando o gmodels
library(gmodels)

# Criando uma tabela cruzada dos dados previstos x dados atuais
# Usaremos amostra com 100 observações: length(dados_teste_labels)
CrossTable(x = dados_teste_labels, y = modelo_knn_v1, prop.chisq = FALSE) #Não imprime a Qui-Quadrado
# No exemplo o modelo classificou 2 registros errados .

# Interpretando os Resultados
# A tabela cruzada mostra 4 possíveis valores, que representam os falso/verdadeiro positivo e negativo
# Temos duas colunas listando os labels originais nos dados observados
# Temos duas linhas listando os labels dos dados de teste

# Temos:
# Cenário 1: Célula Benigno (Observado) x Benigno (Previsto) - 61 casos - true positive 
# Cenário 2: Célula Maligno (Observado) x Benigno (Previsto) - 00 casos - false positive (o modelo errou)
# Cenário 3: Célula Benigno (Observado) x Maligno (Previsto) - 02 casos - false negative (o modelo errou)
# Cenário 4: Célula Maligno (Observado) x Maligno (Previsto) - 37 casos - true negative 

# Lendo a Confusion Matrix (Perspectiva de ter ou não a doença):

# True Negative  = nosso modelo previu que a pessoa NÃO tinha a doença e os dados mostraram que realmente a pessoa NÃO tinha a doença
# False Positive = nosso modelo previu que a pessoa tinha a doença e os dados mostraram que NÃO, a pessoa tinha a doença
# False Negative = nosso modelo previu que a pessoa NÃO tinha a doença e os dados mostraram que SIM, a pessoa tinha a doença
# True Positive = nosso modelo previu que a pessoa tinha a doença e os dados mostraram que SIM, a pessoa tinha a doença

# Falso Positivo - Erro Tipo I
# Falso Negativo - Erro Tipo II

# Taxa de acerto do Modelo: 98% (acertou 98 em 100)


## Etapa 5: Otimizando a Performance do Modelo

# Usando a função scale() para padronizar o z-score 
# Padronizar o z-score ajuda o algoritmo
?scale()
dados_z <- as.data.frame(scale(dados[-1])) # Removendo a primeira coluna
View(dados_z)

# Confirmando transformação realizada com sucesso
summary(dados_z$area_mean)

# Criando novos datasets de treino e de teste
dados_treino <- dados_z[1:469, ]
dados_teste <- dados_z[470:569, ]

dados_treino_labels <- dados[ 1: 469, 1] 
dados_teste_labels <- dados[ 470: 569, 1]

# Reclassificando
modelo_knn_v2 <- knn(train = dados_treino, 
                     test = dados_teste,
                     cl = dados_treino_labels, 
                     k = 21)

# Criando uma tabela cruzada dos dados previstos x dados atuais
CrossTable(x = dados_teste_labels, y = modelo_knn_v2, prop.chisq = FALSE)


# Ao analisar o CrossTable é possível identificar que o modelo_knn_v2 ficou pior que o modelo_knn_v1,
# ou seja, teve uma taxa de erro maior.

# Podemos concluir que sem realizar a padronização do z-score, o algoritmo teve uma previsão melhor.

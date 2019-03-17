# Construindo um Modelo com Algoritmo Support Vector Machine (SVM)


setwd("{SET_YOUR_HOME_DIRECTORY_HERE}")
getwd()

# SVM pode ser usado para modelos de regressão e modelos de regressão.
# SVM é usado para Dados Não Linearmente separaveis;

# Definindo a semente para resultados reproduzíveis
set.seed(40) 

# Prepara o dataset
dados <- read.csv("dataset.csv", stringsAsFactors = FALSE)
dados$id = NULL
# A criaçao do index está sendo feita de forma randomica sob uma distribuição uniforme
dados[,'index'] <- ifelse(runif(nrow(dados)) < 0.8,1,0) #Coluna Index será utilizada para dividir os dados em treino e teste
View(dados)

# Dados de treino e teste
trainset <- dados[dados$index==1,] #Pega os dados onde a coluna index == 1
testset <- dados[dados$index==0,]
View(trainset)

# Obter o índice 
# está varrendo os nomes das colunas atrás da coluna com nome "index" e salvando a posição da coluna na variavel
trainColNum <- grep('index', names(trainset)) 
trainColNum
# Remover o índice dos datasets
trainset <- trainset[,-trainColNum]
testset <- testset[,-trainColNum]

# Obter índice de coluna da variável target no conjunto de dados que começa com diag
typeColNum <- grep('diag',names(dados))

# Cria o modelo
# Nós ajustamos o kernel para radial, já que este conjunto de dados não tem um 
# plano linear que pode ser desenhado
library(e1071)
?svm
modelo_svm_v1 <- svm(diagnosis ~ ., 
                     data = trainset, 
                     type = 'C-classification', 
                     kernel = 'radial') #Base do Algoritmo SVM

# Previsões

# Previsões nos dados de treino
pred_train <- predict(modelo_svm_v1, trainset) 

# Percentual de previsões corretas com dataset de treino
# Comparando o valor do previsto pelo algoritmo com o valor real do dataset
mean(pred_train == trainset$diagnosis)  


# Previsões nos dados de teste
pred_test <- predict(modelo_svm_v1, testset) 

# Percentual de previsões corretas com dataset de teste
mean(pred_test == testset$diagnosis)  

# Confusion Matrix
table(pred_test, testset$diagnosis)
CrossTable(pred_test, testset$diagnosis)
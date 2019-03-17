# Construindo um Modelo com Algoritmo Random Forest
# Conjunto de arvores de decisão que juntos trabalham para fazer a previsão
# Algoritmo com alta taxa de acertividade para variaveis + relevantes


setwd("{SET_YOUR_HOME_DIRECTORY_HERE}")
getwd()

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

# Obter o índice 
# está varrendo os nomes das colunas atrás da coluna com nome "index" e salvando a posição da coluna na variavel
trainColNum <- grep('index', names(trainset)) 

# Remover o índice dos datasets
trainset <- trainset[,-trainColNum]
testset <- testset[,-trainColNum]

# Obter índice de coluna da variável target no conjunto de dados que começa com diag
typeColNum <- grep('diag',names(dados))

# Criando o modelo
library(rpart)
modelo_rf_v1 = rpart(diagnosis ~ ., data = trainset, control = rpart.control(cp = .0005)) 
# control = rpart.control(cp = .0005) -> NIvel das folhas

# Previsões nos dados de teste
tree_pred = predict(modelo_rf_v1, testset, type='class')
?predict
# Percentual de previsões corretas com dataset de teste
mean(tree_pred==testset$diagnosis) 

# Confusion Matrix
table(tree_pred, testset$diagnosis)
library(gmodels)
CrossTable(tree_pred, testset$diagnosis)

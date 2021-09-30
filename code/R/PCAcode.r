library(caret)
library(gridExtra)
library(dplyr)
library(randomForest)
library(Boruta)
library(neuralnet)
library(doParallel)
library(gbm)

metrics <- function(n.pred, model, train, test, y.train, y.test){
  n.pred = as.numeric(n.pred)
  y.train = as.numeric(y.train)
  y.test = as.numeric(y.test)
  
  pred.train = predict(model, newdata=train)
  pred.test = predict(model, newdata=test)
  pred.train[pred.train>1]=1
  pred.train[pred.train<0]=0
  pred.test[pred.test>1]=1
  pred.test[pred.test<0]=0
  
  sse.train = sum((pred.train-y.train)**2)
  sse.test = sum((pred.test-y.test)**2)
  ssto.train = sum((mean(y.train)-y.train)**2)
  ssto.test = sum((mean(y.test)-y.test)**2)
  n.train = length(y.train)
  n.test = length(y.test)
  
  mse.train = sse.train/n.train
  mse.test = sse.test/n.test
  r2.train = cor(y.train,pred.train)**2
  r2.test = cor(y.test,pred.test)**2
  
  return(as.numeric(c(mse.train, mse.test, r2.train, r2.test)))
}

data = read.csv("data.csv")
data = na.omit(data)
set.seed(99)

child.acc = data$Child_acc
adult.acc = data$Adult_acc
trmt = data$Treatment

n.sub = dim(data)[1]
n.var = dim(data)[2]
pp = preProcess(data[2:(n.var-3)], method = c("BoxCox"), na.remove=T)
ppx = as.data.frame(scale(predict(pp, data[2:(n.var-3)])))

ppdata = as.data.frame(ppx)
ppdata$Child_acc = data$Child_acc
ppdata$Adult_acc = data$Adult_acc

train = sample(1:nrow(ppdata), nrow(ppdata)*.7)

pptrain = ppdata[train,]
pptest = ppdata[-train,]

# supervised model (multilinear regression)
child.lm = lm(Child_acc~.-Adult_acc, data=pptrain)
child.lm.measures = metrics(30, child.lm, pptrain, pptest, pptrain$Child_acc, pptest$Child_acc)
adult.lm = lm(Adult_acc~.-Child_acc, data=pptrain)
adult.lm.measures = metrics(30, adult.lm, pptrain, pptest, pptrain$Adult_acc, pptest$Adult_acc)


# supervised model (artificial neural network)
fitControl <- trainControl(method = "repeatedcv",
                           number = 5,
                           repeats = 10)

nn.grid <- expand.grid(layer1 = (1:10),
                       layer2 = (0:10),
                       layer3 = (0:10))

cl <- makePSOCKcluster(12)
registerDoParallel(cl)
child.nn.tune <- train(Child_acc~.-Adult_acc, data=pptrain,
                     method = "neuralnet", 
                     trControl = fitControl,
                     tuneGrid = nn.grid,
                     act.fct = "logistic",
                     stepmax = 1e+05,
                     algorithm = "rprop+",
                     linear.output = TRUE)
stopCluster(cl)
cl1 = as.integer(child.nn.tune$bestTune[1])
cl2 = as.integer(child.nn.tune$bestTune[2])
cl3 = as.integer(child.nn.tune$bestTune[3])

child.nn = neuralnet(Child_acc~.-Adult_acc, data=pptrain,
                     hidden = c(cl1,cl2,cl3),
                     stepmax = 1e+05,
                     act.fct="logistic", 
                     algorithm = "rprop+",
                     linear.output = TRUE,
                     rep=10)

child.nn.measures = metrics(30, child.nn, pptrain, pptest, pptrain$Child_acc, pptest$Child_acc)
#measures <- function(n.pred, pred.train, pred.test, y.train, y.test)

cl <- makePSOCKcluster(12)
registerDoParallel(cl)
adult.nn.tune <- train(Adult_acc~.-Child_acc, data=pptrain,
                       method = "neuralnet", 
                       trControl = fitControl,
                       tuneGrid = nn.grid,
                       act.fct = "logistic",
                       stepmax = 1e+05,
                       algorithm = "rprop+",
                       linear.output = TRUE)
stopCluster(cl)
al1 = as.integer(adult.nn.tune$bestTune[1])
al2 = as.integer(adult.nn.tune$bestTune[2])
al3 = as.integer(adult.nn.tune$bestTune[3])

adult.nn = neuralnet(Adult_acc~.-Child_acc, data=pptrain,
                     hidden = c(cl1,cl2,cl3),
                     stepmax = 1e+05,
                     act.fct="logistic", 
                     algorithm = "rprop+",
                     linear.output = TRUE,
                     rep=10)

adult.pred.test = predict(adult.nn,newdata=pptest)
adult.pred.train = predict(adult.nn,newdata=pptrain)
adult.nn.measures = metrics(30, adult.nn, pptrain, pptest, pptrain$Adult_acc, pptest$Adult_acc)

# unsupervised model (pca)


data.dist=dist(t(ppx), method = "euclidean")
par(mfrow=c(1,1))
hc.out=hclust(data.dist, method = "complete")
plot(hc.out)
rect.hclust(hc.out, k = 10, border = "blue") 
hc.clusters=cutree(hc.out, 10) 


# feature selection
child.boruta <- Boruta(Child_acc~.-Adult_acc, data=ppdata,
                       doTrace = 0,
                       pValue = 0.05,
                       holdHistory = TRUE,
                       maxRuns = 10000)
plot(child.boruta, las = 2, cex.axis = 0.6, main="Var Imp")
child.boruta.var <- getSelectedAttributes(child.boruta, withTentative = TRUE)
print(child.boruta.var) 

adult.boruta <- Boruta(Adult_acc~.-Child_acc, data=ppdata,
                     doTrace = 0,
                     pValue = 0.05,
                     holdHistory = TRUE,
                     maxRuns = 10000)
plot(adult.boruta, las = 2, cex.axis = 0.6, main="Var Imp")
adult.boruta.var <- getSelectedAttributes(adult.boruta, withTentative = TRUE)
print(adult.boruta.var) 

save.image(file = "processed.RData")


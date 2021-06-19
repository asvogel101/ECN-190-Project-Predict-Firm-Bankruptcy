library(dplyr)
library(tidyverse)
library(ggplot2)
library(boot)
library(gamlr)
library(glmnet)
library(caret)
library(stargazer)
set.seed(100)
options(scipen=100)
library(broom)
data<-read_csv('data.csv')

### PRINCIPAL COMPONENT ANALYSIS ###




#logmod<-cv.glmnet(x=Xtrain,y=train$`Bankrupt?` , data=train, family='binomial',k=5)
#logmod.predict<-predict(logmod, newx=Xtest, type='response')
#logmod.predict<-ifelse(logmod.predict>0.5,1,0)
#tablelogmod<-table(logmod.predict, test$`Bankrupt?`)
#logmod.mse.min<-logmod$cvm[logmod$lambda==logmod$lambda.min]
#coef(logmod, s=0)

#logmod_pred<-predict(logmod,newx=Xtest, s=0, type='response')

# Calculate oos MSE
#ridge_oos<-mean((test$`Bankrupt?`-logmod_pred^2))

#ridge_oos

### CLASSIFICATION AND REGRESSION TREE (CART) ###
set.seed(100)
data<-read.csv('data.csv')
data<-data[ , which(apply(data, 2, var) != 0)]

train<-sample(dim(data)[1], dim(data[1])/5)
train.data<-data[-train,]
test.data<-data[train,]
X.train<-data[-train, -1]
X.test<-data[train, -1]
Y.train<-as.factor(data[-train, 1])
Y.test<-as.factor(data[train, 1])
p<-dim(X.train)[2]
p.2<-p/2
p.sq<-sqrt(p)

tree.bankrupt<-tree(Bankrupt. ~ ., data=train.data)

plot(tree.bankrupt)
text(tree.bankrupt, cex=1/2, pretty=0)

pred.bankrupt<-predict(tree.bankrupt, test.data)
TestPredict<-ifelse(pred.bankrupt>.5, 1, 0)
table<-table(TestPredict, test.data$`Bankrupt.`)
colnames(table)<-c('Not Bankrupt', 'Bankrupt')
rownames(table)<-c('Predicted Not Bankrupt', 'Predicted Bankrupt')
table
## Cross validated tree ## 
set.seed(100)
rf.p.cv<-cv.tree(tree.bankrupt, FUN=prune.tree)
plot(rf.p.cv$size, rf.p.cv$dev, type='b', xlab='size', ylab='deviance')
plot(rf.p.cv$k, rf.p.cv$dev, type='b')
plot(rf.p.cv)

rf.p.prune<-prune.tree(tree.bankrupt,best=4)
plot(rf.p.prune)
text(rf.p.prune, pretty=0, cex=0.5)
pred.bankrupt<-predict(rf.p.prune, test.data)
TestPredict<-ifelse(pred.bankrupt>.5, 1, 0)
table<-table(TestPredict, test.data$`Bankrupt.`)
colnames(table)<-c('Not Bankrupt', 'Bankrupt')
rownames(table)<-c('Predicted Not Bankrupt', 'Predicted Bankrupt')
table

MSE<-mean((data.test$Bankrupt.-TestPredict)^2)
### RANDOM FOREST ###


rf.p<-randomForest(X.train, Y.train, 
                   xtest=X.test, ytest=Y.test, 
                   ntree=500, mtry=p.sq)
plot(rf.p)
################################################
## plain old logistic regression (no lasso) ######
#################################################
set.seed(100)
model_full<-glm(`Bankrupt.`~., family='binomial', data=train.data)
cv_model_full<-cv.glm(train.data, model_full, K=10)
MSE_model_full<-cv_model_full$delta
predict_model_full<-predict(model_full, newdata=test.data,type='response',s=0)
TestPredict<-ifelse(predict_model_full>.5, 1, 0)
table<-table(TestPredict, test.data$`Bankrupt.`)
colnames(table)<-c('Not Bankrupt', 'Bankrupt')
rownames(table)<-c('Predicted Not Bankrupt', 'Predicted Bankrupt')
table

## plain old logistic regression with principal components (no lasso) ##

set.seed(100)
pr<-prcomp(train.data[,-1], scale=TRUE)
pr.test<-prcomp(test.data[,-1], scale=T)
data.new<-cbind(pr$x, train.data$Bankrupt.)
test.data.new<-cbind(pr.test$x, test.data$Bankrupt.)
model_full<-glm(`V95`~., family='binomial', data=data.frame(data.new))
cv_model_full<-cv.glm(data=data.frame(test.data.new),model_full, K=10)
MSE_model_full<-cv_model_full$delta
predict_model_full<-predict(model_full, newdata=data.frame(test.data.new),
                            type='response', s=0)
TestPredict<-ifelse(predict_model_full>.5, 1, 0)
table<-table(TestPredict, test.data$`Bankrupt.`)
colnames(table)<-c('Not Bankrupt', 'Bankrupt')
rownames(table)<-c('Predicted Not Bankrupt', 'Predicted Bankrupt')
table

####### LASSO WITH AND WITHOUT PCA ############################
###############################################################
###############################################################
set.seed(100)
data<-data[ , which(apply(data, 2, var) != 0)]

n=nrow(data)
k=5
# vector of fold memberships #
foldid<-rep(1:k, each=ceiling(n/k))[sample(1:n)]
# select training and test subsample (training data will be all except fold k=1) #
train<-data[which(foldid!=1),]
test<-data[-which(foldid!=1),]

Xtrain=model.matrix(`Bankrupt.`~.,data=train)
Xtrain<-Xtrain[ , which(apply(Xtrain, 2, var) != 0)]
Xtest=model.matrix(`Bankrupt.`~.,data=test)
Xtest<-Xtest[ , which(apply(Xtest, 2, var) != 0)]

pr<-prcomp(Xtrain, scale=TRUE)
smoothScatter(pr$x[,1:2],col=c(rep(2,20), rep(5,20)), nbin=1000)
plot(summary(pr)$importance[3,], ylab='Cumulative Proportion of Variance Explained', xlab='Number of Components', type='b') 


### CAN USE CROSS VALIDATION LASSO TO CHOOSE K NUMBER OF COMPONENTS ###

cvlassoPCR<-cv.gamlr(x=pr$x, y=train[,1], nfold=10, family='binomial')
plot(cvlassoPCR,    main="Lasso on Principal Components",       ylim=c(0.15, 0.3), xlab="log lambda", df=FALSE) +
  abline(v=log(cvlassoPCR$lambda.min), col='red')
Xtestprcomp<-prcomp(Xtest,scale=T)
lasso_pred<-predict(cvlassoPCR, newdata=Xtestprcomp$x, scale=TRUE,select='min',type='response')
### OOS R2 & TABLE###
min.dev<-cvlassoPCR$cvm
lambda<-cvlassoPCR$lambda.min

mean((test$`Bankrupt`-lasso_pred)^2)

coef(cvlassoPCR, select="min")

TestPredictPCR<-ifelse(lasso_pred>.5,1,0)

tablePCR<-table(TestPredictPCR, test$`Bankrupt.`)
colnames(tablePCR)<-c('Not Bankrupt', 'Bankrupt')
rownames(tablePCR)<-c('Predicted Not Bankrupt', 'Predicted Bankrupt    ')
tablePCR


### CAN USE CROSS VALIDATION LASSO TO CHOOSE I REAL PREDICTORS ###
set.seed(100)
cvlasso<-cv.gamlr(x=Xtrain, y=train[,1], nfold=5, family='binomial')
plot(cvlasso, main='Lasso on Real Variables', df=FALSE, ylim=c(0.15,0.3)) + abline(v=log(cvlasso$lambda.min), col='red')

lasso_pred2<-predict(cvlasso, newdata=Xtest, scale=TRUE, select='min',type='response')

min.dev<-cvlasso$cvm
mean((test$`Bankrupt.`-lasso_pred2)^2)
coefficients(cvlasso, select='min')

## Predicted ###
TestPredict<-ifelse(lasso_pred2>.5, 1, 0)

table<-table(TestPredict, test$`Bankrupt.`)
colnames(table)<-c('Not Bankrupt', 'Bankrupt')
rownames(table)<-c('Predicted Not Bankrupt', 'Predicted Bankrupt')


##plot
ggplot(data, aes(x=as.factor(data$Bankrupt.), y=data$ROA.A..before.interest.and...after.tax)) + 
  geom_boxplot() + ylab('Return on Assets (Before Interest, After Tax)')

ggplot(data, aes(x=data$ROA.A..before.interest.and...after.tax,
       y=data$ROA.C..before.interest.and.depreciation.before.interest)) + geom_point() +
  xlab('Return on Assets Before Interest and After Tax') + 
  ylab('Return on Assets Before Interest and Depreciation Before Interest')
    

       
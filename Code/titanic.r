#Kaggle titanic competition
library(glmnet)       #for cross-validated logistic reg
library(rpart)        #simple trees
library(ggplot2)
library(lmtest)       #for resettest Ramsey RESET functional specification test
library(MASS)         #for rlm - robust regression
#library(e1071)        #svm, not used, use kernlab for caret tuning
library(kernlab)      #svm for caret
library(randomForest)
library(neuralnet)
library(mgcv)         #for gam
library(caret)

train <- read.csv("DataRaw/train.csv")
test  <- read.csv("DataRaw/test.csv")

############## Exploration #########

sum(train$Survived==1)/nrow(train)

# Pclass
(tablePclass <- table(train$Survived,train$Pclass,dnn=c("Survived","Class")))
tablePclass[2,]/colSums(tablePclass) #survival rates per class
summary(tablePclass)

# Sex
(tableSex <- table(train$Survived,train$Sex,dnn=c("Survived","Sex")))
tableSex[2,]/colSums(tableSex) #survival rates per sex
summary(tableSex)

# SibSp
(tableSibSp <- table(train$Survived,train$SibSp,dnn=c("Survived","Sibling")))
tableSibSp[2,]/colSums(tableSibSp) #survival rates per sibling
summary(tableSibSp)

#Add feature for Sibling - 0 and 1 only seem relevant, others too few samples
sibnames <- as.numeric(names(table(train$SibSp)))
for (i in sibnames) {
  varsib <- paste0("Sib",i)
  train[,varsib] <- as.numeric(train$SibSp==i)
  test[,varsib] <- as.numeric(test$SibSp==i)
}
train$SibGt1 <- as.numeric(train$SibSp>1) #add single var for >1 siblings
test$SibGt1 <- as.numeric(test$SibSp>1)
(tableSibSpGt <- table(train$Survived,train$SibGt1,dnn=c("Survived","Sibling")))
tableSibSpGt[2,]/colSums(tableSibSpGt) #survival rates per sibling
summary(tableSibSpGt)

# Parch
(tableParch <- table(train$Survived,train$Parch,dnn=c("Survived","ParCh")))
tableParch[2,]/colSums(tableParch) #survival rates per parents/children #
summary(tableParch)
train$ParchGt1 <- as.numeric(train$Parch>0) #add single var for >1 parch
test$ParchGt1  <- as.numeric(test$Parch>0)
(tableParchGt1 <- table(train$Survived,train$ParchGt1,dnn=c("Survived","ParChGt1")))
tableParchGt1[2,]/colSums(tableParchGt1) #survival rates per parents/children #
summary(tableParchGt1)

ggplot(train,aes(x=Age,y=Parch))+geom_point()+geom_smooth()
ggplot(train,aes(x=Age,y=SibSp))+geom_point()+geom_smooth()
ggplot(train,aes(x=Age,y=Sex))+geom_point()+geom_smooth()
summary(train[grep("Mrs",train$Name),"Age"])
summary(train[grep("Mr",train$Name),"Age"])
summary(train[grep("Master",train$Name),"Age"])
summary(train[grep("Miss",train$Name),"Age"])

#age missing data - set to average
AvgAgeTrain <- mean(train$Age,na.rm=T)
# train[is.na(train$Age),"Age"] <- AvgAgeTrain
# test[is.na(test$Age),"Age"] <- AvgAgeTrain

######   imputation based on bagging in caret      ############
pp <- preProcess(train[,c("Age","SibSp","Parch")],method=c("bagImpute"))
t2 <- predict(pp,train[,c("Age","SibSp","Parch")])
train$Age <- t2$Age
pp <- preProcess(test[,c("Age","SibSp","Parch")],method=c("bagImpute"))
t2 <- predict(pp,test[,c("Age","SibSp","Parch")])
test$Age <- t2$Age

##############  simple imputation based on common sense ######
# train[((is.na(train$Age))&(train$Parch==0)),"Age"] <- AvgAgeTrain
# train[((is.na(train$Age))&(train$SibSp>1)),"Age"] <- 5
# train[is.na(train$Age),"Age"] <- 10
# test[((is.na(test$Age))&(test$Parch==0)),"Age"] <- AvgAgeTrain
# test[((is.na(test$Age))&(test$SibSp>1)),"Age"] <- 5
# test[is.na(test$Age),"Age"] <- 10
train$LogAge <- log(train$Age)
test$LogAge  <- log(test$Age)

#Embarked
(tableEmb <- table(train$Survived,train$Embarked,dnn=c("Survived","Embarked")))
tableEmb[2,]/colSums(tableEmb) #survival rates per embarcation port
summary(tableEmb)
#embarkation missing data - not significant based on linear regression R2
train[!(train$Embarked %in% c("C","Q","S")),]
train$EmbC <- as.numeric(train$Embarked=="C")
train$EmbQ <- as.numeric(train$Embarked=="Q")
train$EmbS <- as.numeric(train$Embarked=="S")
test$EmbC <- as.numeric(test$Embarked=="C")
test$EmbQ <- as.numeric(test$Embarked=="Q")
test$EmbS <- as.numeric(test$Embarked=="S")

#scatterplot
pairs(~Survived+Sex+Pclass+Age+SibSp+Embarked+Fare,train)

#Add feature for cabin letter
for (i in LETTERS) {
  if (length(grep(i,train$Cabin)>0)) {
    print(i)
    varcab <- paste0("Cabin",i)
    train[,varcab] <- 0
    train[grep(i,train$Cabin),varcab] <- 1
    test[,varcab] <- 0
    test[grep(i,test$Cabin),varcab] <- 1
  }
}

train$CabinDigits <- 0
existdigit <- grep("[0-9]",train$Cabin)
train$CabinDigits[existdigit] <- 1
(tableDig <- table(train$Survived,train$CabinDigits,dnn=c("Survived","CabinDigit")))
tableDig[2,]/colSums(tableDig) #survival rates per embarcation port
summary(tableDig)
fdig <- function(x) {
  x <- strsplit(as.character(x),split=" ") #only first number
  x <- unlist(x)
  x <- x[grep("[0-9]",x)[1]]
  rmdig <- regmatches(x,gregexpr("[[:digit:]]",x))
  return(as.numeric(paste(unlist(rmdig),collapse="")))
} 
train$CabinNum <- 0
train$CabinNum[train$CabinDigits==1] <- 
  sapply(train$Cabin[train$CabinDigits==1],fdig)
test$CabinDigits <- 0
existdigit <- grep("[0-9]",test$Cabin)
test$CabinDigits[existdigit] <- 1
test$CabinNum <- 0
test$CabinNum[test$CabinDigits==1] <- 
  sapply(test$Cabin[test$CabinDigits==1],fdig)


#for neural network need number value instead of char
train$Male <- as.numeric(train$Sex=="male")
test$Male  <- as.numeric(test$Sex=="male")

#clean NAs from Fare
train$Fare[which(is.na(train$Fare))] <- 0
test$Fare[which(is.na(test$Fare))] <- 0

train$PC1 <- as.numeric(train$Pclass==1)
train$PC2 <- as.numeric(train$Pclass==2)
train$PC3 <- as.numeric(train$Pclass==3)
test$PC1  <- as.numeric(test$Pclass==1)
test$PC2  <- as.numeric(test$Pclass==2)
test$PC3  <- as.numeric(test$Pclass==3)

##### Modelling #############

caseswitch <- 14 #choose model to create predict.csv file

switch(caseswitch,
       
       {
         #1 - Naive predict - 61% on train
         naive_predict <- round(sum(train$Survived)/dim(train)[1])
         pred.train <- rep(naive_predict,nrow(train))
         pred.dat <- data.frame(PassengerId=test$PassengerId,Survived=naive_predict)
       },
       
       {
         #2 - Simple linear regression based on Pclass predict -67% on train
         lreg1 <- lm(Survived~Pclass,train)
         pred.train <- round(predict(lreg1))
         p1 <- round(predict(lreg1,newdata=test))
         pred.dat <- data.frame(PassengerId=test$PassengerId,Survived=p1)
       },
       
       {
         #3 - Simple linear regression based on Pclass+Sex predict - 78% on train
         lreg2 <- lm(Survived~Pclass+Sex,train)
         pred.train <- round(predict(lreg2))
         p1 <- round(predict(lreg2,newdata=test))
         pred.dat <- data.frame(PassengerId=test$PassengerId,Survived=p1)
       },
       
       {
         #4 - Simple linear regression based on Pclass+Sex predict - 79% on train
         lreg3 <- lm(Survived~Age+Sex+Pclass+SibSp,train)
         pred.train <- round(predict(lreg3))
         p1 <- round(predict(lreg3,newdata=test))
         pred.dat <- data.frame(PassengerId=test$PassengerId,Survived=p1)
       },
       
       {
         #5 - Logit with glm - 82% on train, 77% on test - BEST
         logit1 <- step(glm(data=train,Survived~LogAge+Male+PC1+PC2+CabinD+CabinE+
                         Sib0+Sib1+EmbS,family=binomial(link="logit")))
         resp.train <- predict(logit1,type="response")
         print(ggplot(data.frame(survive=as.factor(train$Survived),
                           prediction=resp.train),
                aes(x=prediction,color=survive,linetype=survive))+geom_density())
         cutoff_search <- seq(0.2,0.9,0.01)
         c <- rep(0,length(cutoff_search))
         for (i in seq_along(cutoff_search)) {
           cutoff <- cutoff_search[i]
           pred.train <- ifelse(resp.train>cutoff,1,0)
           contingency <- prop.table(table(pred=pred.train,train=train$Survived))
           c[i] <- sum(diag(contingency))
         }
         cutoff <- cutoff_search[which.max(c)]
         pred.train <- ifelse(resp.train>cutoff,1,0)
         (contingency <- prop.table(table(pred=pred.train,train=train$Survived)))
         sum(diag(contingency))
         
         p1 <- ifelse(predict(logit1,newdata=test,type="response")>cutoff,1,0)
         pred.dat <- data.frame(PassengerId=test$PassengerId,Survived=p1)
       },
       
       {
         #6 - Logit with glmnet - 80% on train
         x <- data.matrix(train[,c("Sex","Pclass","Age","SibSp","Embarked")])
         y <- train$Survived
         cvfit <- cv.glmnet(x,y,family="binomial",alpha=0)
         #use 1 standard deviation from minimum lambda for regularization
         pred.train <- predict(cvfit,newx=x,type="class",s="lambda.1se")
         x <- data.matrix(test[,c("Sex","Pclass","Age","SibSp","Embarked")])
         p1 <- as.numeric(predict(cvfit,newx=x,type="class",s="lambda.1se"))
         pred.dat <- data.frame(PassengerId=test$PassengerId,Survived=p1)
       },
       
       {
         #7 - Simple linear regression with Cabin info - 80% on train
         train7 <- train[,grep("Cabin",names(train))[-1]]
         train7 <- cbind(train7,train[,c("Survived","Age","Sex","Pclass","SibSp")])
         lreg7 <- lm(Survived~.,train7)
         pred.train <- round(predict(lreg7))
         test7 <- test[,grep("Cabin",names(test))[-1]]
         test7 <- cbind(test7,test[,c("Age","Sex","Pclass","SibSp")])
         p1 <- round(predict(lreg7,newdata=test))
         pred.dat <- data.frame(PassengerId=test$PassengerId,Survived=p1)
       },
       
       {
         #8 - Simple linear regression with significant Cabin info - 81% on train
         train8 <- train[,c("CabinD","CabinE","CabinF")]
         train8 <- cbind(train8,train[,c("Survived","Age","Sex","Pclass","SibSp")])
         lreg8 <- lm(Survived~.,train8)
         pred.train <- round(predict(lreg8))
         test8 <- test[,c("CabinD","CabinE","CabinF")]
         test8 <- cbind(test8,test[,c("Age","Sex","Pclass","SibSp")])
         p1 <- round(predict(lreg8,newdata=test))
         pred.dat <- data.frame(PassengerId=test$PassengerId,Survived=p1)
       },
       
       {
         #9 - Logit with glmnet + cabin info - 81% on train
         x <- data.matrix(train[,c("Sex","Pclass","Age","SibSp","Embarked",
                                   "CabinD","CabinE","CabinF")])
         y <- train$Survived
         cvfit <- cv.glmnet(x,y,family="binomial",alpha=0,type.measure = "class")
         #use 1 standard deviation from minimum lambda for regularization
         pred.train <- predict(cvfit,newx=x,type="class",s="lambda.1se")
         x <- data.matrix(test[,c("Sex","Pclass","Age","SibSp","Embarked",
                                  "CabinD","CabinE","CabinF")])
         p1 <- as.numeric(predict(cvfit,newx=x,type="class",s="lambda.1se"))
         pred.dat <- data.frame(PassengerId=test$PassengerId,Survived=p1)
       },
       
       {
         #10 - Simple linear regression with sig Cabin Info and Sibling - 80% on train
         vars <- c("Sex","Pclass","LogAge","CabinD","CabinE","CabinF",
                   "Sib0","Sib1","ParchGt1","EmbS","CabinDigits","CabinNum")
         f <- paste(vars,collapse ="+")
         train10 <- train[,c(vars,"Survived")]
         lreg10 <- lm(as.formula(paste0("Survived~",f,"+Sex:Pclass+Sex:LogAge")),train10)
         pred.train <- round(predict(lreg10))
         
         #RESET test significant, functional misspecification for simple reg
         resettest(lreg10,type="fitted")
         #misspecification not result of powers of regressors, not significant
         resettest(lreg10,type="regressor")
         
         #residuals analyze
         print(ggplot(data=data.frame(residuals=lreg10$residuals,Age=train$Age),
                aes(x=Age,y=residuals))+geom_point()+geom_smooth())
         print(ggplot(data=data.frame(residuals=lreg10$residuals,Fare=train$Fare),
                aes(x=Fare,y=residuals))+geom_point()+geom_smooth())
         
         p1 <- round(predict(lreg10,newdata=test))
         pred.dat <- data.frame(PassengerId=test$PassengerId,Survived=p1)
       },
       
       {
         #11 - Tree partitioning with sig Cabin Info and Sibling - 82% on train
         treemod1 <- rpart(Survived~Sex+Pclass+Age+CabinD+CabinE+CabinF+
                             Sib0+Sib1+SibGt1+ParchGt1,data=train,method="class")
         pred.train <- round(predict(treemod1)[,2])
         p1 <- round(predict(treemod1,newdata=test)[,2])
         pred.dat <- data.frame(PassengerId=test$PassengerId,Survived=p1)
       },
       
       {
         #12 - Robust linear regression with sig Cabin Info and Sibling - 72% on train
         #not giving better results - outliers not a problem
         train12 <- train[,c("CabinD","CabinE","CabinF",
                             "Sib0","Sib1","ParchGt1")]
         train12 <- cbind(train12,train[,c("Survived","Age","Sex","Pclass")])
         lreg12 <- rlm(Survived~.,train12,maxit=30,method="MM")
         pred.train <- ifelse(predict(lreg12)>0.5,1,0)
         test12 <- test[,c("CabinD","CabinE","CabinF",
                           "Sib0","Sib1","ParchGt1")]
         test12 <- cbind(test12,test[,c("Age","Sex","Pclass")])
         p1 <- ifelse(predict(lreg12,newdata=test)>0.5,1,0)
         pred.dat <- data.frame(PassengerId=test$PassengerId,Survived=p1)
       },
       
       {
         #13 - Logit with glmnet + cabin info + Sibling - 81% on train
         vars <- c("Sex","Pclass","LogAge","CabinD","CabinE","CabinF",
                   "Sib0","Sib1","SibGt1","ParchGt1")
         x <- data.matrix(train[,vars])
         y <- train$Survived
         cvfit2 <- cv.glmnet(x,y,family="binomial",alpha=0) #,type.measure = "class")
         #use 1 standard deviation from minimum lambda for regularization
         pred.train <- predict(cvfit2,newx=x,type="class",s="lambda.1se")
         x <- data.matrix(test[,vars])
         p1 <- as.numeric(predict(cvfit2,newx=x,type="class",s="lambda.1se"))
         pred.dat <- data.frame(PassengerId=test$PassengerId,Survived=p1)
       },
       
       {
         #14 - svm - 84% on train, 76% on test
         objControl <- trainControl(method='repeatedcv',number=5,repeats=10)
         svm_traingrid <- expand.grid(sigma=seq(0.01,0.07,0.01),C=seq(3,5,0.5))
         svmfit1 <- train(as.factor(Survived)~LogAge+Male+PC1+PC2+PC3+
                            CabinD+CabinE+CabinF+CabinNum+EmbC+EmbS+
                            Sib0+Sib1+SibGt1+ParchGt1+Fare,data=train,
                          method="svmRadial",trControl=objControl,
                          tuneGrid=svm_traingrid)
         print(plot(svmfit1))
         #svmfit1 <- ksvm(as.factor(Survived)~Sex+Pclass+Age+CabinD+CabinE+CabinF+
         #              Sib0+Sib1+SibGt1+ParchGt1,data=train,kernel='rbfdot')
         pred.train <- predict(svmfit1,newdata=train)
         p1 <- predict(svmfit1,newdata=test)
         pred.dat <- data.frame(PassengerId=test$PassengerId,Survived=p1)
       },
       
       {
         #15 - Random forest with sig Cabin Info and Sibling - 82% on train
         vars <- c("Sex","PC1","PC2","Age","CabinD","CabinE","CabinF","CabinNum",
                         "Sib0","Sib1","ParchGt1","Fare","EmbC","EmbQ","EmbS")
         rforest1 <- randomForest(x=train[,vars],y=as.factor(train$Survived),
                                  importance=T)
         pred.train <- predict(rforest1,type="class")
         p1 <- predict(rforest1,newdata=test[,vars],type="class")
         pred.dat <- data.frame(PassengerId=test$PassengerId,Survived=p1)
       },
       
       {
         #16 - Neural net with sig Cabin Info and Sibling - 85% on train
         vars <- c("Pclass","Age","CabinD","CabinE","CabinF",
                   "Sib0","Sib1","SibGt1","ParchGt1","Male")
         f <- paste("Survived",paste(vars,collapse ="+"),sep="~")
         nnfit1 <- neuralnet(f,data=train,hidden=6, linear.output = F)
         resp.train <- compute(nnfit1,train[,vars])$net.result
         print(ggplot(data.frame(survive=as.factor(train$Survived),
                                 prediction=resp.train),
                      aes(x=prediction,color=survive,linetype=survive))+geom_density())
         cutoff <- 0.5 #based on density graph
         pred.train <- ifelse(resp.train>cutoff,1,0)
         p1 <- ifelse(compute(nnfit1,test[,vars])$net.result>cutoff,1,0)
         pred.dat <- data.frame(PassengerId=test$PassengerId,Survived=p1)
         
         #debug
         print(nnfit1)
         plot(nnfit1)
       },
       
       {
         #17 - GAM - 82% on train
         vars <- c("Male","PC1","PC2","s(LogAge)","CabinD","CabinE","CabinF",
                   "Sib0","Sib1","ParchGt1","s(Fare)","s(CabinNum)",
                   "Male:PC1","Male:LogAge","EmbS")
         f <- paste("Survived",paste(vars,collapse ="+"),sep="~")
         gamfit1 <- gam(as.formula(f),data=train,family=binomial(link="logit"))
         pred.train <- round(predict(gamfit1,type="response"))
         p1 <- round(predict(gamfit1,newdata=test,type="response"))
         pred.dat <- data.frame(PassengerId=test$PassengerId,Survived=p1)
       }
       
)


### Statistics and write prediction file ####

(contingency <- prop.table(table(pred=pred.train,train=train$Survived)))
sum(diag(contingency))

write.csv(pred.dat,file="DataWork/predict.csv",row.names = F)

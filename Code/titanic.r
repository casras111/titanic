#Kaggle titanic competition
library(glmnet)
library(rpart)
library(ggplot2)
library(MASS)

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

#age missing data - set to average
AvgAgeTrain <- mean(train$Age,na.rm=T)
train[is.na(train$Age),"Age"] <- AvgAgeTrain
test[is.na(test$Age),"Age"] <- AvgAgeTrain
train$LogAge <- log(train$Age)
test$LogAge  <- log(test$Age)

#embarkation missing data - not significant based on linear regression R2
train[!(train$Embarked %in% c("C","Q","S")),]

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


##### Modelling #############

caseswitch <- 13 #choose model to create predict.csv file

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
         #5 - Logit with glm - 81% on train
         logit1 <- glm(data=train,Survived~Age+Sex+Pclass+SibSp,
                       family=binomial(link="logit"))
         resp.train <- predict(logit1,type="response")
         print(ggplot(data.frame(survive=as.factor(train$Survived),
                           prediction=resp.train),
                aes(x=prediction,color=survive,linetype=survive))+geom_density())
         cutoff <- 0.6 #based on density graph
         pred.train <- ifelse(resp.train>cutoff,1,0)
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
         train10 <- train[,c("CabinD","CabinE","CabinF",
                             "Sib0","Sib1","ParchGt1")]
         train10 <- cbind(train10,train[,c("Survived","Age","Sex","Pclass")])
         lreg10 <- lm(Survived~.,train10)
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
         
         test10 <- test[,c("CabinD","CabinE","CabinF",
                           "Sib0","Sib1","ParchGt1")]
         test10 <- cbind(test10,test[,c("Age","Sex","Pclass")])
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
         x <- data.matrix(train[,c("Sex","Pclass","Age","Sib0","Sib1","ParchGt1",
                                   "CabinD","CabinE","CabinF")])
         y <- train$Survived
         cvfit2 <- cv.glmnet(x,y,family="binomial",alpha=0) #,type.measure = "class")
         #use 1 standard deviation from minimum lambda for regularization
         pred.train <- predict(cvfit2,newx=x,type="class",s="lambda.1se")
         x <- data.matrix(test[,c("Sex","Pclass","Age","Sib0","Sib1","ParchGt1",
                                  "CabinD","CabinE","CabinF")])
         p1 <- as.numeric(predict(cvfit2,newx=x,type="class",s="lambda.1se"))
         pred.dat <- data.frame(PassengerId=test$PassengerId,Survived=p1)
       }
       
)


(contingency <- prop.table(table(pred=pred.train,train=train$Survived)))
sum(diag(contingency))

write.csv(pred.dat,file="DataWork/predict.csv",row.names = F)

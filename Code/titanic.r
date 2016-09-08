#Kaggle titanic competition
library(glmnet)

train <- read.csv("DataRaw/train.csv")
test  <- read.csv("DataRaw/test.csv")

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


caseswitch <- 8

switch(caseswitch,
       
       {
         #1 - Naive predict
         naive_predict <- round(sum(train$Survived)/dim(train)[1])
         pred.train <- rep(naive_predict,nrow(train))
         pred.dat <- data.frame(PassengerId=test$PassengerId,Survived=naive_predict)
       },
       
       {
         #2 - Simple linear regression based on Pclass predict
         lreg1 <- lm(Survived~Pclass,train)
         pred.train <- round(predict(lreg1))
         p1 <- round(predict(lreg1,newdata=test))
         pred.dat <- data.frame(PassengerId=test$PassengerId,Survived=p1)
       },
       
       {
         #3 - Simple linear regression based on Pclass+Sex predict
         lreg2 <- lm(Survived~Pclass+Sex,train)
         pred.train <- round(predict(lreg2))
         p1 <- round(predict(lreg2,newdata=test))
         pred.dat <- data.frame(PassengerId=test$PassengerId,Survived=p1)
       },
       
       {
         #4 - Simple linear regression based on Pclass+Sex predict
         lreg3 <- lm(Survived~Age+Sex+Pclass+SibSp,train)
         pred.train <- round(predict(lreg3))
         p1 <- round(predict(lreg3,newdata=test))
         pred.dat <- data.frame(PassengerId=test$PassengerId,Survived=p1)
       },
       
       {
         #5 - Logit with glm
         logit1 <- glm(data=train,Survived~Age+Sex+Pclass+SibSp,
                       family=binomial(link="logit"))
         pred.train <- round(predict(logit1,type="response"))
         p1 <- round(predict(logit1,newdata=test,type="response"))
         pred.dat <- data.frame(PassengerId=test$PassengerId,Survived=p1)
       },
       
       {
         #6 - Logit with glmnet
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
         #7 - Simple linear regression with Cabin info
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
         #8 - Simple linear regression with significant Cabin info
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
         #9 - Logit with glmnet + cabin info
         x <- data.matrix(train[,c("Sex","Pclass","Age","SibSp","Embarked",
                                   "CabinD","CabinE","CabinF")])
         y <- train$Survived
         cvfit <- cv.glmnet(x,y,family="binomial",alpha=0)
         #use 1 standard deviation from minimum lambda for regularization
         pred.train <- predict(cvfit,newx=x,type="class",s="lambda.1se")
         x <- data.matrix(test[,c("Sex","Pclass","Age","SibSp","Embarked",
                                  "CabinD","CabinE","CabinF")])
         p1 <- as.numeric(predict(cvfit,newx=x,type="class",s="lambda.1se"))
         pred.dat <- data.frame(PassengerId=test$PassengerId,Survived=p1)
       }
       
)

(contingency <- prop.table(table(pred=pred.train,train=train$Survived)))
sum(diag(contingency))

write.csv(pred.dat,file="DataWork/predict.csv",row.names = F)

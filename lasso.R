setwd("C:/RFiles/LASSO/")
load("DataSet.RData")
#install.packages("glmnet")
library(glmnet)

dim(DataSet)
head(DataSet)

Y <- DataSet$Y
X <- DataSet[-Y]
n <- nrow(X)

#H <- diag(n) - 1/n*matrix(1, ncol=n, nrow=n)
X <- scale(X, center = TRUE, scale = TRUE)
table(Y)

# selecting a training and a test set
set.seed(17061985)
nTrain <- round(0.8*n)
indTrain <- sample(1:n, nTrain)
XTrain <- X[indTrain, ]
YTrain <- Y[indTrain]
XTest <- X[-indTrain, ]
YTest <- Y[-indTrain]
table(YTest)  # percentage of 0 and 1 stays the same

# Q2) selecting an optimal lambda through CV on training data
m.cv <- cv.glmnet(x=XTrain, y=YTrain, alpha=1,
                  type.measure="class",
                  family="binomial")
print(m.cv$lambda.min)
print(m.cv$lambda.1se)
plot(m.cv)

# Q3) fit logistic regression model with lasso for selected lambda
m <- glmnet(x=XTrain, y=YTrain, alpha=1,
            family="binomial", lambda=m.cv$lambda.1se)
summary(coef(m))

# Q4) Construct ROC curve on test data
#install.packages("ROCR")
library("ROCR")
pred.m <- prediction(predict(m, newx=XTest,
                             s=m.cv$lambda.1se,
                             type="response"), YTest)
perf.m.min <- performance(pred.m, 'tpr', 'fpr')
plot(perf.m.min)
# AUC
auc <- performance(pred.m, 'auc')@y.values[[1]]
print(auc)
# select an apropriate treshold c
cutoffs = data.frame(cut=perf.m.min@alpha.values[[1]],
                     fpr=perf.m.min@x.values[[1]],
                     tpr=perf.m.min@y.values[[1]])

# create a new parameter delta = TPR - FPR
library(tidyverse)
cutoffs <- cutoffs %>% 
  mutate(delta = cutoffs$tpr - cutoffs$fpr)

# determine ideal treshold value, sensitivity and specificity
indMax <- which.max(cutoffs[,4])
c <- cutoffs[indMax, 1]
print(c)

sensitivity <- cutoffs[indMax, 3]
specificity <- 1 - cutoffs[indMax, 2]
print(sensitivity)
print(specificity)

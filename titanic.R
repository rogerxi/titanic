#' ---
#' title: "Titanic: Machine Learning from Disaster (Kaggle)"
#' date: "Dec. 09, 2015"
#' ---
rm(list = ls())
set.seed(123)
library(rpart)
library(randomForest)
library(party)
library(ggplot2)
library(Matrix)
library(xgboost)
library(VIM)
library(e1071)
library(corrplot)
library(ROCR)
library(cvTools)


perf.measures <- function(actual, predicted) {
  # confusion matrix
  cm = table(actual, predicted)
  cm
  # accuracy
  accuracy = (cm[1,1] + cm[2,2]) / nrow(test)
  # precision
  if ((cm[1,1]) + (cm[2,1]) == 0) {
    precision = 0
  } else {
    precision = (cm[1,1]) / ((cm[1,1]) + (cm[2,1]))
  }
  # recall
  if ((cm[1,1]) + (cm[1,2]) == 0) {
    recall = 0
  } else {
    recall = (cm[1,1]) / ((cm[1,1]) + (cm[1,2]))
  }
  # fscore
  fscore = (2 * precision * recall) / (precision + recall)
  measures = c(accuracy, precision, recall, fscore)
  measures
}

## random forest
rf <- function(formula, train, test, ntree = 5000, mtry = 5) {
  fit = cforest(formula, data = train, controls = cforest_unbiased(ntree = ntree, mtry = mtry))
  predicted = predict(fit, test, OOB = TRUE, type = "response")
  predicted
  
  actual = test$Survived
  perf.measures(actual, predicted)
}


## plot ROC and lift chart
plot.roc.lift <- function(data, probs, actuals) {
  ## ROC
  df = data.frame(probs, actuals)
  pred = prediction(df$probs, df$actuals)
  perf = performance(pred, "tpr", "fpr")
  plot(perf, main = "ROC")
  
  #   # Lift chart
  #   df_rank = as.data.frame(df[order(probs, decreasing = TRUE),])
  #   colnames(df_rank) = c('predicted', 'actual')
  #   # overall "yes" probability for the test dataset
  #   baseRate = mean(actuals)
  #   n_total = length(data$y)
  #   ax = dim(n_total)
  #   ay_base = dim(n_total)
  #   ay_pred = dim(n_total)
  #   ax[1] = 1
  #   ay_base[1] = baseRate
  #   ay_pred[1] = df_rank$actual[1]
  #   for (i in 2:n_total) {
  #     ax[i] = i
  #     ay_base[i] = baseRate * i
  #     ay_pred[i] = ay_pred[i-1] + df_rank$actual[i]
  #   }
  #   
  #   plot(ax, ay_pred, xlab = "Number of cases", ylab = "Number of successes", 
  #        main = "Lift")
  #   points(ax, ay_base, type = "l")
}


## implement k-fold cross validation
k.fold.cv <- function(formula, data, ntree = 5000, mtry = 5, cv.k = 4) {
  set.seed(123)
  # Split observations into k groups
  cvGroup = cvFolds(nrow(data), K = cv.k)
  accuracys = dim(cv.k)
  precisions = dim(cv.k)
  recalls = dim(cv.k)
  fscores= dim(cv.k)
  probs = NULL
  actuals = NULL
  
  for (i in (1:cv.k)) {
    trainIdx = which(cvGroup$which != i)
    testIdx = which(cvGroup$which == i)
    # train model
    fit = cforest(formula, data = data[trainIdx,], controls = cforest_unbiased(ntree = ntree, mtry = mtry))
    prob = predict(fit, data[testIdx,], OOB = TRUE, type = "prob")
    prob = as.data.table(prob)[2,]
    prob = as.vector(t(prob))
    
    # recode as 1 if probability >= 0.5
    predicted = floor(prob + 0.5)
    actual = as.numeric(as.character(data[testIdx,]$Survived))
    
    # confusion matrix
    cm = table(actual, predicted)
    cm
    # accuracy
    accuracy = (cm[1,1] + cm[2,2]) / nrow(data[testIdx,])
    accuracys[i] = accuracy
    # precision
    if ((cm[1,1]) + (cm[2,1]) == 0) {
      precision = 0
    } else {
      precision = (cm[1,1]) / ((cm[1,1]) + (cm[2,1]))
    }
    precisions[i] = precision
    # recall
    if ((cm[1,1]) + (cm[1,2]) == 0) {
      recall = 0
    } else {
      recall = (cm[1,1]) / ((cm[1,1]) + (cm[1,2]))
    }
    recalls[i] = recall
    
    # fscore
    fscore = (2 * precision * recall) / (precision + recall)
    fscores[i] = fscore
    probs = c(probs, prob)
    actuals = c(actuals, actual)
  }
  
  # calculate the avarage of accuracy, precision, recall, and fscore
  avg_accuracy = mean(accuracys)
  avg_precision = mean(precisions)
  avg_recall = mean(recalls)
  avg_fscore = mean(fscores)  
  Rowname = c("Model")
  Columnname = c("Accuracy", "Precission", "Recall", "F1 Scores")
  measures = matrix(c(avg_accuracy, avg_precision, avg_recall, avg_fscore), 
                    nrow = 1, ncol = 4, byrow = TRUE, dimnames = list(Rowname, Columnname))
  
  # plot ROC and lift
  plot.roc.lift(data, probs, actuals)
  return(measures)
}



##------------------------------ load data ------------------------------
## import ds
filePath = 'http://s3.amazonaws.com/assets.dscamp.com/course/Kaggle/'
filePath = '/Users/roger/Downloads/'
train_raw = read.csv(sprintf('%s%s', filePath, 'train.csv'), stringsAsFactors = F)
test_raw = read.csv(sprintf('%s%s', filePath, 'test.csv'), stringsAsFactors = F)
names(train_raw)


##------------------------------ process data ------------------------------
## combine two sets
test_raw$Survived = NA
ds = rbind(train_raw, test_raw)
ds$Survived = as.factor(ds$Survived)
ds$Pclass = as.factor(ds$Pclass)
ds$Sex = as.factor(ds$Sex)
is.na(ds$Age) = (ds$Age == '')
is.na(ds$Fare) = (ds$Fare == '')
is.na(ds$Cabin) = (ds$Cabin == '')
is.na(ds$Embarked) = (ds$Embarked == '')


## visualize data
## missing values
ss = ds[,-c(1:2)]
aggr(ss, labels = colnames(ss), ylim = 0.9)

train = ds[1:891,]
## Survived, Pclass, Age
ggplot(train[!is.na(train$Age),], aes(x = Age, fill = Survived, alpha = 0.5)) + geom_density() + facet_grid(Pclass ~ .)
## Survived, Pclass, Sex
ggplot(train, aes(x = Sex, fill = Survived)) + geom_histogram(binwidth = 0.5) + facet_grid(Pclass ~ .)
## Survived ~ Fare
ggplot(train[!is.na(train$Fare),], aes(Survived, Fare)) + geom_boxplot(na.rm = TRUE)
## Survived
ggplot(train[!is.na(train$Age),], aes(Survived, Age)) + geom_violin(aes(fill = Survived))


## name
ds$Name = as.character(ds$Name)
ds$Surname = sapply(ds$Name, FUN = function(x) {strsplit(x, split = '[,.]')[[1]][1]})
# ## unique surnames
# unique_names = sort(unique(ds$Surname))
# dist_mat = as.data.frame(adist(unique_names, unique_names))
# colnames(dist_mat) = unique_names
# rownames(dist_mat) = unique_names
# ## similar surnames
# unique(rownames(which(dist_mat == 1, arr.ind = TRUE)))
# ds$Surname[ds$Surname %in% c("Laitinen", "Lahtinen")] = "Lahtinen"
# ds$Surname[ds$Surname %in% c("Lundahl", "Lindahl")] = "Lundahl"
# ds$Surname[ds$Surname %in% c("McCrie", "McCrae")] = "McCrae"
# ds$Surname[ds$Surname %in% c("Mitkoff", "Minkoff")] = "Minkoff"
# ds$Surname[ds$Surname %in% c("Petersen", "Pettersson", "Petterson")] = "Petersen"
# ds$Surname[ds$Surname %in% c("Saade", "Saad")] = "Saad"
# ds$Surname[ds$Surname %in% c("Troutt", "Trout")] = "Trout"
# ds$Surname[ds$Surname %in% c("Yousseff", "Youseff")] = "Youseff" 
# ds$Surname = as.factor(ds$Surname)

## family size
ds$FamilySize = ds$SibSp + ds$Parch + 1
ds$FamilyID = paste(as.character(ds$FamilySize), ds$Surname, sep = "")
ds$FamilyID[ds$FamilySize < 2] = 'Small'
famIDs = as.data.frame(table(ds$FamilyID))
famIDs = famIDs[famIDs$Freq <= 2,]
ds$FamilyID[ds$FamilyID %in% famIDs$Var1] = 'Small'
ds$FamilyID = as.factor(ds$FamilyID)
ds$FamilyID2 = ds$FamilyID
ds$FamilyID2 = as.character(ds$FamilyID2)
ds$FamilyID2[ds$FamilySize <= 3] = 'Small'
ds$FamilyID2 = as.factor(ds$FamilyID2)
ggplot(ds[1:891,], aes(x = FamilySize, fill = Survived)) + geom_histogram(binwidth = 0.5) + facet_grid(Survived ~ .)


## title
ds$Title = sapply(ds$Name, FUN = function(x) {strsplit(x, split = '[,.]')[[1]][2]})
ds$Title = sub(' ', '', ds$Title)
table(ds$Title)
ds$Title[ds$Title %in% c('Mme', 'Mlle', 'Ms')] = 'Miss'
ds$Title[ds$Title %in% c('Capt', 'Don', 'Jonkheer', 'Major', 'Rev')] = 'Mr'
ds$Title[ds$Title %in% c('Lady', 'the Countess')] = 'Mrs'
ds$Title = as.factor(ds$Title)
ggplot(ds[1:891,], aes(x = Title, fill = Survived)) + geom_histogram(binwidth = 0.5) + facet_grid(Survived ~ .)


## embarkation
table(ds$Embarked)
which(is.na(ds$Embarked))
ds$Embarked[c(62,830)] = "S"
ds$Embarked = as.factor(ds$Embarked)
ggplot(ds[1:891,], aes(x = Embarked, fill = Survived)) + geom_histogram() + facet_grid(Embarked ~ .)


## ticket
##    1. for ticket number having only digits, count the number of digits;
##    2. for ticket number having letters, get the prefix
ds$TicketID = sapply(ds$Ticket, 
                     FUN = function(x) {
                       if (grepl("^[0-9]+$", x, perl = TRUE)) {
                         nchar(x)
                       } else {
                         strsplit(x, '[./ ]')[[1]][1]
                       }})
ds$TicketID = as.factor(ds$TicketID)
ggplot(ds, aes(x = TicketID)) + geom_histogram()

## same ticket number
length(unique(ds$Ticket))
ds$SameTicket = sapply(ds$Ticket, FUN = function(x) {table(ds$Ticket == x)['TRUE']})
ds$SameTicketID = paste(as.character(ds$SameTicket), ds$Ticket, sep = "_")
ds$SameTicketID[ds$SameTicket < 2] = 1
ds$SameTicketID = as.factor(ds$SameTicketID)
ggplot(ds[1:891,], aes(x = SameTicket, fill = Survived)) + geom_histogram(binwidth = 0.5) + facet_grid(Survived ~ .)


## fare
which(is.na(ds$Fare))
ds$Fare[1044] = median(ds$Fare, na.rm = TRUE)
ds$Fare2 = ds$Fare / ds$SameTicket


## age
fit.age = rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare2 + Embarked + Title + FamilySize,
                data = ds[!is.na(ds$Age),], method = "anova")
ds$Age[is.na(ds$Age)] = predict(fit.age, ds[is.na(ds$Age),])


## children
ds$Child = 0
ds$Child[which(ds$Age < 19 & ds$Pclass == "1")] = 1
ds$Child[which(ds$Age < 16 & ds$Pclass == "2")] = 1
ds$Child[which(ds$Age < 14 & ds$Pclass == "3")] = 1
ds$Child = as.factor(ds$Child)
ggplot(ds[1:891,], aes(x = Child, fill = Survived)) + geom_histogram(binwidth = 0.5) + facet_grid(Pclass ~ .)


## cabin
table(ds$Cabin)
ds$CabinID = substring(ds$Cabin, 1, 1)
table(ds$CabinID)
is.na(ds$CabinID) = (ds$CabinID == '')
ds$CabinID = as.factor(ds$CabinID)
## decision tree
fit = rpart(CabinID ~ Pclass + Sex + Age + SibSp + Parch + Fare2 + Embarked + 
                      Title + FamilySize + TicketID, 
            data = ds[!is.na(ds$CabinID),], method = "class")
pfit = prune(fit, cp = fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"])
prob = predict(pfit, ds[is.na(ds$CabinID),])
pred = sapply(data.frame(t(prob)), FUN = function(x) {which.max(x)})
ds$CabinID[is.na(ds$CabinID)] = levels(ds$CabinID)[pred]


## correlation
idx = c("Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Fare2", "Embarked", 
        "Title", "FamilySize", "FamilyID", "FamilyID2", "TicketID", "Child",
        "SameTicket", "SameTicketID", "CabinID")
ss = ds[1:891,idx]
for (i in 1:ncol(ss)) {
  ss[,i] = as.numeric(ss[,i])
}
corrplot.mixed(cor(ss), lower="ellipse", upper="color", 
               tl.pos="lt", diag="n", order="hclust", hclust.method="complete")



##------------------------------ random forest ------------------------------
train = ds[1:891,]
test = ds[892:1309,]

## cross validation
k.fold.cv(formula = Survived ~ Pclass + Sex + Age + Fare + Embarked + 
            Title + FamilySize, train, ntree = 1000, mtry = 3, cv.k = 5)
# 0.829383  0.8363485 0.8997149 0.8668062

k.fold.cv(formula = Survived ~ Pclass + Sex + Age + Parch + Fare2 + Embarked +
            Title + FamilySize + FamilyID + CabinID + Child, 
          train, ntree = 1000, mtry = 3, cv.k = 5)
# 0.8327412  0.8304258 0.9159519 0.8710278

k.fold.cv(formula = Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare2 + Embarked +
            Title + FamilySize + FamilyID + SameTicketID + CabinID + Child, 
          train, ntree = 1000, mtry = 3, cv.k = 5)
# 0.8327475  0.8294244 0.91784 0.8713101

k.fold.cv(formula = Survived ~ Pclass + Sex + Age + Parch + Fare2 + Embarked +
            Title + FamilySize + FamilyID + SameTicketID + CabinID + Child, 
          train, ntree = 1000, mtry = 3, cv.k = 5)
# 0.8349884  0.8309863 0.9197389 0.8730406

k.fold.cv(formula = Survived ~ Pclass + Sex + Age + Parch + Fare + Embarked +
            Title + FamilySize + FamilyID, train, ntree = 1000, mtry = 3, cv.k = 5)
#0.8349884  0.8316742 0.9176425 0.8724918

k.fold.cv(formula = Survived ~ Pclass + Sex + Age + Parch + Fare + Embarked +
            Title + FamilySize + FamilyID + TicketID + SameTicket + SameTicketID, 
          train, ntree = 1000, mtry = 3, cv.k = 5)
# 0.8394828  0.8319997 0.9270017 0.8767752


## predict testing data
fit = cforest(Survived ~ Pclass + Sex + Age + Parch + Fare2 + Embarked +
                Title + FamilySize + FamilyID + TicketID + SameTicket + SameTicketID +
                CabinID + Child, 
              data = train, controls = cforest_unbiased(ntree=1000, mtry=3))
#0.81818
fit = cforest(Survived ~ Pclass + Sex + Age + Parch + Fare + Embarked +
                Title + FamilySize + FamilyID + TicketID + SameTicket + SameTicketID, 
              data = train, controls = cforest_unbiased(ntree=1000, mtry=3))
#0.82297
predicted = predict(fit, test, OOB = TRUE, type = "response")
predicted
output = data.frame(PassengerId = test$PassengerId, Survived = predicted)
write.csv(output, file = sprintf("/Users/roger/Desktop/rf_%s.csv", Sys.time()), row.names = FALSE)


### alternative method:
# ##------------------------------ XGBoost ------------------------------
# xgboosting <- function(xTrain, yTrain, xTest) {
#   for (i in 1:ncol(xTrain)) {
#     xTrain[,i] = as.double(xTrain[,i])
#   }
#   xTrain = as(as.matrix(xTrain), "dgCMatrix")
#   ## train the classification model
#   xgb = xgboost(data = xTrain, label = as.numeric(yTrain) - 1, missing = -999,
#                 nround = 20, objective = "binary:logistic")
#   
#   #   xgb_cv = xgb.cv(data = xTrain, nfold = 10, label = as.numeric(yTrain) - 1,
#   #                   nround = 200, objective = "binary:logistic")
#   
#   ## predict on the testing set
#   xgb_pred = round(predict(xgb, newds = ds.matrix(xTest)))
#   xgb_pred
# }
# 
# # idx = c(3,5,6,8,10,12,14,15,17)
# idx = c("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", 
#         #"Cabin", "Surname", "Ticket", 
#         "Embarked", "Child", "Title", "FamilySize", "FamilyID", "FamilyID2",
#         "TicketID")
# xTrain = train[,idx]
# yTrain = train[,2]
# xTest = test[,idx]
# predicted = xgboosting(xTrain, yTrain, xTest)
# predicted
# output = ds.frame(PassengerId = test$PassengerId, Survived = predicted)
# write.csv(output, file = sprintf("/Users/roger/Desktop/rf_%s.csv", Sys.time()), row.names = FALSE)
# 
# 
# xTrain = train[,idx]
# yTrain = train[,2]
# xTest = xTrain[501:891,]
# actual = yTrain[501:891]
# xTrain = xTrain[1:500,]
# yTrain = yTrain[1:500]
# predicted = xgboosting(xTrain, yTrain, xTest)
# predicted
# perf.measures(actual, predicted)
# 20: 0.7846890 0.8377358 0.9173554 0.8757396
# 10: 0.7775120 0.8211679 0.9297521 0.8720930
# 12: 0.7822967 0.8320896 0.9214876 0.8745098
# 
# 
# ## plot the most important features
# names = colnames(dtm_train[, -ncol(dtm_train)])
# importance_matrix = xgb.importance(names, model = xgb)
# xgb.plot.importance(importance_matrix[1:15,])
# # xgb.plot.tree(feature_names = names, model = xgb1, n_first_tree = 2)

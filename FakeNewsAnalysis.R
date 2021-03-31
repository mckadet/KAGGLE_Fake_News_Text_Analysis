# McKade Thomas
# KAGGLE Competition: Fake News
# URL: https://www.kaggle.com/c/fake-news

# Libraries needed
library(tidyverse)
library(tidytext)
library(vroom)
library(caret)

# Read in the data
news <- vroom('CleanFakeNews.csv')
news.train <- news %>% filter(Set=="train")
news.test <- news %>% filter(Set=="test")

news.train$isFake <- as.factor(news.train$isFake)
filtered_train <- news.train %>% select(-c(language.x,Set))


### Submission 1
## XGBoost with Default Parameters and TFIDF for Article
xgbGrid <- expand.grid(nrounds = c(50, 150),
                       max_depth = 6,
                       eta = 1,
                       gamma = 0,
                       colsample_bytree = 1,
                       min_child_weight = 1,
                       subsample = 1)

newsXGModel <- train(form=isFake~.-Id,
                data=filtered_train,
                method = "xgbTree",
                tuneGrid = xgbGrid,
                trControl=trainControl(
                  method="repeatedcv",
                  number=5,
                  repeats=1,
                  verboseIter = TRUE))

print(newsXGModel)
plot(newsXGModel)

XGpreds <- predict(newsXGModel,newdata=news %>% filter(Set=="test"))
submission <- data.frame(id=news %>% filter(Set=="test") %>% pull(Id),
                         label=XGpreds)

write.csv(x=submission,file="./XGDefaultSubmission.csv",row.names=FALSE)



### Submission 2
tgrid <- expand.grid(
  mtry = 4,
  splitrule = "gini",
  min.node.size = 10
)

## Random Forest
news.forest <- train(form=isFake~.-Id,
                     data=filtered_train, 
                     num.trees=100,
                    method="ranger",
                    trControl=trainControl(
                      method="repeatedcv",
                      number=3,
                      repeats=1,
                      verboseIter = TRUE),
                    tuneGrid = tgrid)

forestPreds <- predict(news.forest,newdata=news %>% filter(Set=="test"))
submissionForest <- data.frame(id=news %>% filter(Set=="test") %>% pull(Id),
                         label=forestPreds)

write.csv(x=submissionForest,file="./rangerSubmission.csv",row.names=FALSE)


### Submission 3
xgbGrid2 <- expand.grid(nrounds = 250,
                       max_depth = c(4, 6),
                       eta = c(0.4, 1),
                       gamma = 0,
                       colsample_bytree = 1,
                       min_child_weight = 1,
                       subsample = 1)

newsXGModel2 <- train(form=isFake~.-Id,
                     data=filtered_train,
                     method = "xgbTree",
                     objective='binary:logistic',
                     tuneGrid = xgbGrid2,
                     trControl=trainControl(
                       method="repeatedcv",
                       number=3,
                       repeats=1,
                       verboseIter = TRUE))

print(newsXGModel2)
plot(newsXGModel2)

XGpreds3 <- predict(newsXGModel2,newdata=news %>% filter(Set=="test"))
submission3 <- data.frame(id=news %>% filter(Set=="test") %>% pull(Id),
                         label=XGpreds3)

write.csv(x=submission3,file="./XG3Submission.csv",row.names=FALSE)

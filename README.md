# KAGGLE Competition: Text Analysis to Predict Fake News

## Project Overview
The purpose of this project is to summarize the work I did on the Kaggle Fake News competition (https://www.kaggle.com/c/fake-news/overview). The goal of this competition was to predict whether or not a news article was fake via text analysis. The dataset we were given contained the author, title, and content for thousands of news articles.

## File Descriptions
FakeNewsDataCleaning.R - This file contains the code used to clean and prepare the data for modeling.
FakeNewsAnalysis.R - This file contains the code used to model the data and predict whether or not an article was fake.
XGDefaultSubmission.csv - The datafile for the submission created by the first XGBoost model using default tuning and hyper parameters.
XG2Submission.csv - The datafile for the submission created by the second XGBoost model.
rangerSubmission.csv - The datafile for the submission creaeted by the random forest model.

Note: The files for the data used for this analysis are not contained in this repository but can be found at the following link: https://www.kaggle.com/c/fake-news/data

## Methods Used to Clean the Data
The main explanatory or independent variables in the models used were the term frequency-inverse document frequency (or TFIDF) scores were calculated for over 4,000 of the most common words used in the articles. To obtain these scores, it was first determined which language each article was published in. After summarizing the articles across some of the more common languages, stopwords were then removed using both the stopwords and tidytext libraries in R. Next, the dataset was tranformed into long format with words for each article being in seperate rows. The TFIDF scores were then calculated for each word and the most common words, aside from stopwords, were kept. The dataset was then transformed again into wide format so that each column corresponded to a word while the tfidf score for that word in each article was present in each row of the dataframe.

## Methods Used to Generate Predictions
For text analysis and prediction, several methods all employing elements of the caret library in R were used. For the first model, I used a dataset containing the tfidf values for 4,489 of the most important words from the articles as explanatory variables for predicting whether or not the article was fake. I created a random forest model using ranger from the caret library with the following tuning parameters: ntrees = 100, mtry=4, splitrule = gini, min.node.size=10. I also used cross-validation as the model validation with 3 folds and 1 repeat. This model had almost 90% accuracy but needed much improvement.

For the second model, the same dataset as before was used with tfidf scores for various important words from the articles. This model employed an XGB Tree model from the caret library with repeated cross validation of 5 folds and 1 repeat. Tuning parameters included two options for nrounds, 50 and 100, with default parameters otherwise. The model had 96% accuracy which was a significant improvement from the ranger model.

For the third and final model, similar methods were used from the previous submission but with advanced tuning and hyper parameter selection and additional model parameter specification. This model used the same XGB Tree option from the caret library with the same explanatory and response variables. However, after trying several tuning parameter options in various models, it was clear that some options were performing better than others. For that reason, the following tuning parameter options were used: nrounds = 250, max_depth = c(4,6), eta = c(0.4), and gamma, min_child_weight, and sub_sample = 1. Lastly, the objective was specified to be binary:logistic, ensuring that the outcome predicted would be of a correct type for the data (a categorical response variable of 0 or 1 for fake news). This model performed best out of three with an accuracy score of 97%.

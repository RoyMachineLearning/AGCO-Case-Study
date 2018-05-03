
# coding: utf-8

# AGCO Case Analysis and Models created by Ashish Gupta

import numpy as np
import pandas as pd

# to make this notebook's output stable across runs
np.random.seed(123)

# To plot pretty figures
get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


# Load the csv file.

import os
from six.moves import urllib

DataFile = pd.read_csv("U:\\data.csv") #read the data from the csv file.


# Explore / Visualize the data

import matplotlib.pyplot as plt
DataFile.hist(bins=20, figsize=(20,15))
plt.show()


# We can see that the days are not normalized. But before normalizing the data, we create the train and test split.

from sklearn.model_selection import StratifiedShuffleSplit

DataFile = DataFile.drop('Establishment ID',axis=1)
DataFile = DataFile.drop('Postal First', axis=1)
DataFile = DataFile.drop('Email on file', axis=1)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=123)
for train_index, test_index in split.split(DataFile, DataFile["Current Infraction"] ):
    train_set = DataFile.loc[train_index]
    test_set = DataFile.loc[test_index]

#the Y Variable
train_set_y = train_set["Current Infraction"].copy()
test_set_y = test_set["Current Infraction"].copy()

#the X variables
train_set_X = train_set.drop("Current Infraction", axis=1)
test_set_X = test_set.drop("Current Infraction", axis=1)


#Now to normalize the number of days - we will create a pipeline.

Training_Transformed_X = train_set_X[['Days Since License Effective']]


Training_NonTransformed_X = train_set_X[['Club', 'Business Building', 'Small Hotel' ,'Small Restaurant',
                                        'Large Hotel Chain','Concert Facility','Stadium/Sports Facility',
                                        'Karaoke Bar','Other','Large Restaurant Chain','Previous Infraction']]


#Create numerical pipeline to transform numerical values

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#Convert the non transformed Dataframe into list.
Training_Transformed_list = list(Training_Transformed_X)
Training_NonTransformed_list = list(Training_NonTransformed_X)

from sklearn.base import BaseEstimator, TransformerMixin

# Create a class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(Training_Transformed_list)),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(Training_NonTransformed_list))
    ])


import warnings
warnings.filterwarnings('ignore')
from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
    ])

Final_training_X = full_pipeline.fit_transform(train_set_X)
Final_test_X = full_pipeline.transform(test_set_X)


#Now we will build the prediction models - Logistic Regression, Gradient Boosting and Random Forest.
#import the libraries for all the classifiers.

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


#We start our analysis with the Logistic Regression Classifer using Stats models
import statsmodels.api as sm
Logistic_Regression = sm.Logit(train_set_y,Final_training_X).fit()
Logistic_Regression.summary()


#Now Perform Backward Elimination to get Final Logistic Model

X_opt = Final_training_X[:,[0,1,2,3,4,5,7,8,9,11]] # Only take the significant variables.

logregressor_stp = sm.Logit(endog = train_set_y, exog = X_opt).fit()
logregressor_stp.summary()


#Odds Ratio for the new logistic regression model
np.exp(logregressor_stp.params)


#Confidence Intervals for the new Logistic Regressions

params = logregressor_stp.params
conf = logregressor_stp.conf_int()
conf['OR'] = params
conf.columns = ['2.5%', '97.5%', 'OR']
print(np.exp(conf))


Test: # Lets try getting similar insights using sklearn

Logistic_Regression = LogisticRegression(random_state = 42)
Logistic_Regression.fit(Final_training_X,train_set_y)
Logistic_Regression.coef_


np.exp(Logistic_Regression.coef_)


# As you may notice, that most of the statistics are not easy to get from scikit-learn (where model evaluation is mostly done using cross-validation), So Stats models can be used to present the business insights. 


#Now create Random Forest and Gradient Boosting.

# Random Forest Classifier
random_forest_Classifier = RandomForestClassifier()
random_forest_Classifier.fit(Final_training_X,train_set_y)


# Gradient Boosting Classifier
GB_Classifier = GradientBoostingClassifier(random_state = 42)
GB_Classifier.fit(Final_training_X,train_set_y)


# Now Tune the models.

#Design the random forest classifier and use Grid Search technique to tune the hyper parameters.

# Random Forest Classifier

n_estimators = [3, 7]
max_features = [0.1, 0.3]
max_depth = [2, 7, 10] 
oob_score = [True, False]
min_samples_split = [0.1, 0.5]
min_samples_leaf = [0.1, 0.5]
max_leaf_nodes = [2, 10, 50]

parameter_random_forest = {'n_estimators' : n_estimators, 'max_features' : max_features,
                     'max_depth' : max_depth, 'min_samples_split' : min_samples_split,
                    'oob_score' : oob_score, 'min_samples_leaf': min_samples_leaf, 
                     'max_leaf_nodes' : max_leaf_nodes}
             
Random_Forest_Classifier = RandomForestClassifier(random_state = 42)

#use grid search to tune the model

grid_search_RndmForest = GridSearchCV(Random_Forest_Classifier,parameter_random_forest, cv = 4, scoring='roc_auc', refit = True,
                                     n_jobs = -1, verbose=2)

grid_search_RndmForest.fit(Final_training_X,train_set_y)
             
forest_best_params_ = grid_search_RndmForest.best_params_
forest_best_estimators_ = grid_search_RndmForest.best_estimator_

print(forest_best_params_)
print(forest_best_estimators_)


# Build the next classifier : Gradient Boosting Classifier

GB_Classifier = GradientBoostingClassifier(random_state = 42)

n_estimators = [3, 7]
learning_rate = [0.1, 0.01, .001]
max_depth = [2,7,10]
max_features = [4, 6]
min_samples_split = [0.1, 0.5]
min_samples_leaf = [0.1, 0.5]
max_leaf_nodes = [2, 10, 50]
                            
parameter_GB_Classifier = {'n_estimators' : n_estimators, 'learning_rate' : learning_rate,
                              'max_depth' : max_depth, 'min_samples_split' : min_samples_split,
                              'min_samples_leaf' : min_samples_leaf, 'max_features' : max_features,
                              'max_leaf_nodes' : max_leaf_nodes}

grid_search_GB_Classifier = GridSearchCV(GB_Classifier, parameter_GB_Classifier, cv = 4, scoring='roc_auc', 
                               refit = True, n_jobs = -1, verbose = 2)

grid_search_GB_Classifier.fit(Final_training_X,train_set_y)

GB_Classifier_best_params_ = grid_search_GB_Classifier.best_params_

GB_Classifier_best_estimators_ = grid_search_GB_Classifier.best_estimator_

print(GB_Classifier_best_params_)

print(GB_Classifier_best_estimators_)


# Now Create the Logistic Regression model using Sklearn approach

X_opt = Final_training_X[:,[0,1,2,3,4,5,7,8,9,11]] #taking the significant variables - we will tune the model further.

Logistic_Regression = LogisticRegression(random_state = 42)
C = [0.0001, 0.001, 0.01, 0.1]
                            
parameter_LogReg = {'C' : C}
grid_search_LogReg = GridSearchCV(Logistic_Regression, parameter_LogReg, cv = 4, scoring='roc_auc', 
                               refit = True, n_jobs = -1, verbose = 2)

grid_search_LogReg.fit(X_opt,train_set_y)

LogReg_best_params_ = grid_search_LogReg.best_params_
LogReg_best_estimators_ = grid_search_LogReg.best_estimator_

print(LogReg_best_params_)
print(LogReg_best_estimators_)


#Evaluate the Random Forest Model on the Training set

from sklearn.model_selection import cross_val_predict
y_probas_forest = cross_val_predict(Random_Forest_Classifier,Final_training_X,train_set_y, cv=4,
                                    method="predict_proba")
y_forest_scores = y_probas_forest[:, 1] # score = proba of positive class

#Evaluate the Gradient Boosting Model on the Training set

y_probas_gradient = cross_val_predict(GB_Classifier,Final_training_X,train_set_y, cv=4,
                                    method="predict_proba")

y_gradient_scores = y_probas_gradient[:, 1] # score = proba of positive class

#Evaluate The logistic Regression model on Training set

y_probas_logistic = cross_val_predict(Logistic_Regression,Final_training_X,train_set_y, cv=4,
                                    method="predict_proba")

y_logistic_scores = y_probas_logistic[:, 1] # score = proba of positive class


#Evaluate the model performance using ROC Curve

from sklearn.metrics import roc_curve

# for random forest
fpr_forest, tpr_forest, thresholds_forest = roc_curve(train_set_y,y_forest_scores)

# for gradient boosting
fpr_gradient, tpr_gradient, thresholds_gradient = roc_curve(train_set_y,y_gradient_scores)

# for logistic regression
fpr_logistic, tpr_logistic, thresholds_logistic = roc_curve(train_set_y,y_logistic_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    
#Plot the Roc Curve
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.plot(fpr_gradient, tpr_gradient, linewidth=2, label="Gradient Boosting")
plt.plot(fpr_logistic, tpr_logistic, linewidth=1, label="Logistic Regression")
plt.legend(loc="lower right", fontsize=12)
plt.show()


# Calculate the AUC and choose the best model. From the result - we can see Gradient Boosting is the best model.

from sklearn.metrics import roc_auc_score

auc_RandomClassifier = roc_auc_score(train_set_y,y_forest_scores)
auc_GradientBoosting = roc_auc_score(train_set_y,y_gradient_scores)
auc_LogisticRegression = roc_auc_score(train_set_y,y_logistic_scores)

print(auc_RandomClassifier)
print(auc_GradientBoosting)
print(auc_LogisticRegression)


#Create the confusion Matrix for Gradient Boosting - which is the best model
from sklearn.metrics import confusion_matrix

#need to create a binary classifier as Classification metrics can't handle a mix of binary and continuous targets
predict_Gboosting = cross_val_predict(GB_Classifier,Final_training_X,train_set_y, cv=4)
print(confusion_matrix(train_set_y,predict_Gboosting))

#Print the Precision / Recall Score
from sklearn.metrics import precision_score, recall_score
print(precision_score(train_set_y,predict_Gboosting))
print(recall_score(train_set_y,predict_Gboosting))


#Now I will create a precision - recall curve
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(train_set_y,y_gradient_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="best")
    plt.grid(True)
    
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

plt.show()

#plot the precision and recall curve

def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])

    
plot_precision_vs_recall(precisions, recalls)
plt.show()


# Since we got the best model : Gradient Boosting, We got 79% of recall. 
#We can now adjust the threshhold to 90-95% to get more positive cases - Positive cases means more defaults.

Recall_Threshhold = (y_gradient_scores > .11)
print(precision_score(train_set_y,Recall_Threshhold))
print(recall_score(train_set_y,Recall_Threshhold))


#Now we use this new threshhold to predict the values on Training set using new threshhold
from sklearn.metrics import accuracy_score

NewR_Threshhold = Recall_Threshhold.astype(int)
grid_search_GB_Classifier.fit(Final_training_X,NewR_Threshhold)

y_Train_pred = GB_Classifier_best_estimators_.predict(Final_training_X)
accuracy_score(train_set_y, y_Train_pred)


#Next we predict the results on the test set
y_Test_pred = GB_Classifier_best_estimators_.predict(Final_test_X)
accuracy_score(test_set_y , y_Test_pred)
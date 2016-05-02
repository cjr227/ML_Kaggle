#!/usr/bin/python

import csv
import pandas as pd
import numpy as np
from scipy.stats import randint, expon
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LinearRegression
from sklearn.grid_search import RandomizedSearchCV
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.lda import LDA

# LOAD DATA
train = pd.read_csv('../input/data.csv')
test = pd.read_csv('../input/quiz.csv')

n_test = np.shape(test)[0]

# LOAD LABELS
y = train.ix[:,-1].values

# DATA PREPROCESSING
# Drop concatenated features
X = train.drop(['16','7','56','57','35','32','31','29','2','11','9','0','5','14','label'], axis=1)
X_test = test.drop(['16','7','56','57','35','32','31','29','2','11','9','0','5','14'], axis=1)

# Binarize select factor variables
X_dum = pd.get_dummies(X,columns=['8','17','18','20','23','25','26','58'])
X_dum_test = pd.get_dummies(X_test,columns=['8','17','18','20','23','25','26','58'])

# List of features in training set, but not in testing set
X_cols_to_use = list(X_dum.columns.difference(X_dum_test.columns))

# List of features in testing set, but not in training set
X_test_cols_to_use = list(X_dum_test.columns.difference(X_dum.columns))

# Drop features that are not in both training and testing sets
X_dum_2 = X_dum.drop(X_cols_to_use,axis=1)
X_dum_test_2 = X_dum_test.drop(X_test_cols_to_use,axis=1)

# SEPARATE DATA INTO TRAINING & HOLDOUT DATA
X_train, X_hold, y_train, y_hold = train_test_split(
     X_dum_2, y, test_size=0.3, random_state=42)
y_train = y_train.astype(int)
y_hold = y_hold.astype(int)

# SPECIFY CROSS-VALIDATION SETTINGS
cv_call = StratifiedKFold(y_train,n_folds=10)
n_iter_search = 30

##### MODEL 1: RANDOM FOREST #####


# RUN RANDOMIZED SEARCH FOR HYPERPARAMETER OPTIMIZATION
clf = RandomForestClassifier(random_state=42,n_jobs=-1)

param_dist = {n_estimators": randint(5, 200)}

random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search,cv=cv_call,
                                   scoring='accuracy')

random_search = random_search.fit(X_train, y_train)

# RETRIEVE OPTIMAL HYPERPARAMETER VALUES FROM RANDOM SEARCH
best_parameters, score, _ = max(random_search.grid_scores_, key=lambda x: x[1])
clf = RandomForestClassifier(random_state=42,n_jobs=-1,
				n_components=best_parameters["n_estimators"])

# RUN MODEL WITH OPTIMIZED PARAMETERS
clf = clf.fit(X_train, y_train)

# MAKE PREDICTIONS ON HOLDOUT DATA
preds_hold = clf.predict(X_hold)
print "Holdout data overall accuracy: ", np.mean(preds_hold == y_hold)

# MAKE PREDICTIONS ON TEST DATA
preds = clf.predict(X_dum_test_2)
    
# SUBMISSION FILE
idx = range(1,n_test)
submission = pd.DataFrame({"id": idx, "prediction": preds})
submission.to_csv("some_ensemble_required_sub.csv", index=False)


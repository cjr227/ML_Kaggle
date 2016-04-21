#!/usr/bin/python

import csv
import pandas as pd
import numpy as np
from scipy.stats import randint
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LinearRegression
from sklearn.grid_search import RandomizedSearchCV
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.lda import LDA


if __name__ == '__main__':
    
    # LOAD DATA 
    # (Data transformations (add extra features,split into features/labels etc.)
    # done a priori)
    train = pd.read_csv('../input/data.csv')
    test = pd.read_csv('../input/quiz.csv')
    
    # FEATURE EXTRACTION
    pca = PCA(n_components = ...)
    # Number of P.C. decided on which number does best on holdout
    _ = pca.fit(train)
    X_train = pca.transform(train)
    X_test = pca.transform(test)
    
    # RUN MODEL WITH OPTIMIZED PARAMETERS
    clf = ...
    # Variable for specific classifier (decided on which one does best on holdout)
    clf = clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    
    # SUBMISSION FILE
    idx = range(1,31710)
    submission = pd.DataFrame({"id": idx, "prediction": preds})
    submission.to_csv("some_ensemble_required_sub.csv", index=False)

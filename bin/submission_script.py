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
from sklearn.preprocessing import LabelEncoder


if __name__ == '__main__':
    
    # LOAD DATA 
    train = pd.read_csv('../input/data.csv')
    test = pd.read_csv('../input/quiz.csv')
    
    y = train.ix[:,-1].values
    y_test = test.ix[:,-1].values
    
    
    X = train.drop(['16','7','56','57','35','32','31','29','2','11','9','0','5','14','label'], axis=1)
    # Drop concatenated features
    X_test = test.drop(['16','7','56','57','35','32','31','29','2','11','9','0','5','14'], axis=1)
    # Drop concatenated features
    
    #X_factor_columns = list(set(X.columns.values.tolist()) - set(['59','60']))
    X_dum = pd.get_dummies(X,columns=['8','17','18','20','23','25','26','58'])
    X_dum_test = pd.get_dummies(X_test,columns=['8','17','18','20','23','25','26','58'])
    
    X_cols_to_use = list(X_dum.columns.difference(X_dum_test.columns))
    X_test_cols_to_use = list(X_dum_test.columns.difference(X_dum.columns))
	
    # FEATURE EXTRACTION
    pca = PCA(n_components = 0.8)
    # Number of P.C. decided on which number does best on holdout
    X_train_pca = pca.fit_transform(X_train)
    X_hold_pca = pca.transform(X_hold)
    X_test_pca = pca.transform(X_dum_test_2)
    
    # RUN MODEL WITH OPTIMIZED PARAMETERS
    clf = RandomForestClassifier(random_state=42,n_jobs=-1,
				n_estimators=167)
    # Variable for specific classifier (decided on which one does best on holdout)
    clf = clf.fit(X_train_pca, y_train)
    preds = clf.predict(X_test_pca)
    
    # SUBMISSION FILE
    idx = range(1,31710)
    submission = pd.DataFrame({"id": idx, "prediction": preds})
    submission.to_csv("some_ensemble_required_sub.csv", index=False)

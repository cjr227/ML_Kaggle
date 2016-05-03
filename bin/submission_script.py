#!/usr/bin/python

from __future__ import print_function
import csv
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sys import argv, exit, stderr


if __name__ == '__main__':
    if len(argv) < 4:
        print("Usage: python %s DATAFILE QUIZFILE OUTPUTFILE" % argv[0], file=stderr)
        exit(1)
    
    # LOAD DATA 
    train = pd.read_csv(argv[1])
    test = pd.read_csv(argv[2])
    
    n_test = np.shape(test)[0]
    
    # Extract data labels
    y = train.ix[:,-1].values
    y = y.astype(int)
    
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
    
    # RUN MODEL WITH OPTIMIZED PARAMETERS
    # Cross-validated random search over hyperparameter space resulted in 
    # random forest model with 167 estimators
    clf = RandomForestClassifier(random_state=42,n_jobs=-1,
                n_estimators=167)

    # Fit model on training dataset
    clf = clf.fit(X_dum_2, y)
    
    # Apply model on testing dataset
    preds = clf.predict(X_dum_test_2)
    
    # SUBMISSION FILE
    idx = range(1,31710)
    submission = pd.DataFrame({"id": idx, "prediction": preds})
    submission.to_csv(argv[3], index=False)

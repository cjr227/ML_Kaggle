#!/usr/bin/python

import numpy as np
from sklearn.cross_validation import train_test_split

# LOAD DATA
with open("featuresFactorized.npy/featuresFactorized.npy", "rb") as npy:
    X = np.load(npy)
# LOAD LABELS
with open("labels.npy", "rb") as npy:
    y = np.load(npy)

# SEPARATE DATA INTO TRAINING & HOLDOUT DATA
X_train, X_hold, y_train, y_hold = train_test_split(
     X, Y, test_size=0.3, random_state=42)
y_train = y_train.astype(int)
y_hold = y_hold.astype(int)

# Kaggle Competition for Machine Learning for Data Science
In an effort to reduce the size our data set, we decided to exclude features from our training data set.  These features were dropped because they either didn't tell us anything useful (ie all entries were 1) or they were redundant. There were 52 features in our original training set +1 column of labels. We dropped 14 of the 52 features, leaving us 38 feature vectors that describe the training data set. 

The following 14 features were dropped from our data set: 

features 0 & 9: concatenation represented in feature 18

features 2 & 11: concatenation represented in feature 20

features 7 & 16: concatenation represented in feature 25

features 56 & 57: concatenation represented in feature 58

features 5 & 14: concatenation represented in feature 25

features 29, 31, 32, 35: all entries were 1. There's nothing much we can learn from this.


SOME FEATURE DESCRIPTIONS:
Feature 18 is described here. 12 possible entries (13 including null)
http://groups.inf.ed.ac.uk/maptask/interface/expl.html

A nice tabular description of features 8 and 17 (both dialogue moves) can be found in Table 1 here.
http://people.cs.uchicago.edu/~dinoj/svmhmmda.pdf

our final submission ended up not using PCA and not using AdaBoost. we were able to get to  90%+ accuracy by doing manual feature reduction (see above documentation), limiting the features that were binarized, and only using the random forest algorithm. the features we binarized using panda's get_dummies were features 8,17,18,20,23,25,26 and 58.


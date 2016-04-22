# Kaggle Competition for Machine Learning for Data Science
In an effort to reduce the size our data set, we decided to exclude features from our training data set.  These features were dropped because they either didn't tell us anything useful (ie all entries were 1) or they were redundant. There were 52 features in our original training set +1 column of labels. We dropped 12 of the 52 features, leaving us 40 feature vectors that describe the training data set. 

The following 12 features were dropped from our data set: 

features 0 & 9: concatenation represented in feature 18

features 2 & 11: concatenation represented in feature 20

features 7 & 16: concatenation represented in feature 25

features 56 & 57: concatenation represented in feature 58

features 29, 31, 32, 35: all entries were 1. There's nothing much we can learn from this.


Feature descriptions:
Feature 18 is described here. 12 possible entries (13 including null)
http://groups.inf.ed.ac.uk/maptask/interface/expl.html


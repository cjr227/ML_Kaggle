# Kaggle Competition for Machine Learning for Data Science

This paper describes the data processing, feature reduction, model selection
through cross validation and finally presents the model with minimal prediction error
on the HCRC (Human Communication Research Centre) Map Task corpus dataset. This
experiment was conducted for the [Spring 2016 Machine Learning course (COMS 4721)
predictive modeling project] (https://inclass.kaggle.com/c/coms4721spring2016), where the final predictions were submitted to Kaggle for competition
among students. We attempted to reduce the feature dimensionality by applying
a combination of descriptive statistics and feature research for a manual feature selection
and Lasso Regression and PCA as reduction algorithms. The manual feature selection
turned out to be most successful for this task. After running a cross-validated random
search our final model consisted of a random forest with 167 decision trees that obtained
a prediction accuracy of 93.9% on the holdout dataset, 93.7% and 93.9% on the public and
private Kaggle validation datasets respectively.

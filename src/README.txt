1. To reproduce the prediction result submitted on kaggle, simply run bert.ipynb

2. To see performance of different classifier, run fit_sklearn.py


fit_sklearn.py:
	-include libraries import, 
	-data pre-processing, 
	-classifier fit, 
	-cross validation

cross_validation.py:
	-where algorithm of cross validation is implemented

naive_bayes.py:
	-our version of naive bayes classifier.

bert.ipynb:
	-special Bert model from ktrain library for prediction only.
	-details are commented inside this file. 
	-You may choose to run with kaggle kernel to speed up the training process. 
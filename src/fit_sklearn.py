import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit as tt
import seaborn as sns
import csv
import re
import functools
import string
# import graphviz
import nltk 
from nltk.corpus import stopwords
from nltk.stem import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm, tree, decomposition, ensemble, preprocessing
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.random_projection import sparse_random_matrix
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2,mutual_info_classif,RFE
from cross_validation import KFoldCrossValidation
from naive_bayes import BernoulliNaiveBayes

# nltk.download('wordnet')
pd.set_option('display.max_columns', None)
pd.set_option("display.precision", 3)

data_train = pd.read_csv('./data/reddit_train.csv')

train_com = data_train.comments
label = data_train.subreddits


stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')

# ##############################################################################################
# #                               Noise Reduction
# ##############################################################################################

def clean_document(text):
    
    text_ln = re.sub(r'^https?:\/\/.*[\r\n]*', '', text) #remove all links
    text_ln = re.sub(r'http\S+', '',text_ln)
    text_lc = "".join([word.lower() for word in text_ln if word not in string.punctuation])
    text_rc = re.sub('[0-9]+', '', text_lc) #substitute all numbers
    text_rc = re.sub(r'\W', ' ', text_rc)
    text_rc = re.sub(r'\s+[a-zA-Z]\s+', ' ',text_rc)
    text_rc = re.sub(r'\s+', ' ', text_rc)
    tokens = re.split('\W+', text_rc) #tokenize
    text = [stemmer.stem(word) for word in tokens if word not in stopwords] #remove stopwords and do stemming
    
    return text


# # X_train, X_test, Y_train, Y_test = train_test_split(train_com, label,test_size= 0.2)
X_train = train_com
Y_train = label

td_vec = TfidfVectorizer(
    max_features=50000, 
    binary=True, 
    ngram_range=(1, 1), 
    analyzer=clean_document)
vec_td = td_vec.fit(train_com.values.astype('U'))

X_train_td = td_vec.transform(X_train)
# # X_test_td = td_vec.transform(X_test)

print('{} reddit comments with feature size of {} words'.format(X_train_td.shape[0], X_train_td.shape[1]))

count_vec = CountVectorizer(
    max_features=10000, 
    binary=True, 
    ngram_range=(1, 1), 
    analyzer=clean_document)
vec_count = count_vec.fit(train_com.values.astype('U'))

X_train_cv = count_vec.transform(X_train)
# # X_test_cv = count_vec.transform(X_test)

print('{} reddit comments with feature size of {} words'.format(X_train_cv.shape[0], X_train_cv.shape[1]))

mi = SelectKBest(mutual_info_classif, k = 40000)
X_mi = mi.fit_transform(X_train_td, Y_train)

# # X_test_mi = mi.transform(X_test_td)



# # ch2_result = []
# # for n in np.arange(2500, 25000, 2500):
# #     ch2 = SelectKBest(chi2, k=n)
# #     x_train_chi2      = ch2.fit_transform(X_train, y_train)
# #     x_validation_chi2 = ch2.transform(X_test)
# #     clf = LogisticRegression()
# #     clf.fit(x_train_chi2, y_train)
# #     score = clf.score(x_validation_chi2, y_test)
# #     ch2_result.append(score)
# #     print ("chi2 feature selection finished for {} features".format(n))

# # ch2 = SelectKBest(chi2, k=12500)
# # x_train_ch2 = ch2.fit_transform(X_train, y_train)
# # x_val_ch2 = ch2.transform(X_test)


# ##############################################################################################
# #                              Cross Validation
# ##############################################################################################

l = LogisticRegression()    #scikit learn logReg
b = BernoulliNaiveBayes()   #self implemented Bnb
dt = tree.DecisionTreeClassifier()  #scikit learn decision tree

for i in k:

    Kfold = KFoldCrossValidation(i)
    print("Accuracy of {} fold validation with logReg model{}: ".format(i,Kfold.run_cross_validation(l, X_train_td, Y_train)))
    print("Accuracy of {} fold validation with bnb model{}: ".format(i,Kfold.run_cross_validation(b, X_train_td, Y_train)))
    print("Accuracy of {} fold validation with decision tree model{}: ".format(i,Kfold.run_cross_validation(dt, X_train_td, Y_train)))


# ##############################################################################################
# #                              LogReg
# ##############################################################################################
# logReg = LogisticRegression()
# print(logReg)


# # logReg.fit(X_train_td, Y_train)     #fit with td-idf 
# # y_pred = logReg.predict(X_test_td)
# # print(accuracy_score(Y_test, y_pred))
# # print(classification_report(Y_test,y_pred))

# # rfe = RFE(logReg, step = 1)         #fit with recursive featrue elimination
# # rfe.fit(X_train_td, Y_train)

# # y_pred = logReg.predict(X_test_td)
# # print(accuracy_score(Y_test, y_pred))
# # print(classification_report(Y_test,y_pred))

# # logReg.fit(X_mi, Y_train)           #fit with Mutual info vectorizer
# # y_pred = logReg.predict(X_test_mi)
# # print(accuracy_score(Y_test, y_pred))
# # print(classification_report(Y_test,y_pred))

# # logReg.fit(X_train_cv, Y_train)     #fit with count vectorizer

# # y_pred = logReg.predict(X_test_cv)
# # print(accuracy_score(Y_test, y_pred))
# # print(classification_report(Y_test,y_pred))

# # logReg.fit(X_train_ch2, Y_train)     #fit with chi square vectorizer

# # y_pred = logReg.predict(X_test_ch2)
# # print(accuracy_score(Y_test, y_pred))
# # print(classification_report(Y_test,y_pred))



# ##############################################################################################
# #                              Support vector machine
# ##############################################################################################
# svp = svm.SVC()
# print(svp)

# # svp.fit(X_train_td, Y_train)
# # y_pred = dt.predict(X_test_td)
# # print(accuracy_score(Y_test, y_pred))
# # print(classification_report(Y_test,y_pred))


# ##############################################################################################
# #                              Decision Tree
# ##############################################################################################


# dt = tree.DecisionTreeClassifier()
# print(dt)

# # dt = dt.fit(X_train_td,Y_train)
# # y_pred = dt.predict(X_test_td)
# # print(accuracy_score(Y_test, y_pred))
# # print(classification_report(Y_test,y_pred))

# ##############################################################################################
# #                              Random Forest Classifier
# ##############################################################################################


# # rf = ensemble.RandomForestClassifier()
# # rf.fit(X_train_td,Y_train)

# # y_pred = rf.predict(X_test_td)
# # print(accuracy_score(Y_test, y_pred))
# # print(classification_report(Y_test,y_pred))


# ##############################################################################################
# #                              Self-Implemented BNB
# ##############################################################################################

# bnb = BernoulliNaiveBayes()
# # bnb.fit(X_train_td,Y_train)

# # y_pred = bnb.predict(X_test_td)
# # print(accuracy_score(Y_test, y_pred))
# # print(classification_report(Y_test,y_pred))

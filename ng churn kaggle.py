# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 17:23:28 2018

@author: Dmob
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import turtle
#import time
import random
import dbm
import pickle
import copy
from collections import namedtuple
import re
import gensim
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")


#train_inputs = train_data.ix[:,0]
#train_labels = train_data.drop(0, axis=1)

DIR=r'C:\Users\Dmob\Desktop\ANDROID DEVELOPMENT\KAGGLE\kaggle models\churn ng data science'
train_data = pd.read_csv(DIR+'/train.csv', delimiter=',')
#test_data = pd.read_csv(DIR+'/test.csv', delimiter=',')


#print(train_data.head(4))
#print(test_data.head(2))

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

#df2 = pd.get_dummies(train_data)
#print(train_data.columns.values)  #print columns
train2=train_data.dropna()

#enc = preprocessing.OneHotEncoder()
#train3= enc.fit(train2) 
#enc.transform(train3).toarray()

le = preprocessing.LabelEncoder()
train3 = train2.apply(le.fit_transform)

x= train3.iloc[:,:-1]
y = train3['Churn Status']
#print(x1.head(5))
#x1 = x.fillna

#print(x.isnull().any().sum())
#print(y.isnull().any().sum())



#print(train_data.isnull().values.any())
#print(train_data.isnull().sum)
#print(train2.isnull().any().any())
#print(train2.isnull().T.any().T.sum())

 
#OneHotEncoder(categorical_features='all', dtype=<... 'numpy.float64'>,
 #      handle_unknown='error', n_values=[2, 3, 4], sparse=True)


pca = PCA(n_components=2).fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(pca, y, random_state=0)


'''
model = LogisticRegression()
model.fit(X_train,y_train)

#predictig test set results and creating cofsion matrix

y_pred = model.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
'''


from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1).fit(X_train, y_train)

svm_predicted = svm.predict(X_test)
confusion = confusion_matrix(y_test, svm_predicted)

print('SVM  classifier (default settings)\n', confusion)

from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, svm_predicted))








# In[ ]:
###########################
'''
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train,y_train)

plot_mushroom_boundary(X_test, y_test, model)


# In[ ]:

from sklearn.neural_network import MLPClassifier

model = MLPClassifier()
model.fit(X_train,y_train)

plot_mushroom_boundary(X_test, y_test, model)
'''

# In[ ]:





#sns.regplot(x=X_test[:,:-1],y= y_pred, data=train3, logistic=True)

#plt.figure(dpi=120)
#plt.scatter(X_test[:,0],y_pred, dpi=120)


















'''
plt.figure(dpi=120)
plt.scatter(pca[y.values==0,0], pca[y.values==0,1], alpha=0.5, label='YES', s=2, color='navy')
plt.scatter(pca[y.values==1,0], pca[y.values==1,1], alpha=0.5, label='NO', s=2, color='darkorange')
plt.legend()
plt.title('churn telecommunications')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.show()

print(y_pred.head(2))
'''







#x1 = x.values.reshape(-1,1)
#y1 = y.values.reshape (-1,1)
#(X_test, y_test, model)

#plt.scatter(pca[y.values==0,0], pca[y.values==0,1], alpha=0.5, label='YES', s=2, color='navy')
#plt.scatter(pca[y.values==1,0], pca[y.values==1,1], alpha=0.5, label='NO', s=2, color='darkorange')

#X_test2 = X_test.reshape(-1,1)
#y_pred2 = y_pred.reshape (-1,1)

####plt.scatter(X_test2,y_pred2)
#plt.figure()
#plt.show()
#####print(len(X_test))
####print(y_pred)
#print(y_pred.head(8))

















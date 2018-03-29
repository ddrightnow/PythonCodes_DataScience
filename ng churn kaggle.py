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

#train_inputs = train_data.ix[:,0]
#train_labels = train_data.drop(0, axis=1)

#DIR=r'C:\Users\Dmob\Desktop\ANDROID DEVELOPMENT\KAGGLE\kaggle models\churn ng data science'
#train_data = pd.read_csv(DIR+'/train.csv', delimiter=',')
#test_data = pd.read_csv(DIR+'/test.csv', delimiter=',')


#print(train_data.head(4))
#print(test_data.head(2))

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#df2 = pd.get_dummies(train_data)
#print(train_data.columns.values)  #print columns
#train2=train_data.dropna()

x= train2.iloc[:,:-1]
y = train2['Churn Status']
#print(x1.head(5))
#x1 = x.fillna

#print(x.isnull().any().sum())
#print(y.isnull().any().sum())



#print(train_data.isnull().values.any())
#print(train_data.isnull().sum)
#print(train2.isnull().any().any())
#print(train2.isnull().T.any().T.sum())

enc = preprocessing.OneHotEncoder()
enc.fit(x)  
#OneHotEncoder(categorical_features='all', dtype=<... 'numpy.float64'>,
 #      handle_unknown='error', n_values=[2, 3, 4], sparse=True)


pca = PCA(n_components=2).fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(pca, y, random_state=0)

model = LogisticRegression()
model.fit(X_train,y_train)

#(X_test, y_test, model)

plt.figure(dpi=120)
plt.scatter(pca[y.values==0,0], pca[y.values==0,1], alpha=0.5, label='Edible', s=2)
plt.scatter(pca[y.values==1,0], pca[y.values==1,1], alpha=0.5, label='Poisonous', s=2)
plt.legend()

















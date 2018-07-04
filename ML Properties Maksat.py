# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 18:35:11 2018

@author: Dmob
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import seaborn as sns
sns.set(style="white")
import os


from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


os.chdir(r'C:\Users\Dmob\Desktop\ML Maksat')
with open('FinalDataSheetWork.csv', 'rb') as csvfile:
     work = csv.reader(csvfile, delimiter=' ', quotechar='|')
     pdwork = pd.read_csv('FinalDataSheetWork.csv')
     
#print(pdwork.head(15))
print(pdwork.columns)

###  Fit Imputer

# Create an imputer object that looks for 'Nan' values, then replaces them with the mean value of the feature by columns (axis=0)
mean_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
# Train the imputor on the df dataset
pdwork["parking"]=mean_imputer.fit_transform(pdwork[["parking"]]).ravel()
pdwork["bathrooms"]=mean_imputer.fit_transform(pdwork[["bathrooms"]]).ravel()

#print(pdwork["parking"])
#print(pdwork.loc[[90602]])


#creating copy
pdwork2 = pdwork.copy()

#check -OK
#print(pdwork2['location'][20])
#print(pdwork2.loc[[5][2]])


#labelencode
#integer encode
label_encoder = LabelEncoder()
#pdwork2[:,1] = label_encoder.fit_transform(pdwork2[:,1])
pdwork2['location'] = label_encoder.fit_transform(pdwork2['location'])
pdwork2['furnishingLevel'] = label_encoder.fit_transform(pdwork2['furnishingLevel'])

#check - OK
#print(pdwork2.head(5))


pdwork3 = pdwork2.copy()

#heck for missing/null values - OK
#print(pdwork3.isnull().any(axis=0) )

y = pdwork3['price']
x = pdwork3.iloc[:,:-1]


# OneHotEncoder for Dataset
onehot_encoder = OneHotEncoder()
x = onehot_encoder.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

'''
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=0).fit(X_train,y_train)
ll = model.predict(X_test)
print ('rforest ',accuracy_score(y_test, ll))
'''

'''
Model_1 = ll.copy()

import pickle
filename = 'Model_1.sav'
pickle.dump(Model_1, open(filename, 'wb'))
'''





'''
###
from sklearn.neural_network import MLPClassifier

modelee = MLPClassifier([13,8,5],activation = 'logistic', alpha=0.1,early_stopping=True,
       epsilon=1e-08, random_state=0).fit(X_train,y_train)
ee = modelee.predict(X_test)
print ('neural ',accuracy_score(y_test, ee))
'''

###  

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=0).fit(X_train,y_train)
qq = model.predict(X_test)
print ('rforest ',accuracy_score(y_test, qq))








#print(X_train)


#PCA to remove sparseness of data



'''
#train_test_split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 50, random_state = 0)

#Standardize the data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Impute missing values at random
prop = int(X_train.size * 0.5) #Set the % of values to be replaced
prop1 = int(X_test.size * 0.5)
''' 
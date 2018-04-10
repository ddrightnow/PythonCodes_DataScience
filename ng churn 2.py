# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 15:17:10 2018

@author: Dmob
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")


DIR=r'C:\Users\Dmob\Desktop\ANDROID DEVELOPMENT\KAGGLE\kaggle models\churn ng data science'
train_data = pd.read_csv(DIR+'/train.csv', delimiter=',')
test_data = pd.read_csv(DIR+'/test.csv', delimiter=',')

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

train2=train_data.dropna()
#train2=train_data

le = preprocessing.LabelEncoder()
#train2['Network type subscription in Month 1'] = le.fit_transform(train2['Network type subscription in Month 1'].astype(str))
#train2['Network type subscription in Month 2'] = le.fit_transform(train2['Network type subscription in Month 2'].astype(str))

train2 = train2.apply(le.fit_transform)

#drop cus id/Network type subscription in Month 1
train4 = train2.drop(['Customer ID', 'Network type subscription in Month 1','network_age'], axis=1)

x= train4.iloc[:,:-1]
y = train4['Churn Status']


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

test_data['Network type subscription in Month 1'] = le.fit_transform(test_data['Network type subscription in Month 1'].astype(str))
test_data['Network type subscription in Month 2'] = le.fit_transform(test_data['Network type subscription in Month 2'].astype(str))

test_data2 = test_data.apply(le.fit_transform)
test_data22=test_data2.drop(['Customer ID', 'Network type subscription in Month 1','network_age'], axis=1)

# In[ ]:

from sklearn.ensemble import RandomForestClassifier


model = RandomForestClassifier(max_features=8,n_estimators =1,min_samples_split=2,
                                 min_samples_leaf=13,max_depth=10,
                                 criterion="gini",class_weight='balanced',
                                 random_state=0).fit(X_train,y_train)
ll = model.predict(X_test)
print ('rforest ',accuracy_score(y_test, ll))

kk = model.predict(test_data22)
kk2 = kk.copy()
s1 = pd.Series(kk2, name='Churn Status')
result2 = pd.concat([test_data, s1], axis=1)
result2.to_csv('TestA1.csv', sep=',', index=False)



model = RandomForestClassifier(random_state=0).fit(X_train,y_train)
ii = model.predict(X_test)
print ('rforest ',accuracy_score(y_test, ii))
pp3 = ii.copy()
s3 = pd.Series(pp3, name='Churn Status')
result4 = pd.concat([test_data, s3], axis=1)
result4.to_csv('TestA4rf.csv', sep=',', index=False)
##
    

model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print ('log ',accuracy_score(y_test, y_pred))


import pickle
filename = 'churn_telco_model.sav'
pickle.dump(model, open(filename, 'wb'))

###
'''
model22 = GaussianNB()
model22.fit(X_train,y_train)
aa= model22.predict(X_test)
print ('gaussian ',accuracy_score(y_test, aa))
s1a = pd.Series(aa, name='Churn Status')
result2 = pd.concat([test_data, s1a], axis=1)
result2.to_csv('TestA3.csv', sep=',', index=False)
####

from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=0.01, random_state=0).fit(X_train, y_train)

svm_predicted = svm.predict(X_test)
print ('svm ',accuracy_score(y_test, svm_predicted))


###
from sklearn.neural_network import MLPClassifier

modelee = MLPClassifier([13,8,5],activation = 'logistic', alpha=0.1, random_state=0).fit(X_train,y_train)
ee = modelee.predict(X_test)
print ('neural ',accuracy_score(y_test, ee))

'''






from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(random_state = 0).fit(X_train, y_train)
pp = clf.predict(test_data22)
print('gdbc',clf.score(X_test, y_test))
pp2 = pp.copy()
s2 = pd.Series(pp2, name='Churn Status')
result3 = pd.concat([test_data, s2], axis=1)
result3.to_csv('TestA2.csv', sep=',', index=False)


'''
0.86 model

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=0).fit(X_train,y_train)
ll = model.predict(X_test)
print ('rforest ',accuracy_score(y_test, ll))

kk = model.predict(test_data22)
kk2 = kk.copy()
s1 = pd.Series(kk2, name='Churn Status')
result2 = pd.concat([test_data, s1], axis=1)
result2.to_csv('TestA1.csv', sep=',', index=False)
'''
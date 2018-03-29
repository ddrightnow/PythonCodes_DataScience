# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 18:57:43 2018

@author: Dmob
"""

#Load KDD dataset

dataset = pandas.read_csv('Datasets/KDDCUP 99/kddcup.csv', names = ['duration','protocol_type','service','src_bytes','dst_bytes','flag','land','wrong_fragment','urgent',
'hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations',
'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','serror_rate',
'rerror_rate','same_srv_rate','diff_srv_rate','srv_count','srv_serror_rate','srv_rerror_rate','srv_diff_host_rate',
'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','class'])


# split data into X and y
array = dataset.values
X = array[:,0:41]
Y = array[:,41]

# Split-out validation dataset
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
num_folds = 7
num_instances = len(X_train)
seed = 7
scoring = 'accuracy'

#  Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, 

random_state=seed)

    #Here is where the error is spit out
{
            cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring) # Could not convert string to float happens here. Scoring uses string. 
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean()*100, cv_results.std()*100)#multiplying by 100 to show percentage
            print(msg)
}

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(Y)
plt.show()
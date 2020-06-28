#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 05:46:17 2020

@author: arushimadan
"""

import pandas as pd
import numpy as np

pd.pandas.set_option('display.max_columns',None)
pd.pandas.set_option('display.max_rows',None)

train = pd.read_csv('../Desktop/COVIDData.csv')
train.head()

data = train.copy()
data = data.drop(['Severity_None','None_Sympton','None_Experiencing','Contact_Dont-Know','Country','Contact_No'],axis = 1)
data.head()

data1 = data.copy()
data1 = data.drop(['Severity_Moderate','Severity_Mild'],axis = 1)
y_data = data1['Severity_Severe']
x_data = data1.drop(['Severity_Severe'],axis = 1)

SEED = 42
from sklearn.model_selection import train_test_split
X_train,X_val,Y_train,Y_val = train_test_split(x_data,y_data,test_size = 0.3,random_state = SEED)

X_train.head()


Y_train.head()


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
rf = RandomForestClassifier()
params = {"max_depth":[15,20,25], "n_estimators":[27,30,33]}
rf_reg = GridSearchCV(rf, params, cv = 10, n_jobs =10)
rf_reg.fit(X_train, Y_train)
print(rf_reg.best_estimator_)
best_estimator=rf_reg.best_estimator_
y_pred_train = best_estimator.predict(X_train)
y_pred_val = best_estimator.predict(X_val)
# rf.fit(X_train,Y_train)

from sklearn.metrics import confusion_matrix
confusion_matrix(Y_val,y_pred_val)

scoring = 'accuracy'
score = cross_val_score(rf,X_val,Y_val,cv = k_fold,n_jobs=1,scoring=scoring)
print(score)
type(score)

score.mean()


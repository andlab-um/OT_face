#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 08:44:01 2021

@author: yuanchenwang
"""

import math
import numpy as np
import pandas as pd
import prettytable as pt
from sklearn import preprocessing
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc
from mlxtend.feature_selection import SequentialFeatureSelector as sfs


def bootstrap(df):
    selectionIndex = np.random.randint(len(df), size = len(df))
    new_df = df.iloc[selectionIndex]
    return new_df


ques_data = pd.read_excel('/Users/yuanchenwang/Library/Mobile Documents/com~apple~CloudDocs/Projects/fmri/main/OT_questionnaire data.xlsx',
                          header=1)


ques_data['sub'] = ques_data['编号']%1000
ques_data = ques_data.drop(['编号','姓名'],axis=1)

predictors = ques_data.columns.values
predictors = predictors[predictors!='sub']

root = "/Users/yuanchenwang/Library/Mobile Documents/com~apple~CloudDocs/Projects/fmri"
behav_data = pd.read_excel(f'{root}/main/OT_fMRI_all data.csv.xls',sheet_name="OT_fMRI_all data")


nsub = ques_data.shape[0]
ques_data['child_acc']=np.zeros(nsub)
ques_data['adult_acc']=np.zeros(nsub)

for i in range(nsub):
    subject = ques_data['sub'][i]
    
    nc = behav_data[(behav_data['sub'] == subject)&((behav_data['face'] == 'oc')|(behav_data['face'] == 'sc'))].count()[0]
    cc = behav_data[(behav_data['sub'] == subject)&((behav_data['face'] == 'oc')|(behav_data['face'] == 'sc')) 
                    &( behav_data['STIM.CRESP']-1==behav_data['STIM.RESP'])].count()[0]
    na = behav_data[(behav_data['sub'] == subject)&((behav_data['face'] == 'oa')|(behav_data['face'] == 'sa'))].count()[0]
    ca = behav_data[(behav_data['sub'] == subject)&((behav_data['face'] == 'oa')|(behav_data['face'] == 'sa')) 
                    &( behav_data['STIM.CRESP']-1==behav_data['STIM.RESP'])].count()[0]
    ques_data['child_acc'][i] = cc/nc
    ques_data['adult_acc'][i] = ca/na
    
ques_data = ques_data.dropna(axis=0)



itrain, itest = train_test_split(np.arange(ques_data.shape[0]), test_size=0.2, random_state = 979)

dftrain = ques_data.iloc[itrain]
dftest = ques_data.iloc[itest]
response = ['child_acc']


# Xtrain = dftrain[predictors].values
# Xtrain = np.nan_to_num(Xtrain,0)
# Xtrain = preprocessing.normalize(Xtrain)
# Xtest = dftest[predictors].values
# Xtest = np.nan_to_num(Xtest,0)
# Xtest = preprocessing.normalize(Xtest)
# ytrain = dftrain[response].values
# ytrain = np.nan_to_num(ytrain,0)
# ytrain = np.reshape(ytrain,(1,46))
# ytest = dftest[response].values
# ytest = np.nan_to_num(ytest,0)

# clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

# # Build step forward feature selection
# sfs1 = sfs(clf,
#            k_features=10,
#            forward=True,
#            floating=False,
#            verbose=2,
#            scoring='accuracy',
#            cv=5)

# # Perform SFFS
# sfs1 = sfs1.fit(Xtrain, ytrain)



alphas = [0]
for i in range(-20,0):
    a = math.pow(10,i)
    alphas.append(a)

alphas = np.array(alphas)

meanCV = []
for alpha in alphas:
    X = dftrain[predictors].values
    X = np.nan_to_num(X,0)
    X = preprocessing.normalize(X)
    y = dftrain[response].values
    y = np.nan_to_num(y,0)
    # y = np.interp(y, (y.min(), y.max()), (0, +1))
    mlreg = Lasso(alpha=alpha,max_iter=1000)
    cv_scores = cross_val_score(mlreg,X,y,cv=5,scoring="neg_mean_squared_error")
    meanCV.append(np.mean(cv_scores))

alpha = alphas[np.argmax(meanCV)]

Xtrain = dftrain[predictors].values
Xtrain = np.nan_to_num(Xtrain,0)
Xtrain = preprocessing.normalize(Xtrain)
Xtest = dftest[predictors].values
Xtest = np.nan_to_num(Xtest,0)
Xtest = preprocessing.normalize(Xtest)
ytrain = dftrain[response].values
ytrain = np.nan_to_num(ytrain,0)
ytest = dftest[response].values
ytest = np.nan_to_num(ytest,0)
mlreg = Lasso(alpha=alpha,max_iter=1000)
mlreg.fit(Xtrain, ytrain)
ypred = mlreg.predict(Xtest)
mse = mean_squared_error(ytest,ypred)
r2 = r2_score(ytest,ypred)

print(f"MSE: {mse}, R2: {r2}")


tb = pt.PrettyTable()
tb.field_names = ["Predictor", "Coefficient"]
for i in range(len(predictors)):
    tb.add_row([predictors[i], mlreg.coef_[i]])
    # tb.add_row([predictors[i], mlreg.coef_[0][i], mlreg.coef_[1][i]])

print(tb)




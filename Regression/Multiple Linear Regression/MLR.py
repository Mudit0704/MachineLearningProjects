# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:03:48 2019

@author: Mudit Maheshwari
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('50_Startups.csv')

X = data.iloc[:, :-1].values
Y = data.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
X[:,3] = labelencoder_x.fit_transform(X[:,3])

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding Dummy Trap
X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

Y_pred = regressor.predict(X_test)

#Backward Elimination

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]

regressor_ols = sm.OLS(endog = Y, exog = X_opt).fit()

regressor_ols.summary()

X_opt = X[:, [0,1,3, 4, 5]]

regressor_ols = sm.OLS(endog = Y, exog = X_opt).fit()

regressor_ols.summary()


X_opt = X[:, [0,3, 4, 5]]

regressor_ols = sm.OLS(endog = Y, exog = X_opt).fit()

regressor_ols.summary()


X_opt = X[:, [0,3, 5]]

regressor_ols = sm.OLS(endog = Y, exog = X_opt).fit()

regressor_ols.summary()


X_opt = X[:, [0,3]]

regressor_ols = sm.OLS(endog = Y, exog = X_opt).fit()

regressor_ols.summary()


#automatic backward elimination

import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_ols = sm.OLS(Y, x).fit()
        maxVar = max(regressor_ols.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_ols.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_ols.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
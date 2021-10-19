# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 11:42:03 2019

@author: Mudit Maheshwari
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Position_Salaries.csv')

X = data.iloc[:,1:2].values
Y = data.iloc[:,2].values



from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X = sc_x.fit_transform(X)
Y= Y.reshape(-1,1)
sc_y = StandardScaler()
Y = sc_y.fit_transform(Y)

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')

regressor.fit(X,Y)

Y_pred = regressor.predict(sc_x.transform([[6.5]]))
Y_pred = sc_y.inverse_transform(Y_pred)
plt.scatter(X,Y,color = 'red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title("SVR")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
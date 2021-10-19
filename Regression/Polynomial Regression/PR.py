# -*- coding: utf-8 -*-
"""
Created on Thu May 30 09:28:17 2019

@author: Mudit Maheshwari
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Position_Salaries.csv')

X = data.iloc[:,1:2].values
Y = data.iloc[:,2].values


#Polynomial Regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 7)
X_poly = poly_reg.fit_transform(X)
polreg = LinearRegression()
polreg.fit(X_poly,Y)


#visualising Polynomial Regression
plt.scatter(X,Y,color='red')
plt.plot(X,polreg.predict(poly_reg.fit_transform(X)),color='blue')
plt.title("Polynomial Regression Model")
plt.xlabel("Levels")
plt.ylabel("Salaries")

#visualising Polynomial Regression(for higher resolution)

X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,Y,color='red')
plt.plot(X_grid,polreg.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title("Polynomial Regression Model")
plt.xlabel("Levels")
plt.ylabel("Salaries")

#predictiing new result using Polynomial Regression
polreg.predict(poly_reg.fit_transform([[6.5]]))
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 09:23:43 2019

@author: Mudit Maheshwari
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Social_Network_Ads.csv')


X = data.iloc[:,[2,3]].values
Y = data.iloc[:,4].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

#fitting of classifier
from sklearn.neighbors import KNeighborsClassifier
#creation of classifier
classifier = KNeighborsClassifier(n_neighbors = 5 , metric = 'minkowski', p = 2)
classifier.fit(X_train,Y_train)

Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)

from matplotlib.colors import ListedColormap
X_set, Y_set = sc_x.inverse_transform(X_train), Y_train
X1,X2 = np.meshgrid(np.arange(start = X_set[:,0 ].min()-5, stop = X_set[: , 0].max()+5, step = 0.1),np.arange(start = X_set[:, 1].min()-10000, stop = X_set[:, 1].max()+10000, step = 1000))

plt.contourf(X1,X2, classifier.predict(sc_x.transform(np.array([X1.ravel(),X2.ravel()]).T)).reshape(X1.shape), alpha=0.75, cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j,0], X_set[Y_set == j,1], c = ListedColormap(('red','green'))(i), label = j)
plt.title("KNN Train Set")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

from matplotlib.colors import ListedColormap
X_set, Y_set = sc_x.inverse_transform(X_test), Y_test
X1,X2 = np.meshgrid(np.arange(start = X_set[:,0 ].min()-5, stop = X_set[: , 0].max()+5, step = 0.1),np.arange(start = X_set[:, 1].min()-10000, stop = X_set[:, 1].max()+10000, step = 1000))

plt.contourf(X1,X2, classifier.predict(sc_x.transform(np.array([X1.ravel(),X2.ravel()]).T)).reshape(X1.shape), alpha=0.75, cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j,0], X_set[Y_set == j,1], c = ListedColormap(('red','green'))(i), label = j)
plt.title("KNN Test Set")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show() 

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 21:03:04 2019

@author: Mudit Maheshwari
"""
#Data Preprocessing



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Churn_Modelling.csv')


X = data.iloc[:,3:13].values
Y = data.iloc[:,13].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_x_1 = LabelEncoder()
X[:,1] = labelencoder_x_1.fit_transform(X[:,1])
labelencoder_x_2 = LabelEncoder()
X[:,2] = labelencoder_x_2.fit_transform(X[:,2])


from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

#splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

#importing keras library and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialising the ANN
classifier = Sequential()

#adding the i/p layer and first hidden layer
classifier.add(Dense(output_dim = 6, init='uniform', activation = 'relu', input_dim = 11))

#adding the second layer
classifier.add(Dense(output_dim = 6, init='uniform', activation = 'relu'))

#adding the o/p layer
classifier.add(Dense(output_dim = 1, init='uniform', activation = 'sigmoid'))

#compiling the ANN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

#fitting the ANN to dataset
classifier.fit(X_train,Y_train,batch_size=10,epochs = 100)

#making the predictions and evaluating the model

Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)


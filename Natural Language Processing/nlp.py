# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 12:42:08 2019

@author: Mudit Maheshwari
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("Restaurant_Reviews.tsv", delimiter = "\t", quoting =3)

#Cleaning the texts
import re
#import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',data['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    #Removing the unwanted words or useless words and stemming(loved~loving~love are same words) the words
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

#Creating the Bag Of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
Y = data.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.25, random_state = 0)

#creation of classifier
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
#fitting of classifier
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)


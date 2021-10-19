# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 13:04:42 2019

@author: Mudit Maheshwari
"""

#Building CNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initialising CNN
classifier = Sequential()

#Convolution Layer
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation = 'relu'))

#Pooling Layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Adding second convolution layer and pooling layer
classifier.add(Convolution2D(64, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(64, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))
#Flattening Layer
classifier.add(Flatten())

#Full Connection Layer
classifier.add(Dense(activation='relu',units = 128))
classifier.add(Dense(activation='softmax', units = 29))

#Compiling the CNN
classifier.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

#Fitting the CNN
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                                                    'dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory(
                                            'dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(
                    training_set,
                    steps_per_epoch=8000,
                    epochs=25,
                    validation_data=test_set,
                    validation_steps=2000)

#from keras.models import load_model

#classifier.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
#del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
#classifier = load_model('my_model.h5')

"""import cv2
import numpy as np
from keras import models
import sys
from PIL import Image
# Start capturing Video through webcam
video = cv2.VideoCapture(0)

while True:
    _, frame = video.read()
    kernel = np.ones((3,3),np.uint8)
     
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# define range of skin color in HSV
    lower_skin = np.array([0,20,70], dtype=np.uint8)
    upper_skin = np.array([20,255,255], dtype=np.uint8)
#extract skin colur image
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
#extrapolate the hand to fill dark spots within
    mask = cv2.dilate(mask,kernel,iterations = 4)
#blur the image
    mask = cv2.GaussianBlur(mask,(5,5),100)
    mask = cv2.resize(mask,(64,64))
    img_array = np.array(mask)
    #print(img_array.shape)
# Changing dimension from 128x128 to 128x128x3
    img_array = np.stack((img_array,)*3, axis=-1)
    #Our keras model used a 4D tensor, (images x height x width x channel)
    #So changing dimension 128x128x3 into 1x128x128x3 
    img_array_ex = np.expand_dims(img_array, axis=0)
    #print(img_array_ex.shape)
    #Calling the predict method on model to predict gesture in the frame
    result = classifier.predict(img_array_ex)
    prediction = ''
    if result[0][0] == 0:
        prediction = prediction+'cat'
    else:
        prediction = prediction+'dog'
    cv2.imshow("Capturing", frame)
    key=cv2.waitKey(1)
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
"""

#Prediction
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/dog.4016.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
"""if result[0][0] == 0:
 prediction = 'cat'
else:
 prediction = 'dog'
 """

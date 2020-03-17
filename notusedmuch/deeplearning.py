# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 11:13:49 2019

@author: U65988
"""
from sklearn.datasets import make_circles
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']= '2'

X,y=  make_circles(n_samples=1000, factor=.6, noise=.1, random_state=42)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=.3, random_state= 42)


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

model= Sequential()

model.add(Dense(4, input_shape= (2,), activation="tanh"))
model.add(Dense(4, activation="tanh"))
model.add(Dense(1, activation='sigmoid'))

model.compile(Adam(lr=.05), 'binary_crossentropy', metrics= ['accuracy'])
 
model.fit(X_train, y_train, epochs=100, verbose=1)

eval_result= model.evaluate(X_test, y_test)

print("Test Loss: ",eval_result[0], "  Test Accuracy: ", eval_result[1])

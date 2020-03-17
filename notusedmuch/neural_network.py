# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:50:11 2019

@author: U65988
"""

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense



dataset = loadtxt(r'C:\Users\u65988\Downloads\pima-indians-diabetes.data.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu')) #input layer: we give 12 neurons since we have 12 features
model.add(Dense(8, activation='relu')) #Hidden Layer
model.add(Dense(1, activation='sigmoid')) #Output Layer: will usually have only one neuron

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=250, batch_size=10)


, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
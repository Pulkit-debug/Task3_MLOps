#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import keras
import os

dataset=pd.read_csv("mobile price/train.csv")
dataset.head(10)

#To use data from the DataFrame for DL models, we convert the data into numpy arrays
X = dataset.iloc[:,:20].values
y = dataset.iloc[:,20:21].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
print(X)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

from keras.models import Sequential
from keras.layers import Dense

model=Sequential()
model.add(Dense(16,input_dim=20,activation='relu'))
model.add(Dense(12,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

fit_model=model.fit(X_train, y_train, epochs=100, batch_size=128)

accuracy = model.evaluate(X_test, y_test, verbose=0)
accuracy = accuracy[1]*100

file=open("accuracy.txt","w")
file.write("Accuracy of model is: "+str(accuracy))
file.close()


# In[ ]:





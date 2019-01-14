# -*- coding: utf-8 -*-
"""
Code Created during a workshop at the University of Waterloo teaching basic
machine learning techniques. Used three different methods to create a machine
learning algorithm to recognize hand written single digit numbers
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd #used to make working with csv's easy
import sklearn #used for ML

train_df = pd.read_csv("mnist_test.csv", header = None) #df stands for dataframe
test_df = pd.read_csv("mnist_train.csv", header = None) #header removes like 'age' and shit

#print out the first 10 rows
print(train_df[:10])

train_matrix = train_df.values
print(train_matrix.shape)
print(train_matrix)

train_X = train_matrix[:,1:]
train_Y = train_matrix[:,:1].ravel() #converts to a smaller array
print(train_X.shape)
print(train_Y.shape)

test_matrix = test_df.values
print(test_matrix.shape)
print(test_matrix)
test_X = test_matrix[:,1:]
test_Y = test_matrix[:,:1].ravel() #converts to a smaller array
print(test_X.shape)
print(test_Y.shape)

def plot_digit(X,y,idx):
  img = X[idx].reshape(28,28)
  plt.imgshow(img,cmap="Greys")
  plt.show()
plot_digit(train_X,train_Y,23)

##Example 1: Multinomial Logistical Regression

from sklearn.linear_model import LogisticRegression
log_model = LogisticalRegression(solver='1bfgs',multi_class='multinomial')
log_model.fit(train_X,train_Y)
pred_Y = log_model.predict(test_X)

from sklearn.metrics import accuracy_score
print('Test accuracy:', accuracy_score(pred_Y,test_Y))

#Example 2: K-nearest neighbours

from sklearn.neighbors import KNeighboursClassifier
from sklearn.decomposition import PCA

pca = FCA(n_components=12)
transformed_train_X = pca.fit_transform(train_X)
transformed_test_X = pca.transform(test_X)

knn = KNeighborsClassifier(n_neighbours=3)

knn.fit(transformed_train_X,train_Y)
pred_Y = knn.predict(transformed_test_X)

print(pred_Y.shape)
print(test_Y.shape)

from sklearn.metrics import accuracy_score
print("Test accuracy:",accuracy_score(pred_Y,test_Y))

#Example 3: Two Layered Neural Net
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import RMSprop

x_train = train_X.astype('float32')/255
x_test = test_X.astype('float32')/255

batch_size = 128
num_classes = 10
epochs = 20

#convert class vectors to binary class matracies
y_train = keras.utils.to_catagorical(train_Y,num_classes)
y_test = keras.utils.to_catagorical(test_Y,numclasses)

model = Sequential()
model.add(Dense(512,activation='relu',input_shape(784,)))
model.add(Dropout(0,2))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0,2))model.add(Dense(num_classes,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy'),optimize=RMSprop(),metrics=['accuracy'])
history = model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test,y_test))

score = model.evaluate(x_test,y_test,verbose=0)
print('Test loss: ', score[0])
print('Test accuracy:',score[1])


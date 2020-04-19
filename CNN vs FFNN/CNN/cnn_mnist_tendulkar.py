# -*- coding: utf-8 -*-
"""CNN_MNIST_Tendulkar.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gifwXom_6pq2qg5wlbd1grSumiqntkev
"""

#!pip install tensorflow-gpu==2.0.0-betal
import tensorflow as tf 
import datetime,os
from tensorflow.keras import datasets, layers, models
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print("Before reshaping",)
print("No.of.axis in training images",train_images.ndim)
print("No.of.axis in testing images",test_images.ndim)
print("Shape of training images",train_images.shape)
print("Shape of testing images",test_images.shape)

train_images = train_images.reshape((60000,28,28,1))
test_images = test_images.reshape((10000,28,28,1))

print("After reshaping",)
print("No.of.axis in training images",train_images.ndim)
print("No.of.axis in testing images",test_images.ndim)
print("Shape of training images",train_images.shape)
print("Shape of testing images",test_images.shape)

# Create convolutional base
model = models.Sequential()

model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape = (28,28,1)))

model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPool2D(2,2))

model.add(layers.Conv2D(64,(3,3),activation='relu'))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))



model.summary()

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')

model.fit(train_images,train_labels,epochs=5)

test_loss, test_acc = model.evaluate(test_images,test_labels)

print(test_acc)
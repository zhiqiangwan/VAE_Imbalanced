#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 14:45:55 2017

@author: cisa
"""
#standard library imports


#related third party imports
import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import h5py
import matplotlib.pyplot as plt

#local application/library specific imports

#input data processing
input_h5 = 'original_data.h5'
with h5py.File(input_h5,'r') as hf:
    tem = hf.get('train_refined_images')
    train_refined_images = np.array(tem) 
    tem = hf.get('train_refined_labels')
    train_refined_labels = np.array(tem)
    tem = hf.get('test_refined_images')
    test_refined_images = np.array(tem)
    tem = hf.get('test_refined_labels')
    test_refined_labels = np.array(tem)

num_classes = 2    
image_train = np.reshape(train_refined_images, (-1, 28, 28, 1))
label_train = keras.utils.to_categorical(train_refined_labels, num_classes)
image_test = np.reshape(test_refined_images, (-1, 28, 28, 1))
label_test = keras.utils.to_categorical(test_refined_labels, num_classes)
#plt.imshow(np.reshape(train_refined_images[1], (28,28)))

#parameter of the CNN
input_shape = (28, 28, 1)

batch_size = 128
epochs = 12

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()
#loss=keras.losses.categorical_crossentropy,
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(image_train, label_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(image_test, label_test))
#score = model.evaluate(image_test, label_test, verbose=0)


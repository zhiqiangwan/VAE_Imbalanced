#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 14:45:55 2017

@author: cisa
"""
#standard library imports
import os

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
import sklearn.metrics as sk_metric
import pandas as pd
from imblearn import metrics
import pickle

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

#local application/library specific imports

#'../data/affnist/data_50/' #
directory_generate_data = '../data/affnist/data_50_2500/' #'../data/affnist/data_384/' #'../data/affnist/data_128/' 
result_path = './results/'

GENERATE_DATA_TYPE = 'VAE' #'copy' # 'GAN'#
vae_hidden_size = 128
VAE_file = 'VAE_hidden_%d_generated_data.h5' % (vae_hidden_size) # 'VAE_generated_data.h5' #
#input data processing
input_h5 = os.path.join(directory_generate_data, 'original_data.h5')
with h5py.File(input_h5,'r') as hf:
    tem = hf.get('train_refined_images')
    train_refined_images = np.array(tem) 
    tem = hf.get('train_refined_labels')
    train_refined_labels = np.array(tem)
    tem = hf.get('test_refined_images')
    test_refined_images = np.array(tem)
    tem = hf.get('test_refined_labels')
    test_refined_labels = np.array(tem)

if GENERATE_DATA_TYPE == 'copy':
    generate_h5 = os.path.join(directory_generate_data, 'copy_data.h5') 
    with h5py.File(generate_h5,'r') as hf:
        tem = hf.get('copy_images')
        generate_images = np.array(tem)
        tem = hf.get('copy_labels')
        generate_labels = np.array(tem)      
elif GENERATE_DATA_TYPE == 'VAE':    
    generate_h5 = os.path.join(directory_generate_data, VAE_file)
    with h5py.File(generate_h5,'r') as hf:
        tem = hf.get('VAE_images')
        generate_images = np.array(tem)
        tem = hf.get('VAE_labels')
        generate_labels = np.array(tem)    
elif GENERATE_DATA_TYPE == 'GAN':
    generate_h5 = os.path.join(directory_generate_data, 'GAN_generated_data.h5')
    with h5py.File(generate_h5,'r') as hf:
        tem = hf.get('GAN_images')
        generate_images = np.array(tem)
        tem = hf.get('GAN_labels')
        generate_labels = np.array(tem)         

#plt.imshow(np.reshape(generate_images[182+2450*4,:], (28, 28)), cmap='gray')
#plt.axis('off')
        
num_classes = 10
batch_size = 128
epochs = 50
CLASSIFY_TYPE = 'CNN'  #'MLP'# MLP or CNN

accuracy_list = []
sup_list = []
iba_list = []
geo_list = []
f1_list = []
spe_list = []
red_list = []
pre_list = []

with tf.device('/gpu:1'):
    for metric_loop in range(100): 
        print 'This is ' + str(metric_loop) + 'run'
        
        if CLASSIFY_TYPE == 'MLP':
            image_train = np.concatenate( (train_refined_images, generate_images), axis=0)
            label_train = keras.utils.to_categorical(np.concatenate( (train_refined_labels, generate_labels) ), num_classes)
            image_test = test_refined_images
            label_test = keras.utils.to_categorical(test_refined_labels, num_classes)
            
            model = Sequential()
            model.add(Dense(126, activation='relu', input_shape=(784,)))
            model.add(Dropout(0.2))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.2))     
            model.add(Dense(num_classes, activation='softmax'))
            model.summary()
          
        elif CLASSIFY_TYPE == 'CNN':    
            image_shape_for_train = (-1, 28, 28, 1)  #   
            image_train = np.reshape( np.concatenate( (train_refined_images, generate_images), axis=0) , image_shape_for_train)
            label_train = keras.utils.to_categorical( np.concatenate( (train_refined_labels, generate_labels) ) , num_classes)
            image_test = np.reshape(test_refined_images, image_shape_for_train)
            label_test = keras.utils.to_categorical(test_refined_labels, num_classes)
            
            input_shape = image_shape_for_train[1:]#(28, 28, 1)       
        
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
        
        
        #plt.imshow(image_train[1100])
        
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        
        model.fit(image_train, label_train, batch_size=batch_size, epochs=epochs,
                  verbose=0, validation_data=(image_test, label_test))

        accuracy_list.append(model.evaluate(image_test, label_test, verbose=0)[1]) 
        predict_output = np.argmax( model.predict(image_test), axis=1 )
        imb_results = metrics.classification_report_imbalanced(test_refined_labels, predict_output, digits=4)
        imb_results = imb_results.replace(" ", "")
        sup_list.append(float(imb_results[-6:-1]))
        iba_list.append(float(imb_results[-12:-6]))
        geo_list.append(float(imb_results[-18:-12]))
        f1_list.append(float(imb_results[-24:-18]))
        spe_list.append(float(imb_results[-30:-24]))
        red_list.append(float(imb_results[-36:-30]))
        pre_list.append(float(imb_results[-42:-36]))
        
        K.clear_session()


#predict_output = np.argmax( model.predict(image_test), axis=1 )

groundtruth_pd = pd.Series(test_refined_labels, name="groundtruth")
pred_pd = pd.Series(predict_output, name="predicted")
df_confusion = pd.crosstab(groundtruth_pd, pred_pd, margins=True)

averg_acc = sum(accuracy_list)/float(len(accuracy_list))
averg_pre = sum(pre_list)/float(len(pre_list))
averg_red = sum(red_list)/float(len(red_list))
averg_spe = sum(spe_list)/float(len(spe_list))
averg_f1 = sum(f1_list)/float(len(f1_list))
averg_geo = sum(geo_list)/float(len(geo_list))
averg_iba = sum(iba_list)/float(len(iba_list))

print('averg_acc:', averg_acc, 'averg_pre:', averg_pre, 'averg_red:', averg_red, 
      'averg_spe:', averg_spe, 'averg_f1:', averg_f1, 'averg_geo:', averg_geo, 'averg_iba:', averg_iba)

with open( os.path.join(result_path, 'VAE_result_100runs.pkl'), 'w' ) as f:
    pickle.dump([accuracy_list, pre_list, red_list, spe_list, f1_list, geo_list, iba_list], f)

with open( os.path.join(result_path, 'VAE_result_100runs.pkl'), 'r' ) as f:
    VAE_results = pickle.load(f)
    
    

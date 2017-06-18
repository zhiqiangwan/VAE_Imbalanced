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
#import sklearn.metrics as sk_metric
import pandas as pd

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True

from tensorflow.contrib import layers
from imblearn import metrics


#local application/library specific imports

#'../data/affnist/data_50/' #
directory_generate_data = '../data/affnist/data_50/' #'../data/affnist/data_384/' #'../data/affnist/data_128/' 

GENERATE_DATA_TYPE = 'VAE' #'copy' #'GAN'# 
COMBINED_LABEL = ['label_0', 'label_1', 'label_2', 'label_3', 'label_4']
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

if type(COMBINED_LABEL) is list:
    if GENERATE_DATA_TYPE == 'copy':
        generate_h5 = os.path.join(directory_generate_data, 'copy_data.h5') 
        with h5py.File(generate_h5,'r') as hf:
            tem = hf.get('copy_images')
            generate_images = np.array(tem)
            tem = hf.get('copy_labels')
            generate_labels = np.array(tem)      
    elif GENERATE_DATA_TYPE == 'VAE': 
        for i, label_file in enumerate(COMBINED_LABEL):
            generate_h5 = os.path.join(directory_generate_data, label_file, VAE_file)
            if i == 0:
                with h5py.File(generate_h5,'r') as hf:
                    tem = hf.get('VAE_images')
                    generate_images = np.array(tem)
                    tem = hf.get('VAE_labels')
                    generate_labels = np.array(tem)
                    
            else:
                with h5py.File(generate_h5,'r') as hf:
                    tem = hf.get('VAE_images')
                    generate_images = np.concatenate( (generate_images, np.array(tem)), axis=0 ) 
                    tem = hf.get('VAE_labels')
                    generate_labels = np.concatenate( (generate_labels, np.array(tem) ), axis=0 )                
           
    elif GENERATE_DATA_TYPE == 'GAN':
        generate_h5 = os.path.join(directory_generate_data, 'GAN_generated_data.h5')
        with h5py.File(generate_h5,'r') as hf:
            tem = hf.get('GAN_images')
            generate_images = np.array(tem)
            tem = hf.get('GAN_labels')
            generate_labels = np.array(tem)         
else:
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
    
#plt.imshow(np.reshape(generate_images[2122+2450*4,:], (28, 28)))
        
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

IS_KERAS = True #False# 


for metric_loop in range(10):   
    if IS_KERAS:
        set_session(tf.Session(config=config))
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
                  verbose=1, validation_data=(image_test, label_test))
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
        
 
    else:
        sess = tf.InteractiveSession(config=config)
        if CLASSIFY_TYPE == 'CNN':
            image_shape_for_train = (-1, 28, 28, 1)  # 
            image_train = np.reshape( np.concatenate( (train_refined_images, generate_images), axis=0) , image_shape_for_train)
            label_train = np.concatenate( (train_refined_labels, generate_labels) )
            image_test = np.reshape(test_refined_images, image_shape_for_train)
            label_test = test_refined_labels
            
            input_tf = tf.placeholder(tf.float32, [None, 28, 28, 1])
            labels_tf = tf.placeholder( tf.int32, [None], name='labels')
            learning_rate = tf.placeholder( tf.float32, [])
            keep_prob1 = tf.placeholder(tf.float32)
            keep_prob2 = tf.placeholder(tf.float32)
            
            net1 = layers.conv2d(input_tf, 32, 3, stride=1, padding='VALID')
            net2 = layers.conv2d(net1, 64, 3, stride=1, padding='VALID')
            net3 = layers.max_pool2d(net2, 2, 2)
            net4 = tf.nn.dropout(net3, keep_prob1)
            net5 = layers.flatten(net4)
            net6 = layers.fully_connected(net5, 128)
            net7 = tf.nn.dropout(net6, keep_prob2)
            output = layers.fully_connected(net7, num_classes, None)
            
            tf_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits( logits=output, labels=labels_tf ))
            train_op = tf.train.AdadeltaOptimizer(learning_rate).minimize(tf_loss)
            tf.initialize_all_variables().run()


            
            print 'start training...'  
            init_learning_rate = 1.0           
            loss_list = []
            loss_list_test = []
            iterations = 0
            
            num_train = image_train.shape[0]
            num_test = image_test.shape[0]
            result = []
            
            #start = 0
            #end = 10    
            
            for epoch in range(epochs):
                
                trainset_shuffle_index = np.random.permutation(num_train)
                    
                for start, end in zip( range( 0, num_train+batch_size, batch_size ),
                                      range( batch_size, num_train+batch_size, batch_size ) ):
                   if end > num_train: end = num_train
                   
                   current_images = image_train[trainset_shuffle_index[start:end]]
                   current_label = label_train[trainset_shuffle_index[start:end]]
                   
                   _,loss_val, output_val = sess.run(
                           [train_op, tf_loss, output],
                           feed_dict={ 
                           learning_rate : init_learning_rate,
                           input_tf : current_images,
                           labels_tf : current_label,
                           keep_prob1 : 0.75,
                           keep_prob2 : 0.5
                           } )
                   
                   loss_list.append(loss_val)
                   
                   iterations += 1
                   if iterations % 200 == 0:
                       print "======================================"
                       print "Epoch", epoch, "Iteration", iterations
                       print "Processed", start, '/', num_train
                       
            
                       label_predictions = output_val.argmax(axis=1)
                       acc = (label_predictions == current_label).sum()
                       print "Accuracy:", acc, '/', len(current_label)           
                       print "Training Loss:", np.mean(loss_list)
                       print "\n"
                       loss_list = []      
            
                print "start test"            
                num_correct = 0
                
                predict_output = np.zeros_like(label_test)
                for start, end in zip(
                        range(0, num_test+batch_size, batch_size),
                        range(batch_size, num_test+batch_size, batch_size)
                        ):
                    if end > num_test: end = num_test
            
                    current_images = image_test[start:end]
                    current_label = label_test[start:end]
            
                    val_output_test, val_loss_test = sess.run([output, tf_loss], 
                                                              feed_dict={
                                                                         keep_prob1:1, 
                                                                         keep_prob2:1,
                                                                         input_tf : current_images,
                                                                         labels_tf : current_label
                                                                         })
            
                    label_pred_test = val_output_test.argmax(axis=1)
                    predict_output[start:end] = label_pred_test
                    acc = (label_pred_test == current_label).sum()
                    num_correct += acc
                    loss_list_test.append(val_loss_test)
            
                acc_all = num_correct/float(num_test)            
            
                print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                print 'epoch:'+str(epoch)+'\tacc:'+str(acc_all) + '\n'
                print 'testlost:'+str(np.mean(loss_list_test)) + '\n'
                print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
                
                imb_results = metrics.classification_report_imbalanced(label_test, predict_output)
                print imb_results
            
                loss_list_test = []                      
                init_learning_rate *= 0.99
              
            print "end of training"           
                                                                
            tf.reset_default_graph()   
          
groundtruth_pd = pd.Series(test_refined_labels, name="groundtruth")
pred_pd = pd.Series(predict_output, name="predicted")
df_confusion = pd.crosstab(groundtruth_pd, pred_pd, margins=True)


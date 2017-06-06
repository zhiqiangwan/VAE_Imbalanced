'''TensorFlow implementation of http://arxiv.org/pdf/1312.6114v10.pdf'''

from __future__ import absolute_import, division, print_function

import math
import os

import numpy as np
import scipy.misc
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.contrib.framework import arg_scope
from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
import h5py
import matplotlib.pyplot as plt

import keras
from keras.datasets import cifar10
from keras import backend as K

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

import glob
import scipy.io as spio

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("updates_per_epoch", 1000, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 100, "max epoch")
flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
flags.DEFINE_string("working_directory", "./", "")
flags.DEFINE_integer("hidden_size", 128, "size of the hidden VAE unit")
flags.DEFINE_string("model", "vae", "gan or vae")
flags.DEFINE_string("generate_size", 4600, "batch size of generated images")

FLAGS = flags.FLAGS

#'../data/affnist/data_50/' #'../data/affnist/data_1152/' #
directory_generate_data = '../data/affnist/data_384/' #'../data/affnist/data_128/'
if not os.path.exists(directory_generate_data):
    os.makedirs(directory_generate_data)

dataset_path = '../dataset/affnist/'    
    
##########################load affNIST data########################

# train_data=np.array([])
def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict
    
dataset = loadmat(os.path.join( dataset_path, '1.mat' ))
y_train = dataset['affNISTdata']['label_int']
x_train_0 = dataset['affNISTdata']['image'].transpose() 
x_train = np.zeros((x_train_0.shape[0], 28*28), dtype=np.uint8)  

for k in range(x_train.shape[0]):
    resized_image = scipy.misc.imresize(x_train_0[k].reshape(40, 40), (28, 28))
    x_train[k, :] = resized_image.reshape(1, 28*28)
x_train = x_train / 255.0
    
    
test_data = loadmat(os.path.join( dataset_path, 'test.mat' ))
x_test_0 = test_data['affNISTdata']['image'].transpose()[:10000]
y_test = test_data['affNISTdata']['label_int'][:10000]
x_test = np.zeros((x_test_0.shape[0], 28*28), dtype=np.uint8) 

for k in range(x_test.shape[0]):
    resized_image = scipy.misc.imresize(x_test_0[k].reshape(40, 40), (28, 28))
    x_test[k, :] = resized_image.reshape(1, 28*28)
x_test = x_test / 255.0    

#plt.imshow(np.reshape(x_train[2,:], (28, 28)))
#plt.imshow(resized_image)


refined_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
num_minority_label = 384#128#50#1152#
num_majority_label = 5000
num_train_per_label = [num_minority_label, num_majority_label]*5
train_refined_label_idx = np.array([], dtype = np.uint8)
test_refined_label_idx = np.array([], dtype = np.uint8)
for idx, label_value in enumerate(refined_label):
    refined_one_label_idx = np.where( y_train == label_value )[0][:num_train_per_label[idx]]
    train_refined_label_idx = np.append( train_refined_label_idx,  refined_one_label_idx)
    test_refined_one_label_idx = np.where( y_test == label_value )[0]
    test_refined_label_idx = np.append(test_refined_label_idx, test_refined_one_label_idx)



train_refined_images = x_train[train_refined_label_idx, :]
train_refined_labels = y_train[train_refined_label_idx]
test_refined_images = x_test[test_refined_label_idx, :]
test_refined_labels = y_test[test_refined_label_idx]


f = h5py.File(os.path.join(directory_generate_data, 'original_data.h5'), "w")
f.create_dataset("train_refined_images", dtype='float32', data=train_refined_images)
f.create_dataset("train_refined_labels", dtype='uint8', data=train_refined_labels)
f.create_dataset("test_refined_images", dtype='float32', data=test_refined_images)
f.create_dataset("test_refined_labels", dtype='uint8', data=test_refined_labels)
f.close()

#
#
#plt.imshow(np.reshape(train_refined_images[500,:], (28, 28)),)


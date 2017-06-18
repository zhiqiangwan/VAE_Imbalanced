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
#from progressbar import ETA, Bar, Percentage, ProgressBar
import cv2

import keras
from keras.datasets import cifar10
from keras import backend as K

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("updates_per_epoch", 1000, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 100, "max epoch")
flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
flags.DEFINE_string("working_directory", "./", "")
flags.DEFINE_integer("hidden_size", 128, "size of the hidden VAE unit")
flags.DEFINE_string("model", "vae", "gan or vae")
flags.DEFINE_string("generate_size", 2450, "batch size of generated images")

FLAGS = flags.FLAGS

directory_generate_data = '../data/cifar_gray/data_50/' #'../data/cifar/data_50/label_0/' #
if not os.path.exists(directory_generate_data):
    os.makedirs(directory_generate_data)

##########################load cifar data########################
(x_train_0, y_train), (x_test_0, y_test) = cifar10.load_data()


x_train = np.zeros((x_train_0.shape[0], 32, 32))
for i in range(x_train_0.shape[0]):
    x_train[i] = cv2.cvtColor(x_train_0[i], cv2.COLOR_RGB2GRAY)

x_test = np.zeros((x_test_0.shape[0], 32, 32))
for i in range(x_test_0.shape[0]):
    x_test[i] = cv2.cvtColor(x_test_0[i], cv2.COLOR_RGB2GRAY)

#plt.imshow(x_train[60], cmap='gray')

x_train = x_train / 255.0
x_test = x_test / 255.0

refined_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
num_minority_label = 50#128#50#1152#
num_majority_label = 2500
num_train_per_label = [num_minority_label]*5 + [num_majority_label]*5  #+ [num_majority_label]*1
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

#plt.imshow(train_refined_images[8000+50*0])


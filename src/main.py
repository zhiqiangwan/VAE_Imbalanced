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

from progressbar import ETA, Bar, Percentage, ProgressBar

from vae import VAE
from gan import GAN

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("updates_per_epoch", 1000, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 1000, "max epoch")
flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
flags.DEFINE_string("working_directory", "./", "")
flags.DEFINE_integer("hidden_size", 128, "size of the hidden VAE unit")
flags.DEFINE_string("model", "vae", "gan or vae")
flags.DEFINE_string("generate_size", 3000, "batch size of generated images")

FLAGS = flags.FLAGS


data_directory = os.path.join(FLAGS.working_directory, "MNIST")
if not os.path.exists(data_directory):
    os.makedirs(data_directory)
mnist_all = input_data.read_data_sets(data_directory, one_hot=False, validation_size = 0)

mnist_train_images = mnist_all.train.images
mnist_train_labels = mnist_all.train.labels

refined_label = [6]#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
num_train_per_label = 400
train_refined_label_idx = np.array([], dtype = np.uint8)
for idx, label_value in enumerate(refined_label):
    refined_one_label_idx = np.where( mnist_train_labels == label_value )[0][:num_train_per_label]
    train_refined_label_idx = np.append( train_refined_label_idx,  refined_one_label_idx)

train_refined_images = mnist_train_images[train_refined_label_idx, :]


#import matplotlib.pyplot as plt
#
#plt.imshow(np.reshape(gener_image[4,:], (28, 28)),)

assert FLAGS.model in ['vae', 'gan']
if FLAGS.model == 'vae':
    model = VAE(FLAGS.hidden_size, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.generate_size)
elif FLAGS.model == 'gan':
    model = GAN(FLAGS.hidden_size, FLAGS.batch_size, FLAGS.learning_rate)

num_train = train_refined_label_idx.shape[0]
loss_list = []
iterations = 0

for epoch in range(FLAGS.max_epoch):

    trainset_shuffle_index = np.random.permutation(num_train)
        
    for start, end in zip( range( 0, num_train+FLAGS.batch_size, FLAGS.batch_size ),
                          range( FLAGS.batch_size, num_train+FLAGS.batch_size, FLAGS.batch_size ) ):
       if end > num_train: end = num_train
       
       images = train_refined_images[trainset_shuffle_index[start:end], :]

       loss_value = model.update_params(images)
       loss_list.append(loss_value)
       
       iterations += 1
       if iterations % 50 == 0:
           print("======================================")
           print("Epoch", epoch, "Iteration", iterations) 
           
           print ("Training Loss:", np.mean(loss_list))
           print ("\n")
           loss_list = []      
    if epoch % 200 == 0 or epoch == (FLAGS.max_epoch-1):       
        model.generate_and_save_images(
            FLAGS.batch_size, FLAGS.working_directory)


gener_image = model.sess.run(model.sampled_tensor_gener)


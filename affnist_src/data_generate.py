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

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from vae import VAE
from gan import GAN

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("updates_per_epoch", 1000, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 8000, "max epoch")
flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
flags.DEFINE_string("working_directory", "./", "")
flags.DEFINE_integer("hidden_size", 128, "size of the hidden VAE unit")
flags.DEFINE_string("model", "vae", "gan or vae")
flags.DEFINE_string("generate_size", 2450, "batch size of generated images")



FLAGS = flags.FLAGS

#'../data/affnist/data_50/' #
directory_generate_data = '../data/affnist/data_50_2500/' #'../data/affnist/data_384/' #'../data/affnist/data_128/' #
if not os.path.exists(directory_generate_data):
    os.makedirs(directory_generate_data)

#data_directory = '../data/affnist/data_384/'
#if not os.path.exists(data_directory):
#    os.makedirs(data_directory)

#input data processing
input_h5 = os.path.join(directory_generate_data, 'original_data.h5')
with h5py.File(input_h5,'r') as hf:
    tem = hf.get('train_refined_images')
    x_train = np.array(tem) 
    tem = hf.get('train_refined_labels')
    y_train = np.array(tem)
     

refined_label = [0, 1, 2, 3, 4]#[0]#[9]#[0, 2, 4, 6, 8]#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#num_train_per_label = [384] #should be times of batch_size, 128*3=384                      

for idx, label_value in enumerate(refined_label):
    
    refined_one_label_idx = np.where( y_train == label_value )[0]

    train_refined_images = x_train[refined_one_label_idx, :]
    train_refined_labels = y_train[refined_one_label_idx]

    assert FLAGS.model in ['vae', 'gan']
    if FLAGS.model == 'vae':
        model = VAE(FLAGS.hidden_size, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.generate_size)
    elif FLAGS.model == 'gan':
        model = GAN(FLAGS.hidden_size, FLAGS.batch_size, FLAGS.learning_rate, FLAGS.generate_size)
   
    num_train = train_refined_images.shape[0]
    loss_list = []
    g_loss_list = []
    d_loss_list = []
    iterations = 0
    generator_update_freq = 4
    d_loss_avrag = 10.0
    
    for epoch in range(FLAGS.max_epoch):
    
        trainset_shuffle_index = np.random.permutation(num_train)
            
        for start, end in zip( range( 0, num_train+FLAGS.batch_size, FLAGS.batch_size ),
                              range( FLAGS.batch_size, num_train+FLAGS.batch_size, FLAGS.batch_size ) ):
           if end > num_train: end = num_train
           
           images = train_refined_images[trainset_shuffle_index[start:end], :]
           
           if FLAGS.model == 'vae':
               loss_value = model.update_params(images)
               loss_list.append(loss_value)           
               iterations += 1
               if iterations % 500 == 0:
                   print("======================================")
                   print("Epoch", epoch, "Iteration", iterations)                    
                   print ("Training Loss:", np.mean(loss_list))
                   print ("\n")
                   loss_list = []
           elif FLAGS.model == 'gan':
               g_loss_value, d_loss_value = model.update_params(images, generator_update_freq)
               g_loss_list.append(g_loss_value)
               d_loss_list.append(d_loss_value)
               iterations += 1
               if iterations % 100 == 0:
                   g_loss_avrag = np.mean(g_loss_list)
                   d_loss_avrag = np.mean(d_loss_list)
                   print("======================================")
                   print("Epoch", epoch, "Iteration", iterations)
                   print ("Training generator Loss:", g_loss_avrag)
                   print ("Training discriminator Loss:", d_loss_avrag)
                   generator_update_freq = 10*int(g_loss_avrag/d_loss_avrag)
                   if generator_update_freq > 30:
                       generator_update_freq = 30
                   elif generator_update_freq < 4:
                       generator_update_freq = 4
                   print(generator_update_freq)
                   print ("\n")
                   g_loss_list = []
                   d_loss_list = []
                   
                   
               
        if epoch % 400 == 0 or epoch == (FLAGS.max_epoch-1):       
            model.generate_and_save_images(
                FLAGS.batch_size, FLAGS.working_directory)
        
        if FLAGS.model == 'gan':
            if d_loss_avrag < 0.04:
                break
            
    
    gener_image_per_label = model.sess.run(model.sampled_tensor_gener)
    gener_labels_per_label = label_value * np.ones((gener_image_per_label.shape[0]), dtype=np.uint8)
    if idx == 0:
        gener_image = gener_image_per_label[:]
        gener_label = gener_labels_per_label[:]
    else:
        gener_image = np.concatenate( (gener_image, gener_image_per_label), axis=0)
        gener_label = np.concatenate( (gener_label, gener_labels_per_label), axis=0 )

    tf.reset_default_graph() 

if FLAGS.model == 'vae':    
    f = h5py.File(os.path.join(directory_generate_data, 'VAE_hidden_%d_generated_data.h5' % (FLAGS.hidden_size)), "w")
    f.create_dataset("VAE_images", dtype='float32', data=gener_image)
    f.create_dataset("VAE_labels", dtype='uint8', data=gener_label)
    f.close()
elif FLAGS.model == 'gan':
    f = h5py.File(os.path.join(directory_generate_data, 'GAN_generated_data.h5'), "w")
    f.create_dataset("GAN_images", dtype='float32', data=gener_image)
    f.create_dataset("GAN_labels", dtype='uint8', data=gener_label)
    f.close()    
#hh = refined_label*np.ones((gener_image.shape[0]), dtype=np.uint8)
#plt.imshow(np.reshape(gener_image[1+4600*4,:], (28, 28)),)

#aa = model.sess.run(model.one_sampled_tensor_gener)
#plt.imshow(np.reshape(aa, (28, 28)),)
#similarity = model.sess.run([model.similarity], {model.input_tensor: images})
#print(similarity)
#aa1, aa2, aa3 = model.sess.run([model.one_sample_encoded, model.one_sample_encoded_tile, model.similarity], {model.input_tensor: images})


#aa_loss = model.sess.run(model.one_sample_vae_loss)
#print(aa_loss)


import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
import cifar_loader as loader
import numpy as np
import matplotlib.pyplot as plt
import os
from tf_layers import *
import utils

from tensorflow.contrib.layers import batch_norm

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
CHANNEL = 3
BATCH_SIZE = 64
Z_DIM = 100

x = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNEL])
z = tf.placeholder(tf.float32, shape=[None, Z_DIM])
is_training = tf.placeholder(tf.bool)
keep_prob = tf.placeholder(tf.float32)

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def sample_Z(m, n):
    return np.random.normal(loc=0.0, scale=1.0, size=[m, n])
    #return np.random.uniform(-1., 1., size=[m, n])

def flatten(x):
  shape = x.get_shape().as_list()
  dim = np.prod(shape[1:]) 
  flatten_x = tf.reshape(x, [-1, dim])
  
  return flatten_x

def generator(z):
  with tf.variable_scope('G', initializer=tf.truncated_normal_initializer(stddev=0.02)) as scope:
    G_h1 = fully_connected(z, output_dim=4*4*256, activation_fn=tf.nn.relu, keep_prob=None, scope='fc_1')

    bn_1 = batch_norm(G_h1, updates_collections = None, is_training=is_training)

    reshape_G_h1 = tf.reshape(bn_1, [BATCH_SIZE, 4, 4, 256])
    
    G_h2 = transposed_conv2d(reshape_G_h1, kernel_shape=[5, 5, 128, 256], output_shape=[BATCH_SIZE, 8, 8, 128], strides=[1, 2, 2, 1], activation_fn=tf.nn.relu, scope='transposed_conv_2')    
 
    G_h3 = transposed_conv2d(G_h2, kernel_shape=[5, 5, 64, 128], output_shape=[BATCH_SIZE,16, 16, 64], strides=[1, 2, 2, 1], activation_fn=tf.nn.relu, scope='transposed_conv_3')    

    G_h4 = transposed_conv2d(G_h3, kernel_shape=[5, 5, 32, 64], output_shape=[BATCH_SIZE, 32, 32, 32], strides=[1, 2, 2, 1], activation_fn=tf.nn.relu, scope='transposed_conv_4')

    G_prob = transposed_conv2d(G_h4, kernel_shape=[5, 5, 3, 32], output_shape=[BATCH_SIZE, 64, 64, 3], strides=[1, 2, 2, 1], activation_fn=tf.nn.tanh, scope='transposed_conv_5')

  return G_prob

def discriminator(x):
  with tf.variable_scope('D', initializer=tf.truncated_normal_initializer(stddev=.01)):

    D_h1 = conv2d(x, kernel_shape=[3, 3, 3, 64], strides=[1, 2, 2, 1], activation_fn=lrelu, keep_prob=keep_prob, scope='conv_1')

    D_h2 = conv2d(D_h1, kernel_shape=[3, 3, 64, 128], strides=[1, 2, 2, 1], activation_fn=lrelu, keep_prob=keep_prob, scope='conv_2') 
    
    D_h3 = conv2d(D_h2, kernel_shape=[3, 3, 128, 256], strides=[1, 2, 2, 1], activation_fn=lrelu, keep_prob=keep_prob, scope='conv_3')

    #D_h4 = conv2d(D_h3, kernel_shape=[3, 3, 128, 256], strides=[1, 2, 2, 1], activation_fn=lrelu, scope='conv_4')    
    
    flatten_h3 = flatten(D_h3)    
    
    D_logit = fully_connected(flatten_h3, output_dim=1, activation_fn=tf.identity, scope='fc_4')
  
  return D_logit

G_sample = generator(z)
with tf.variable_scope('Discriminator') as scope:
  D_logit_real = discriminator(x)
  scope.reuse_variables()
  D_logit_fake = discriminator(G_sample)


D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)*0.9))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_theta = tf.get_collection(tf.GraphKeys.VARIABLES, scope='D')
G_theta = tf.get_collection(tf.GraphKeys.VARIABLES, scope='G')

D_solver = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(D_loss, var_list=D_theta)
G_solver = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(G_loss, var_list=G_theta)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

i = 0
batch = 0
## CIFAR 10
x_train, _, x_test, _ = loader.cifar10(grayscale=False)
x_train = (x_train - np.min(x_train))/x_train.ptp()
x_train = x_train*2 -1

x_train = x_train.reshape(-1, CHANNEL, IMAGE_WIDTH, IMAGE_HEIGHT)

x_train = np.transpose(x_train, [0, 2, 3, 1])
n_samples = x_train.shape[0]
n_batches = n_samples/BATCH_SIZE

z_samples = sample_Z(BATCH_SIZE, Z_DIM)

for it in range(9000000):
  ### CIFAR 10
  X_mb = x_train[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]
  batch = (batch + 1)%n_batches

  _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={x: X_mb, z: sample_Z(BATCH_SIZE, Z_DIM), is_training:True, keep_prob:0.5})
 


  _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={z: sample_Z(BATCH_SIZE, Z_DIM), is_training: True, keep_prob:0.5})        


  if it % 500 == 0:
    samples = sess.run(G_sample, feed_dict={z: z_samples, is_training: False, keep_prob:1.0})
    samples = (samples[:64] + 1)/2.0
    utils.save_images(samples, [8, 8], 'out/{:04d}.png'.format(i))
    i += 1

    print "Iter: {:d}, D_loss: {:.4f}, G_loss: {:.4f}".format(it, D_loss_curr, G_loss_curr)



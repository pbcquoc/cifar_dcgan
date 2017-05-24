import tensorflow as tf
from tensorflow.contrib.layers import batch_norm

def fully_connected(input, output_dim, activation_fn, scope, keep_prob=None, is_bn_training=None):
  input_dim = input.get_shape()[1].value

  with tf.variable_scope(scope):
    W = tf.get_variable('W', [input_dim, output_dim], initializer=tf.truncated_normal_initializer(stddev=0.01))
    b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0))
    
    pre_activations = tf.matmul(input, W) + b

    if is_bn_training == None:
      bn = pre_activations
    else:
      bn = batch_norm(pre_activations, updates_collections=None, is_training=is_bn_training)

    h = activation_fn(bn)

    if keep_prob == None: 
      h_drop = h
    else:
      h_drop = tf.nn.dropout(h, keep_prob) 
    
  return h_drop

def transposed_conv2d(input, kernel_shape, output_shape, scope, activation_fn, keep_prob=None, is_bn_training=None, strides = [1,2,2,1]):
  with tf.variable_scope(scope):
    kernel = tf.get_variable('filter', kernel_shape, initializer=tf.truncated_normal_initializer(stddev=0.01))
    conv = tf.nn.conv2d_transpose(input, kernel, output_shape=output_shape, strides=strides, padding='SAME')
    b = tf.get_variable('b', [output_shape[3]], initializer=tf.constant_initializer(0))
  
    pre_activations = conv + b

    if is_bn_training == None:
      bn = pre_activations
    else:
      bn = batch_norm(pre_activations, updates_collections=None, is_training=is_bn_training)

    h = activation_fn(bn)

    if keep_prob == None: 
      h_drop = h
    else: 
      h_drop = tf.nn.dropout(h, keep_prob) 
    

  return h_drop

def conv2d(input, kernel_shape, strides, activation_fn, scope, keep_prob=None, is_bn_training=None):
  with tf.variable_scope(scope):
    filter = tf.get_variable('filter', kernel_shape, initializer=tf.truncated_normal_initializer(stddev=0.01))
    b = tf.get_variable('b', [kernel_shape[3]], initializer=tf.constant_initializer(0))
    conv = tf.nn.conv2d(input, filter, strides=strides, padding='SAME')
    
    pre_activations = conv + b

    if is_bn_training == None:
      bn = pre_activations
    else:
      bn = batch_norm(pre_activations, updates_collections=None, is_training=is_bn_training)


    h = activation_fn(bn)

    if keep_prob == None: 
      h_drop = h
    else: 
      h_drop = tf.nn.dropout(h, keep_prob)

  return h_drop


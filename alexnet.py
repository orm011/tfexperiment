import os, datetime
import numpy as np
import tensorflow as tf
import sys
import io
from common import *


# l2 weight decay nudges filters to be smoother.
# (ie, sharp peaks, like those in random noise, are discouraged.)
# note 2^2 + 2^2  << 1^2 + 3^2
CONV_WD = 0.00001 # used for conv filters. 0.001 and 0.0001 makes some of them go to 0.
FC_WD = 0.0005 # used for fully conn layers. tried 0.001 earlier.
BN_WD = 0.0005 # used for batch norm.

#TODO try fp16
#I would like to monitor gradients etc before going this route (make sure you see if something is wrong)
#tf.app.flags.DEFINE_boolean('use_fp16', False,
#                            """Train the model using fp16.""")

# TODOs:
# filters are still looking terrible.
#
# try out exponential averaging of the variables themselve
#   (akin to taking several models and averaging them)

# adapted from CIFAR-10 example
# does two main  things:
# -forces variable to be on CPU
# (we use the CPU to synchronize gpu state after each batch)
# -allows multiple model replicas to share same variable instances. 
# (this is needed for communication)
def _cpu_var(name, shape, initializer, wd=None):
    """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
    #with tf.device('/cpu:0'):
        #dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer)

    if wd != None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name=name + '_weight_loss')
        tf.add_to_collection('losses', weight_decay)
        
    return var

# scales the std dev so that the
# sum of all input fields to our filter
# is not too large or too small.
# see http://neuralnetworksanddeeplearning.com/chap3.html#weight_initialization
# for an explanation
# (the initial code was already set by the TA)
def stddev_for_shape(shape):
    total = 1
    for i in shape[:-1]:
        total *= int(i)
    # eg shape is [11,11,3,96] => total = 11*11*3
    # this is the number of input activations for this kernel.
    
    stddev = np.sqrt(2./total) # used by TA, mentioned in stanford course
    return stddev

def _normal_regularized_cpu_var(name, shape, wd, mean=0.0, stddev=None):
    # TODO: use tf.truncated_normal_initializer
    if stddev == None:
        stddev = stddev_for_shape(shape)
        
    print("Normal variable %s: mean %.3f. stddev %.5f. wd %.5f" % (name, mean, stddev, wd))
    initializer = tf.truncated_normal_initializer(mean,
                                                  stddev=stddev,
                                                  dtype=tf.float32)
    return _cpu_var(name, shape, initializer, wd)


# not regularized
def _zero_cpu_var(name, shape, val=0):
    initializer = tf.constant_initializer(val, dtype=tf.float32)
    return _cpu_var(name, shape, initializer)


MODE_training = 0
MODE_testing = 1
MODE_bncallibrate = 2
# mean to work properly for both CONV and FC layers
# is_training = True in training mode
# init scale is initial value for scale
def batch_normalization(layer, mode, scale_init, local_scope_name):
    depth = layer.get_shape()[-1]
    decay = 0.9 # used for updating population stuff.
    
    ## trained variables.
    offset = _zero_cpu_var('batch_norm_offset', shape=[depth], val=0.,)

    # with 0.005, all will be between 0.9 and 1.1
    scale = _normal_regularized_cpu_var('batch_norm_scale', shape=[depth],
                                    mean=1., stddev=0.02, wd=BN_WD)
    
    
    ## approximation of a population average
    ## in theory it would be better to do the actual average,
    ## makes code harder
    pop_mean =  tf.get_variable('pop_mean',
                                shape=[depth],
                                trainable=False,
                                initializer=tf.constant_initializer())
    
    pop_variance = tf.get_variable('pop_variance',
                                   shape=[depth],
                                   trainable=False,
                                   initializer=tf.constant_initializer(1))
    if pop_mean not in tf.get_collection('pop'):
        tf.add_to_collection('pop', pop_mean)
    if pop_variance not in tf.get_collection('pop'):
        tf.add_to_collection('pop', pop_variance)

    # during training: use batch moments
    # 2 dims for FC, 4 for CONV.
    shape_len = len(layer.get_shape())
    print('shape_len', shape_len)
    assert(shape_len in [2,4])
    # [0,1,2]  for CONV
    # [0]  for FC

    axes = range(shape_len)[:-1] # for averaging
    print('axes', axes)
    (batch_mean, batch_variance) = tf.nn.moments(layer, axes)
    if mode == Mode.training:

        update_mean = tf.assign(pop_mean,
                             pop_mean * decay + (1. - decay)*batch_mean )

        update_variance = tf.assign(pop_variance,
                                 pop_variance * decay + (1.-decay)*batch_variance )
        
        # condition return value on updating these averages
        with tf.control_dependencies([update_mean, update_variance]):
            bn = tf.nn.batch_normalization(layer,
                                           batch_mean, batch_variance,
                                           offset, scale, variance_epsilon=0.00001)
    elif mode == Mode.testing:
        bn = tf.nn.batch_normalization(layer,
                                       pop_mean,
                                       pop_variance,
                                       offset,
                                       scale,
                                       variance_epsilon=0.00001)
    else:
        assert('unknown mode')
        
    return bn



    
## the training and inteference models are slightly different
## after adding batch normalization.
## even though they share the same trained variables.
def _model(x, keep_dropout, is_training, local_scope_name):
    if is_training:
        is_training = Mode.training
    else:
        is_training = Mode.testing

    
    # Conv + ReLU + LRN + Pool, 224->55->27
    with tf.variable_scope('conv1') as scope:
        w = _normal_regularized_cpu_var('weights', shape=[11, 11, 3, 96], wd=CONV_WD)            
        conv1 = tf.nn.conv2d(x, w, strides=[1, 4, 4, 1], padding='SAME')

        # note: bias term gets superseded internal offset of batch norm.
        conv1 = batch_normalization(conv1, is_training, scale_init=1., local_scope_name=(local_scope_name, scope))
        
        conv1 = tf.nn.relu(conv1)
        #lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=5, bias=1.0, alpha=1e-4, beta=0.75)

        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU + LRN + Pool, 27-> 13
    with tf.variable_scope('conv2') as scope:
        w =  _normal_regularized_cpu_var('weights', [5, 5, 96, 256], wd=CONV_WD)
        conv2 = tf.nn.conv2d(pool1, w, strides=[1, 1, 1, 1], padding='SAME')
        conv2 = batch_normalization(conv2, is_training, scale_init=1., local_scope_name=(local_scope_name, scope))
        conv2 = tf.nn.relu(conv2)
        #lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=5, bias=1.0, alpha=1e-4, beta=0.75)

        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU, 13-> 13
    with tf.variable_scope('conv3') as scope:
        w =  _normal_regularized_cpu_var('weights', [3, 3, 256, 384], wd=CONV_WD)
        conv3 = tf.nn.conv2d(pool2, w, strides=[1, 1, 1, 1], padding='SAME')
        conv3 = batch_normalization(conv3, is_training, scale_init=1., local_scope_name=(local_scope_name, scope))
        conv3 = tf.nn.relu(conv3)

    # Conv + ReLU, 13-> 13
    with tf.variable_scope('conv4') as scope:
        w =  _normal_regularized_cpu_var('weights', [3, 3, 384, 256], wd=CONV_WD)
        conv4 = tf.nn.conv2d(conv3, w, strides=[1, 1, 1, 1], padding='SAME')
        conv4 = batch_normalization(conv4, is_training, scale_init=1., local_scope_name=(local_scope_name, scope))
        conv4 = tf.nn.relu(conv4)

    # Conv + ReLU + Pool, 13->6
    with tf.variable_scope('conv5') as scope:
        w = _normal_regularized_cpu_var('weights', [3, 3, 256, 256], wd=CONV_WD)
        conv5 = tf.nn.conv2d(conv4, w, strides=[1, 1, 1, 1], padding='SAME')
        conv5 = batch_normalization(conv5, is_training, scale_init=1., local_scope_name=(local_scope_name, scope))
        conv5 = tf.nn.relu(conv5)
        pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        
    # FC + ReLU + Dropout
    with tf.variable_scope('fc6') as scope:
        w =  _normal_regularized_cpu_var('weights', [7*7*256, 4096], wd=FC_WD)
        fc6 = tf.reshape(pool5, [-1, w.get_shape().as_list()[0]])
        fc6 = tf.matmul(fc6, w)
        fc6 = batch_normalization(fc6, is_training, scale_init=1., local_scope_name=(local_scope_name, scope))
        fc6 = tf.nn.relu(fc6)
        fc6 = tf.nn.dropout(fc6, keep_dropout)
    
    # FC + ReLU + Dropout
    with tf.variable_scope('fc7') as scope:
        w =  _normal_regularized_cpu_var('weights', [4096, 4096], wd=FC_WD)
        fc7 = tf.matmul(fc6, w)
        fc7 = batch_normalization(fc7, is_training, scale_init=1., local_scope_name=(local_scope_name,scope))
        fc7 = tf.nn.relu(fc7)
        fc7 = tf.nn.dropout(fc7, keep_dropout)

    # layer for attributes
    with tf.variable_scope('attr') as scope:
        w =  _normal_regularized_cpu_var('weights',
                                         [4096,
                                          PARAMS.num_scene_attributes], wd=FC_WD)
        #bo =  _zero_cpu_var('bias', [PARAMS.num_scene_attributes])

        # keep the logits name as is (used to look up model op)
        attr_out = tf.matmul(fc7, w)
        # bn_attr = batch_normalization(attr_out, is_training, scale_init=1., local_scope_name=(local_scope_name,scope))
        # attr = tf.nn.relu(bn_attr)

    # layer for categories
    with tf.variable_scope('category') as scope:
        w =  _normal_regularized_cpu_var('weights', [4096,
                                                     PARAMS.num_categories], wd=FC_WD)
        #bo =  _zero_cpu_var('bias', [PARAMS.num_categories])
        out = tf.matmul(fc7, w) # tf.add(, bo, name='logits')

    return (out, attr_out)

def model_train(x, keep_dropout, local_scope_name):
    return _model(x, keep_dropout, is_training=True, local_scope_name=local_scope_name)

def model_run(x, local_scope_name):
    return _model(x, keep_dropout=1., is_training=False, local_scope_name=local_scope_name)

#TODO: make loss depend also on parameters?
def loss_scene_category(logits, y):
    cross_entropy_per_example = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y, name='cross_entropy_per_example')
    cross_entropy = tf.reduce_mean(cross_entropy_per_example, name='cross_entropy')
    return cross_entropy

def loss_scene_attrs(logits, attrs):
    y = tf.sigmoid(attrs) # we are geting logits from the other net
    # this loss is not exclusive
    cross_entropy_per_example = tf.nn.weighted_cross_entropy_with_logits(logits, y, pos_weight=10, name='cross_entropy_per_example')

    # sum losses over all 102 attributes (weighted equally right now)
    cross_entropy = tf.reduce_sum(cross_entropy_per_example,
                                  reduction_indices=1, name='cross_entropy')

    # mean loss over batch
    cross_entropy = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return 2*cross_entropy

def optimizer(learning_rate):
    return tf.train.AdamOptimizer(learning_rate=learning_rate)

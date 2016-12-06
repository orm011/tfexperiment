import os, datetime
import numpy as np
import tensorflow as tf
import sys
import io

FLAGS = tf.app.flags.FLAGS

#TODO try fp16
#tf.app.flags.DEFINE_boolean('use_fp16', False,
#                            """Train the model using fp16.""")

# adapted from CIFAR-10 example
# does two main  things:
# -forces variable to be on CPU
# (we use the CPU to synchronize gpu state after each batch)
# -allows multiple model replicas to share same variable instances. 
# (this is needed for communication)
def _cpu_var(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
    with tf.device('/cpu:0'):
        #dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def stddev_for_shape(shape):
    total = 1
    for i in shape[:-1]:
        total *= int(i)
    # eg shape is [11,11,3,96] => total = 11*11*3
    
    stddev = np.sqrt(2./total) # (variance = 2. / total = E[X^2] )
    return stddev



def _normal_cpu_var(name, shape):
    # TODO: use tf.truncated_normal_initializer
    initializer = tf.random_normal_initializer(stddev=stddev_for_shape(shape),
                                               dtype=tf.float32)
    return _cpu_var(name, shape, initializer)



def _zero_cpu_var(name, shape, val=0):
    initializer = tf.constant_initializer(val, dtype=tf.float32)
    return _cpu_var(name, shape, initializer)

# mean to work properly for both CONV and FC layers
# is_training = True in training mode
# init scale is initial value for scale
def batch_normalization(layer, is_training, scale_init, local_scope_name):
    (towerscope,layerscope) = local_scope_name
    depth = layer.get_shape()[-1]

    # expected behavior: given a name,
    # a new variable will be created at training time (per gpu)
    # then, at run time, given a name match, the old vaue will be retrieved
    with tf.variable_scope(towerscope):
        with tf.variable_scope(layerscope):
            with tf.variable_scope('local'):
                local_mean =  tf.get_variable('batch_mean', shape=[depth],
                                          trainable=False, initializer=tf.constant_initializer())
            
                local_variance = tf.get_variable('batch_variance', shape=[depth],
                                trainable=False, initializer=tf.constant_initializer(1.))

    bn_vars = {
        # running averages (to be used at runtime)
        'mean': local_mean,
        'variance' : local_variance,

        # trainables
        'offset': _zero_cpu_var('offset', shape=[depth], val = 0.),
        'scale': _zero_cpu_var('scale', shape=[depth], val=scale_init)
    }

    if is_training:
        # create variables
        # during training: use batch moments
        # 2 dims for FC, 4 for CONV.
        shape_len = len(layer.get_shape())
        assert(shape_len in [2,4])

        axes = range(shape_len)[:-1] # for averaging
        # [0,1,2]  for CONV
        # [0]  for FC
        
        (batch_mean, batch_variance) = tf.nn.moments(layer, axes)
        ema = tf.train.ExponentialMovingAverage(decay=0.9999)

        #print(bn_vars['mean'], batch_mean)
        update_mean = tf.assign(bn_vars['mean'], batch_mean)
        update_variance = tf.assign(bn_vars['variance'], batch_variance)

        # want to update batch average before computing exponential avg on it
        update_exp_avg_op = ema.apply([update_mean, update_variance])

        # condition return value on updating these averages
        with tf.control_dependencies([update_exp_avg_op]):
            bn = tf.nn.batch_normalization(layer,
                                           batch_mean,
                                           batch_variance,
                                           bn_vars['offset'],
                                           bn_vars['scale'],
                                           variance_epsilon=0.001)
    else:
        # variables have been trained.
        # and average/variance over dataset for this layer
        # was computed already
        bn = tf.nn.batch_normalization(layer,
                                       bn_vars['mean'],
                                       bn_vars['variance'],
                                       bn_vars['offset'],
                                       bn_vars['scale'],
                                       variance_epsilon=0.001)

    return bn


## the training and inteference models are slightly different
## after adding batch normalization.
## even though they share the same trained variables.
def _model(x, keep_dropout, is_training, local_scope_name):

    # these variables will be defined in the enclosing scope.
    weights = {
        'wc1': _normal_cpu_var('wc1', [11, 11, 3, 96]),
        'wc2': _normal_cpu_var('wc2', [5, 5, 96, 256]),
        'wc3': _normal_cpu_var('wc3', [3, 3, 256, 384]),
        'wc4': _normal_cpu_var('wc4', [3, 3, 384, 256]),
        'wc5': _normal_cpu_var('wc5', [3, 3, 256, 256]),

        'wf6': _normal_cpu_var('wf6', [7*7*256, 4096]),
        'wf7': _normal_cpu_var('wf7', [4096, 4096]),
        'wo': _normal_cpu_var('wo', [4096, 100])
    }

    # bias term gets superseded by offset term of batch norm.
    # (see paper)
    biases = {
        # 'bc1': _zero_cpu_var('bc1', [96]),
        # 'bc2': _zero_cpu_var('bc2', [256]),
        # 'bc3': _zero_cpu_var('bc3', [384]),
        # 'bc4': _zero_cpu_var('bc4', [256]),
        # 'bc5': _zero_cpu_var('bc5', [256]),

        # 'bf6': _zero_cpu_var('bf6', [4096]),
        # 'bf7': _zero_cpu_var('bf7', [4096]),
        'bo': _zero_cpu_var('bo', [100])
    }

    # Conv + ReLU + LRN + Pool, 224->55->27
    with tf.variable_scope('conv1') as scope:
        conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 4, 4, 1], padding='SAME')
        conv1 = batch_normalization(conv1, is_training, scale_init=stddev_for_shape(weights['wc1'].get_shape()), local_scope_name=(local_scope_name, scope))
        conv1 = tf.nn.relu(conv1)
        lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=5, bias=1.0, alpha=1e-4, beta=0.75)

        pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    
    # Conv + ReLU + LRN + Pool, 27-> 13
    with tf.variable_scope('conv2') as scope:
        conv2 = tf.nn.conv2d(pool1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
        conv2 = batch_normalization(conv2, is_training, scale_init=stddev_for_shape(weights['wc2'].get_shape()), local_scope_name=(local_scope_name, scope))
        conv2 = tf.nn.relu(conv2)
        lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=5, bias=1.0, alpha=1e-4, beta=0.75)

        pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU, 13-> 13
    with tf.variable_scope('conv3') as scope:
        conv3 = tf.nn.conv2d(pool2, weights['wc3'], strides=[1, 1, 1, 1], padding='SAME')
        conv3 = batch_normalization(conv3, is_training, scale_init=stddev_for_shape(weights['wc3'].get_shape()), local_scope_name=(local_scope_name, scope))
        conv3 = tf.nn.relu(conv3)

    # Conv + ReLU, 13-> 13
    with tf.variable_scope('conv4') as scope:
        conv4 = tf.nn.conv2d(conv3, weights['wc4'], strides=[1, 1, 1, 1], padding='SAME')
        conv4 = batch_normalization(conv4, is_training, scale_init=stddev_for_shape(weights['wc4'].get_shape()), local_scope_name=(local_scope_name, scope))
        conv4 = tf.nn.relu(conv4)

    # Conv + ReLU + Pool, 13->6
    with tf.variable_scope('conv5') as scope:
        conv5 = tf.nn.conv2d(conv4, weights['wc5'], strides=[1, 1, 1, 1], padding='SAME')
        conv5 = batch_normalization(conv5, is_training, scale_init=stddev_for_shape(weights['wc5'].get_shape()), local_scope_name=(local_scope_name, scope))
        conv5 = tf.nn.relu(conv5)
        pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        
    # FC + ReLU + Dropout
    with tf.variable_scope('fc6') as scope:
        fc6 = tf.reshape(pool5, [-1, weights['wf6'].get_shape().as_list()[0]])
        fc6 = tf.matmul(fc6, weights['wf6']) # remove bias
        fc6 = batch_normalization(fc6, is_training, scale_init=1., local_scope_name=(local_scope_name, scope))
        fc6 = tf.nn.relu(fc6)
        fc6 = tf.nn.dropout(fc6, keep_dropout)
    
    # FC + ReLU + Dropout
    with tf.variable_scope('fc7') as scope:
        fc7 = tf.matmul(fc6, weights['wf7'])
        fc7 = batch_normalization(fc7, is_training, scale_init=1., local_scope_name=(local_scope_name,scope))
        fc7 = tf.nn.relu(fc7)
        fc7 = tf.nn.dropout(fc7, keep_dropout)

    # Output FC
    # should we batch normalize this?
    # the rationale seems to be that as we change these weights,
    # then, the next layer will be harder to train.
    # but the next layer is not being trained here, so leaving it
    # unnormalized
    out = tf.add(tf.matmul(fc7, weights['wo']), biases['bo'])
    return out

def model_train(x, keep_dropout, local_scope_name):
    return _model(x, keep_dropout, is_training=True, local_scope_name=local_scope_name)

def model_run(x, local_scope_name):
    return _model(x, keep_dropout=1., is_training=False, local_scope_name=local_scope_name)

#TODO: make loss depend also on parameters?
def loss(logits, y):
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
    return loss


def optimizer(learning_rate):
    return tf.train.AdamOptimizer(learning_rate=learning_rate)

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

def _normal_cpu_var(name, shape):
    total = 1
    for i in shape[:-1]:
        total *= i
    # eg shape is [11,11,3,96] => total = 11*11*3
    
    stddev = np.sqrt(2./total)
    # TODO: use tf.truncated_normal_initializer
    initializer = tf.random_normal_initializer(stddev=stddev, dtype=tf.float32)
    return _cpu_var(name, shape, initializer)

def _zero_cpu_var(name, shape):
    initializer = tf.constant_initializer(dtype=tf.float32) #zeros
    return _cpu_var(name, shape, initializer)

def model(x, keep_dropout):
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

    biases = {
        'bc1': _zero_cpu_var('bc1', [96]),
        'bc2': _zero_cpu_var('bc2', [256]),
        'bc3': _zero_cpu_var('bc3', [384]),
        'bc4': _zero_cpu_var('bc4', [256]),
        'bc5': _zero_cpu_var('bc5', [256]),

        'bf6': _zero_cpu_var('bf6', [4096]),
        'bf7': _zero_cpu_var('bf7', [4096]),
        'bo': _zero_cpu_var('bo', [100])
    }

    # Conv + ReLU + LRN + Pool, 224->55->27
    conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 4, 4, 1], padding='SAME')
    conv1 = tf.nn.relu(tf.nn.bias_add(conv1, biases['bc1']))
    lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=5, bias=1.0, alpha=1e-4, beta=0.75)
    pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU + LRN + Pool, 27-> 13
    conv2 = tf.nn.conv2d(pool1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(tf.nn.bias_add(conv2, biases['bc2']))
    lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=5, bias=1.0, alpha=1e-4, beta=0.75)
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Conv + ReLU, 13-> 13
    conv3 = tf.nn.conv2d(pool2, weights['wc3'], strides=[1, 1, 1, 1], padding='SAME')
    conv3 = tf.nn.relu(tf.nn.bias_add(conv3, biases['bc3']))

    # Conv + ReLU, 13-> 13
    conv4 = tf.nn.conv2d(conv3, weights['wc4'], strides=[1, 1, 1, 1], padding='SAME')
    conv4 = tf.nn.relu(tf.nn.bias_add(conv4, biases['bc4']))

    # Conv + ReLU + Pool, 13->6
    conv5 = tf.nn.conv2d(conv4, weights['wc5'], strides=[1, 1, 1, 1], padding='SAME')
    conv5 = tf.nn.relu(tf.nn.bias_add(conv5, biases['bc5']))
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # FC + ReLU + Dropout
    fc6 = tf.reshape(pool5, [-1, weights['wf6'].get_shape().as_list()[0]])
    fc6 = tf.add(tf.matmul(fc6, weights['wf6']), biases['bf6'])
    fc6 = tf.nn.relu(fc6)
    fc6 = tf.nn.dropout(fc6, keep_dropout)
    
    # FC + ReLU + Dropout
    fc7 = tf.add(tf.matmul(fc6, weights['wf7']), biases['bf7'])
    fc7 = tf.nn.relu(fc7)
    fc7 = tf.nn.dropout(fc7, keep_dropout)

    # Output FC
    out = tf.add(tf.matmul(fc7, weights['wo']), biases['bo'])
    
    return out

#TODO: make loss depend also on parameters?
def loss(logits, y):
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
    return loss


def optimizer(learning_rate):
    return tf.train.AdamOptimizer(learning_rate=learning_rate)

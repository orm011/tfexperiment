import os, datetime
import numpy as np
import tensorflow as tf
from DataLoader import *
import time
import signal
import sys
import io
import re
from common import *
import alexnet

loader_val = DataLoaderH5(**opt_data_val)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model', '',
                """Directory with backups (or specific checkpoint file + .meta).""")

if FLAGS.model != '':
        if os.path.isdir(FLAGS.model):
                cs = tf.train.get_checkpoint_state(FLAGS.model)
                print("manifest file contents:\n%s" % cs)
                data = cs.model_checkpoint_path
                meta = cs.model_checkpoint_path + '.meta'
                assert(os.path.isfile(data))
                assert(os.path.isfile(meta))
        elif os.path.isfile(FLAGS.model):
                data = FLAGS.model
                meta = FLAGS.model + '.meta'
else:
        print('need --model')
        sys.exit(1)


loader = DataLoaderH5(**opt_data_val)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

x = tf.placeholder(tf.float32, [None, PARAMS.fine_size, PARAMS.fine_size, PARAMS.c])

y = tf.placeholder(tf.int64, None)

with tf.variable_scope('model1') as scope:
        logits1 = alexnet.model_run(x, local_scope_name='eval')[0]


logits1 = 
        metrics = performance_metrics(logits, y, alexnet)

saver = tf.train.Saver()
saver.restore(sess, data)
full_validation((x,y,metrics), sess, loader)

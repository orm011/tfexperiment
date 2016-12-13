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
import random
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

K = 1
BSIZE = 40

opt_data_score = {
        'data_h5': 'miniplaces_256_val.h5',
        'load_size': PARAMS.load_size,
        'fine_size': PARAMS.load_size, #we will crop within.
        'data_mean': PARAMS.data_mean,
        'batch_size': BSIZE, # gets multiplied
        'randomize': False,
        'buffered_batches':1
    }


# note we will crop within evaluation
loader = DataLoaderH5(**opt_data_score)
x = tf.placeholder(tf.float32, [BSIZE, PARAMS.load_size, PARAMS.load_size, PARAMS.c])
y = tf.placeholder(tf.int64, BSIZE)
attrs = tf.placeholder(tf.int64, [BSIZE, PARAMS.num_scene_attributes])

def random_view(xbatch):
        tmp_batch = tf.random_crop(xbatch, [BSIZE, PARAMS.fine_size, PARAMS.fine_size, PARAMS.c])
        if random.getrandbits(1):
                tmp_batch = tf.reverse(tmp_batch, [False, False, True, False])
        return tmp_batch
 
# eval target will average logits over k  multiple random views of the same input
# x: input batch
def multi_image_averaging(xbatch):
        ## ( N , height, width, channel)
        batch_replicas = []
        for i in range(K):
                randomized = random_view(xbatch)
                batch_replicas.append(randomized)
                
        batch_of_batches = tf.pack(batch_replicas)
        order_by_picture = tf.transpose(batch_of_batches, perm=[1, 0, 2, 3, 4])
        input_batch = tf.reshape(order_by_picture, shape=[K*BSIZE, PARAMS.fine_size, PARAMS.fine_size, PARAMS.c])

        idx = tf.range(0, BSIZE)
        batch_of_idx = tf.pack([tf.range(0,BSIZE) for i in range(K)])
        ordered_idx_by_pic = tf.transpose(batch_of_idx, perm=[1,0])
        segments = tf.reshape(ordered_idx_by_pic, shape=[K*BSIZE])
        
        out_logits = alexnet.model_run(input_batch, local_scope_name='multi')[0]
        avgs = tf.segment_mean(out_logits,segments)
        return avgs

# with tf.variable_scope('model1') as scope:
#         logits1 = alexnet.model_run(x, local_scope_name='eval')[0]

logits = multi_image_averaging(x)
metrics = cat_perf_metrics(logits, y, alexnet)

saver = tf.train.Saver()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver.restore(sess, data)

def multi_validation(target_tuple, sess, loader, otherph={}):
    # Evaluate on the whole validation set

    start = time.time()
    (x,y,attr, metrics_target) = target_tuple
    print('[%s running scoring]:' %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    num_batch = (loader.size() + BSIZE - 1) // BSIZE
    err1_total = 0.
    err5_total = 0.
    print("Validation set size: %d. Batch size: %d. Num batches %d." % (
        loader.size(), BSIZE, num_batch))
    
    feed_dict = otherph
    err1s = []
    err5s = []

    for i in range(num_batch):
        images_batch, labels_batch, attr_batch = loader.next_batch()
        feed_dict[x] = images_batch
        feed_dict[y] = labels_batch
        feed_dict[attr] = attr_batch
        r = sess.run(metrics_target, feed_dict)
        err1s.append(r['top1'])
        err5s.append(r['top5'])
        print('...')
        
    err1s.sort()
    err5s.sort()
    err1_total = np.mean(err1s)
    err5_total = np.mean(err5s)
        
    rng=(err5s[0], err5s[-1])
    print('Category Evaluation Finished After %.2fs.\nError Top1 = %.4f.\nError Top5 = %.4f.\n(Min,Max) = (%.3f, %.3f)' %
          (time.time() - start, err1_total, err5_total, rng[0], rng[1]))
        
    return err5_total


if __name__ == '__main__':
        multi_validation((x,y,attrs,metrics), sess, loader)

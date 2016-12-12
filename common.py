import numpy as np
from enum import Enum
from collections import namedtuple
import tensorflow as tf
import time
import datetime

class Mode(Enum):
    training = 1
    bn_callibration = 2
    testing = 3


LoadParams = namedtuple('LoadParams',
                        """batch_size load_size fine_size 
                        data_mean random_distort shuffle_window
                        collection_name
                        """)


Params = namedtuple('Params', 'batch_size load_size fine_size c data_mean initial_learning_rate decay_rate dropout num_images grid_x eval_batch_size')

# Dataset Parameters
# Training Parameters
# Add them here so we can print them all out to logs (helps to see if we changed them).
PARAMS = Params(
    batch_size = 200,
    eval_batch_size = 1000,
    load_size = 256,
    fine_size = 224,
    c = 3,
    data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842]),
    initial_learning_rate = 0.001,
    decay_rate = 0.8,
    dropout = 0.5, # Dropout, probability to keep units
    num_images = 100000, # hardcoded for now.
    grid_x = 10, # for showing eval batches in tensorboard
)

EPOCH_SIZE = PARAMS.num_images // PARAMS.batch_size


# Construct dataloader
opt_data_train = {
    'data_h5': 'miniplaces_256_train.h5',
    #'data_root': 'YOURPATH/images/',   # MODIFY PATH ACCORDINGLY
    #'data_list': 'YOURPATH/train.txt', # MODIFY PATH ACCORDINGLY
    'load_size': PARAMS.load_size,
    'fine_size': PARAMS.fine_size,
    'data_mean': PARAMS.data_mean,
    'batch_size': PARAMS.batch_size,
    'buffered_batches':3,
    'randomize': True
    }

opt_data_val = {
    'data_h5': 'miniplaces_256_val.h5',
    #'data_root': 'YOURPATH/images/',   # MODIFY PATH ACCORDINGLY
    #'data_list': 'YOURPATH/val.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': PARAMS.load_size,
    'fine_size': PARAMS.fine_size,
    'data_mean': PARAMS.data_mean,
    'batch_size': PARAMS.eval_batch_size,
    'randomize': False,
    'buffered_batches':1
    }


def topkerror(logits, y, k):
   bools = tf.nn.in_top_k(logits, y, k)
   topkerr  = tf.reduce_mean(tf.cast(1 - tf.cast(bools, tf.int32), tf.float32), name="top%derr" % k)
   return topkerr


def performance_metrics(logits, y, model, summary=True):
    y = y[:,0]
    loss = model.loss_scene_category(logits, y)
    top1err = topkerror(logits, y, 1)
    top5err = topkerror(logits, y, 5)

    if summary:
        tf.scalar_summary('loss', loss)
        tf.scalar_summary('top1err', top1err)
        tf.scalar_summary('top5err', top5err)

    return {'loss':loss, 'top1':top1err, 'top5':top5err}


def full_validation(target_tuple, sess, loader, otherph={}):
    # Evaluate on the whole validation set
    start = time.time()
    (x,y,metrics_target) = target_tuple
    
    print('[%s running full validation set test]:' %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    # (orm) added + batch_size - 1 to make this work
    # with small validation sets (and give 1)
    num_batch = (loader.size() + PARAMS.eval_batch_size - 1) // PARAMS.eval_batch_size
    err1_total = 0.
    err5_total = 0.
    print("Validation set size: %d. Batch size: %d. Num batches %d." % (
            loader.size(), PARAMS.eval_batch_size, num_batch))

    feed_dict = otherph
    err1s = []
    err5s = []
    for i in range(num_batch):
        images_batch, labels_batch = loader.next_batch()
        feed_dict[x] = images_batch
        feed_dict[y] = labels_batch
        r = sess.run(metrics_target, feed_dict)
        err1s.append(r['top1'])
        err5s.append(r['top5'])
        print('...',)

    err1s.sort()
    err5s.sort()
    err1_total = np.mean(err1s)
    err5_total = np.mean(err5s)
    
    rng=(err5s[0], err5s[-1])
    print('Evaluation Finished After %.2fs.\nError Top1 = %.4f.\nError Top5 = %.4f.\n(Min,Max) = (%.3f, %.3f)' %(time.time() - start, err1_total, err5_total, rng[0], rng[1]))

    return err5_total

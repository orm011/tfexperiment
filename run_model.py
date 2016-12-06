import os, datetime
import numpy as np
import tensorflow as tf
from DataLoader import *
import time
import signal
import sys
import io
import re


batch_size = 200
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])


opt_data_val = {
    'data_h5': 'miniplaces_256_val.h5',
    #'data_root': 'YOURPATH/images/',   # MODIFY PATH ACCORDINGLY
    #'data_list': 'YOURPATH/val.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

loader_val = DataLoaderH5(**opt_data_val)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('modeldir', '',
                           """Directory with backups (.meta + checkpoint + file)""")

modeldir = FLAGS.modeldir
if modeldir != '':
        if os.path.isdir(modeldir):
                cs = tf.train.get_checkpoint_state(modeldir)
                print("manifest file contents:\n%s" % cs)
                data = cs.model_checkpoint_path
                meta = cs.model_checkpoint_path + '.meta'
                assert(os.path.isfile(data))
                assert(os.path.isfile(meta))
        else:
                print('modeldir must be a directory')
                sys.exit(1)
else:
        print('need --modeldir')
        sys.exit(1)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

# example averages
# tower_0/conv2/tower_0/conv2/Assign/ExponentialMovingAverage:0 (256,)
# tower_0/conv2/tower_0/conv2/Assign_1/ExponentialMovingAverage:0 (256,)

new_saver = tf.train.import_meta_graph(meta)
new_saver.restore(sess, data)
all_vars = tf.all_variables()

# for o in tf.get_default_graph().get_operations():
#         print(o.name, o)

for v in all_vars:
        if (re.search('.*/([^/]*)/Assign(_1)?/ExponentialMovingAverage:0',v.name)):
                print(v.name, v.get_shape())

def topkerror(logits, y, k):
        bools = tf.nn.in_top_k(logits, y, k)
        topkaccuracy  = tf.reduce_mean(tf.cast(bools, tf.float32))
        return 1 - topkaccuracy

def performance_metrics(logits, y):
        #loss = model.loss(logits, y)
        top1err = topkerror(logits, y, 1)
        top5err = topkerror(logits, y, 5)
        return {'top1err':top1err, 'top5err':top5err}

# hand coded output name. # have a convention
logits = tf.get_default_graph().get_operation_by_name("tower_0/Add").outputs
labels = tf.placeholder(tf.int64, shape=[None])
print(logits, labels)

var_full_validation = performance_metrics(logits, labels)
print(var_full_validation)

def full_validation(var_full_validation):
    # Evaluate on the whole validation set
    print('[%s]:' %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print('Evaluation on the whole validation_set at step %d...' % step)
    # (orm) added + batch_size - 1 to make this work
    # with small validation sets (and give 1)
    num_batch = (loader_val.size() + batch_size - 1) // batch_size
    err1_total = 0.
    err5_total = 0.
    
    loader_val.reset()
    for i in range(num_batch):
        images_batch, labels_batch = loader_val.next_batch(2*batch_size)
        (_, err1, err5, _) = sess.run(val_full_validation, feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1.})
        err1_total += err1
        err5_total += err5

    err1_total /= num_batch
    err5_total /= num_batch

    # need to run this code just for
    # logging into the tensorboard stuff
    top1error = tf.scalar_summary('top-%d Error' % 1, tf.constant(err1_total))
    top5error = tf.scalar_summary('top-%d Error' % 5, tf.constant(err5_total))
    sums = tf.merge_summary([top1error, top5error])
    fullvalsummary = sess.run(sums)
    summary_writer_full_validation.add_summary(fullvalsummary, step)

    print('[%s]:' %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print('Evaluation Finished! Error Top1 = ' + "{:.4f}".format(err1_total) + ", Top5 = " + "{:.4f}".format(err5_total))

full_validation(var_full_validation)

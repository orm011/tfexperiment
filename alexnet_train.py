import os, datetime
import numpy as np
import tensorflow as tf
from DataLoader import *
import time
import signal
import sys
import io

# log stuff also to a file 
class Unbuffered(io.TextIOBase):
    def __init__(self, stream, filename):
        self.stream = stream
        self.tee = open(filename,"a")  # File where you need to keep the logs
    def write(self, data):
        self.stream.write(data)
        self.tee.write(data)    # Write the data of stdout here to a text file as well
        self.stream.flush()        
        self.tee.flush()

TS = time.strftime("%a_%Y-%m-%d_%H-%M-%S",time.localtime())
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('training_iters', 100000,
                            """Number of batches to run.""")

tf.app.flags.DEFINE_integer('step_display', 50,
                            """Number of batches to run before evaluating
                            and printing model performance metrics""")

tf.app.flags.DEFINE_integer('step_save', 10000,
                            """Number of batches to run before saving parameter checkpoint""")

tf.app.flags.DEFINE_string('mnemonic', '',
                           """String used (in addition to timestamp) to tag models and logs""")

tf.app.flags.DEFINE_string('outputdir', 'tf_outputdir',
                           """Path used to save all output directories (including logs and models) of this run. Will make one if needed""")

tf.app.flags.DEFINE_string('start_from', '',
                           """if it is a <folder>, then we assume there is a `checkpoint` manifest file, and we use the most recent entry from there. 
                           if it is a <file>, we just use that directly.
                           if it is <unset>, we use random init.""")

tf.app.flags.DEFINE_integer('step_full_validation', 200,
                           """how often to run valdation metrics on full validation set. Full validation takes about 30 sec on 1 GPU""")

#tf.app.flags.DEFINE_integer('num_gpus', 1,
#                            """How many GPUs to use.""")

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


# Id is used to identify a lot of our output files uniquely within outputdir
ID = TS
if FLAGS.mnemonic != '':
    ID = FLAGS.mnemonic + '_' + TS

MODELDIR = FLAGS.outputdir + '/models_' + ID
LOGDIR = FLAGS.outputdir + '/logs_' + ID

if os.path.exists(MODELDIR):
    print("WARNING: identical folder for outputs already exists. It should be unique to avoid overwrites.")
    sys.exit(1)

if os.path.exists(LOGDIR):
    print("WARNING: identical folder for logs already exists. It should be unique to avoid overwrites.")
    sys.exit(1)

os.makedirs(MODELDIR)
os.makedirs(LOGDIR)
sys.stdout=Unbuffered(sys.stdout, LOGDIR+ '/' + ID + '_logs.out')
sys.stderr=Unbuffered(sys.stderr, LOGDIR+ '/' + ID + '_logs.err')
                                              
# define logger params
summary_writer_train = tf.train.SummaryWriter(LOGDIR+'/train_' + ID, graph=tf.get_default_graph())
summary_writer_eval = tf.train.SummaryWriter(LOGDIR+'/eval_'+ ID, graph=tf.get_default_graph())
summary_writer_full_validation = tf.train.SummaryWriter(LOGDIR+'/fullval_'+ ID, graph=tf.get_default_graph())

# Dataset Parameters
batch_size = 200
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
learning_rate = 0.001
dropout = 0.5 # Dropout, probability to keep units
training_iters = FLAGS.training_iters
step_display = FLAGS.step_display
step_save = FLAGS.step_save
path_save = MODELDIR + '/model_' + ID
start_from = FLAGS.start_from

def print_param_sizes():
    print("Summary of model layer sizes (highest to low):")
    total_parameters = 0
    per_variable = {}
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
            total_parameters += variable_parameters
        per_variable[variable] = variable_parameters

    for (v,w) in sorted(per_variable.items(), key=lambda x : -x[1]):
        print("(%s:%s). shape: %s. total: %d/%d (%.0f%%)" %
              (v.name, v.dtype, v.get_shape(), w, total_parameters, 100*(w/total_parameters)))
        
def alexnet(x, keep_dropout):
    weights = {
        'wc1': tf.Variable(tf.random_normal([11, 11, 3, 96], stddev=np.sqrt(2./(11*11*3)))),
        'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256], stddev=np.sqrt(2./(5*5*96)))),
        'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384], stddev=np.sqrt(2./(3*3*256)))),
        'wc4': tf.Variable(tf.random_normal([3, 3, 384, 256], stddev=np.sqrt(2./(3*3*384)))),
        'wc5': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2./(3*3*256)))),

        'wf6': tf.Variable(tf.random_normal([7*7*256, 4096], stddev=np.sqrt(2./(7*7*256)))),
        'wf7': tf.Variable(tf.random_normal([4096, 4096], stddev=np.sqrt(2./4096))),
        'wo': tf.Variable(tf.random_normal([4096, 100], stddev=np.sqrt(2./4096)))
    }

    biases = {
        'bc1': tf.Variable(tf.zeros(96)),
        'bc2': tf.Variable(tf.zeros(256)),
        'bc3': tf.Variable(tf.zeros(384)),
        'bc4': tf.Variable(tf.zeros(256)),
        'bc5': tf.Variable(tf.zeros(256)),

        'bf6': tf.Variable(tf.ones(4096)),
        'bf7': tf.Variable(tf.ones(4096)),
        'bo': tf.Variable(tf.ones(100))
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

# Construct dataloader
opt_data_train = {
    'data_h5': 'miniplaces_256_train.h5',
    #'data_root': 'YOURPATH/images/',   # MODIFY PATH ACCORDINGLY
    #'data_list': 'YOURPATH/train.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True
    }
opt_data_val = {
    'data_h5': 'miniplaces_256_val.h5',
    #'data_root': 'YOURPATH/images/',   # MODIFY PATH ACCORDINGLY
    #'data_list': 'YOURPATH/val.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

# loader_train = DataLoaderDisk(**opt_data_train)
# loader_val = DataLoaderDisk(**opt_data_val)
loader_train = DataLoaderH5(**opt_data_train)
loader_val = DataLoaderH5(**opt_data_val)

# tf Graph input
x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
y = tf.placeholder(tf.int64, None)
keep_dropout = tf.placeholder(tf.float32)

# Construct model
logits = alexnet(x, keep_dropout)


# Define loss and optimizer
# (orm: added tensorboard annotated scalars to get plots more easily)
def make_named_loss(logits, y):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
    summ = tf.scalar_summary('loss', loss)
    return [loss, summ]


(lossdiff_train,_) = make_named_loss(logits, y)

train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(lossdiff_train)

# Evaluate model
def make_named_top(logits, y, k):
    topkaccuracy  = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, k), tf.float32))
    topkerror = 1 - topkaccuracy
    summ = tf.scalar_summary('top-%d Error' % k, topkerror)
    return [topkerror,summ]


def make_instrumented_target(logits, y):
    loss = make_named_loss(logits, y)
    top1err = make_named_top(logits, y, 1)
    top5err = make_named_top(logits, y, 5)

    (vals, summaries) = zip(loss, top1err, top5err)
    return list(vals) + [tf.merge_summary(summaries)]


training_eval_target = make_instrumented_target(logits, y)
val_eval_target = make_instrumented_target(logits, y)
val_full_validation = make_instrumented_target(logits, y)

# define initialization
with tf.device("/cpu:0"):
    global_step = tf.Variable(0, name='global_step', trainable=False)

init = tf.initialize_all_variables()
# define saver. it will include global step
saver = tf.train.Saver()
print("run id %s" % ID)
print_param_sizes()
# both the saver and print need to be done after variables are
# created by call to alexnet

ctrlc_received = False
# Launch the graph
with tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)) as sess:
    original_sigint = signal.getsignal(signal.SIGINT)
    
    # Initialization
    if len(start_from)>1:
        saver.restore(sess, start_from)
        print("restored state from %s" % (start_from,))
    else:
        print("starting from random state...")
        sess.run(init)

    step = sess.run(global_step) # usable by python code
    print("Initial step is %d" % step)

    # (orm) in case of Ctrl+C, save current model then exit
    def record_signal(signum, frame):
        global ctrlc_received
        signal.signal(signal.SIGINT, original_sigint)
        print("Control-C received. will save before next iteration and exit... Ctrl-C again will kill it immediately")
        ctrlc_received = True

    signal.signal(signal.SIGINT, record_signal)

    def full_validation():
        # Evaluate on the whole validation set
        print('[%s]:' %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        print('Evaluation on the whole validation_set at step %d...' % step)
        # (orm) added + batch_size - 1 to make this work with small validation sets (and give 1)
        num_batch = (loader_val.size() + batch_size - 1) // batch_size
        err1_total = 0.
        err5_total = 0.
        loader_val.reset()
        for i in range(num_batch):
            images_batch, labels_batch = loader_val.next_batch(batch_size)    
            (_, err1, err5, _) = sess.run(val_full_validation, feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1.})
            err1_total += err1
            err5_total += err5

        err1_total /= num_batch
        err5_total /= num_batch

        # need to run this code just for logging into the tensorboard stuff
        top1error = tf.scalar_summary('top-%d Error' % 1, tf.constant(err1_total))
        top5error = tf.scalar_summary('top-%d Error' % 5, tf.constant(err5_total))
        sums = tf.merge_summary([top1error, top5error])
        fullvalsummary = sess.run(sums)
        summary_writer_full_validation.add_summary(fullvalsummary, step)

        print('[%s]:' %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        print('Evaluation Finished! Error Top1 = ' + "{:.4f}".format(err1_total) + ", Top5 = " + "{:.4f}".format(err5_total))

    
    while step < training_iters:
        if ctrlc_received or (step > 0 and step % step_save == 0):
            saver.save(sess, path_save, global_step=step)
            print("Model saved as of before step %d !" %(step))

        if ctrlc_received:
            sys.exit(1)
        
        # Load a batch of training data
        images_batch, labels_batch = loader_train.next_batch(batch_size)
        
        if step % step_display == 0:
            print('[%s]:' %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

            # Calculate batch loss and accuracy on training set
            l, err1, err5, tsummary = sess.run(training_eval_target, feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1.}) 
            print("-Iter " + str(step) + ", Training Loss= " + \
            "{:.4f}".format(l) + ", Error Top1 = " + \
            "{:.2f}".format(err1) + ", Top5 = " + \
            "{:.2f}".format(err5))
            summary_writer_train.add_summary(tsummary, step)
            
            # Calculate batch loss and accuracy on validation set
            images_batch_val, labels_batch_val = loader_val.next_batch(batch_size)    
            l, err1, err5, vsummary = sess.run(val_eval_target, feed_dict={x: images_batch_val, y: labels_batch_val, keep_dropout: 1.}) 
            print("-Iter " + str(step) + ", Validation Loss= " + \
            "{:.4f}".format(l) + ", Error Top1 = " + \
            "{:.2f}".format(err1) + ", Top5 = " + \
            "{:.2f}".format(err5))
            summary_writer_eval.add_summary(vsummary, step)

        if step > 0 and step % FLAGS.step_full_validation == 0:
            full_validation()
                        
        # Run optimization op (backprop)
        sess.run(train_optimizer, feed_dict={x: images_batch, y: labels_batch, keep_dropout: dropout})
        step = sess.run(global_step.assign(step + 1))

            
    saver.save(sess, path_save, global_step=step)
    print("Final model at iter %d saved!" %(step))
    print("Optimization Finished!")


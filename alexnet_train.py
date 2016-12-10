import os, datetime
import numpy as np
import tensorflow as tf
#from DataLoader import *
import loader # tf based loader
import time
import signal
import sys
import io
import alexnet
import re
from tensorflow.python.client import timeline
from collections import namedtuple
from common import *


# allow swapping in of variants
# as long as they define model, loss and optimizer
model = alexnet
from tensorflow.python.client import device_lib

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

tf.app.flags.DEFINE_integer('profile_step', 13,
                            'how often to print the timing info')

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

tf.app.flags.DEFINE_boolean('timeline_trace', False,
                            """whether we should trace training runs and write their putput in Chrome timeline format""")

tf.app.flags.DEFINE_integer('step_full_validation', 100000,
                            """how often to run valdation metrics on full validation set. WARNING: Seems to make the program go slower and slower over time, 
and the smoother batch-error in tensorboard tracks it very well anyway.""")

#subtract the cpu device
detected_gpus = 1 # len(device_lib.list_local_devices()) - 1
tf.app.flags.DEFINE_integer('num_gpus', detected_gpus, """How many GPUs to use. Defaults to using all detected gpus""")


tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

assert(FLAGS.num_gpus == 1)
print("num gpus: ", FLAGS.num_gpus)

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

path_save = MODELDIR + '/model_' + ID + '.ckpt'
start_from = FLAGS.start_from
print(sys.argv)
print("MODELDIR (can use as value for --start_from):\n%s" % MODELDIR)
print("LOGDIR:\n%s" % LOGDIR)

if start_from != '':
    if os.path.isdir(start_from):
        print("CHECKPOINT DIR:\n%s" % start_from)
        cs = tf.train.get_checkpoint_state(start_from)
        print("manifest file contents:\n%s" % cs)
        cf = cs.model_checkpoint_path
    elif os.path.isfile(start_from):
        cf = start_from

    print("CHECKPOINT FILE TO USE: %s" % cf)

Params = namedtuple('Params', 'batch_size eval_batch_size load_size fine_size c data_mean initial_learning_rate decay_rate dropout num_images grid_x eval_batches shuffle_window')

# Dataset Parameters
# Training Parameters
# Add them here so we can print them all out to logs (helps to see if we changed them).
PARAMS = Params(
    batch_size = 200,
    eval_batch_size = 1000, # TODO: eval multiple batches and average.
    load_size = 256,
    fine_size = 224,
    c = 3,
    data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842]),
    initial_learning_rate = 0.001,
    decay_rate = 0.8,
    dropout = 0.5, # Dropout, probability to keep units
    num_images = 100000, # hardcoded for now.
    grid_x = 10, # for showing eval batches in tensorboard
    eval_batches = 5, # number of batches to use for evaluation
    shuffle_window = 0 # images window to shuffle samples from
)

EPOCH_SIZE = PARAMS.num_images // PARAMS.batch_size

step_display = FLAGS.step_display
step_save = FLAGS.step_save

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
        

# makes a single picture out of a bunch of pictures
# (eg a full layer of conv filters)
# this way, we can see all of them (rather than only a couple)
# gotten from https://gist.github.com/kukuruza/03731dc494603ceab0c5
# on a comment "kitovyj commented on Sep 21 • edited"
def put_kernels_on_grid (kernel, grid_Y, grid_X, pad = 1, expand_factor=4):
    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.

    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)
    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    '''
    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)
    
    kernel1 = (kernel - x_min) / (x_max - x_min)
    
    # pad X and Y
    x1 = tf.pad(kernel1, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')
        
    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad
    
    channels = kernel1.get_shape()[2]
    
    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, channels])) #3
    
    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, channels])) #3
    
    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))
    
    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))
    
    # scale to [0, 255] and convert to uint8
    prelim =  tf.image.convert_image_dtype(x7, dtype = tf.uint8)
    if expand_factor > 0:
        new_sizes = [int(dim_sz) * expand_factor for dim_sz in prelim.get_shape()[1:3]]
    else:
        new_sizes = [int(dim_sz) // expand_factor for dim_sz in prelim.get_shape()[1:3]]

    return tf.image.resize_images(prelim, size=new_sizes)

def topkerror(logits, y, k):
   bools = tf.nn.in_top_k(logits, y, k)
   topkerr  = tf.reduce_mean(tf.cast(1 - tf.cast(bools, tf.int32), tf.float32), name="top%derr" % k)
   return topkerr

def performance_metrics(logits, y, collection):
    loss = model.loss(logits, y)
    top1err = topkerror(logits, y, 1)
    top5err = topkerror(logits, y, 5)

    # allow later code to get all three things by key
    tf.add_to_collection(collection, tf.scalar_summary('loss', loss))
    tf.add_to_collection(collection, tf.scalar_summary('top1err', top1err))
    tf.add_to_collection(collection, tf.scalar_summary('top5err', top5err))

    return {'loss':loss, 'top1':top1err, 'top5':top5err}

# (orm: adapted from the CIFAR-10 example in tensorflow.)
# a tower is a version of the model, including loss, that will run on a single gpu
def tower_loss(scope, images, labels, keep_dropout, scope_name):
  """Calculate the total loss on a single model.

  Args:
    scope: unique prefix string identifying the model tower, e.g. 'tower_0'

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """
  # each tower does its IO separately.
  # where is the batch.
  
  # build inference Graph for training
  logits = model.model_train(images, keep_dropout, local_scope_name=scope_name)

  # get loss.
  total_loss = model.loss(logits, labels) 

  # add a summary per tower
  tf.scalar_summary(total_loss.name, total_loss)
  return total_loss


# (orm: adapted from CIFAR-10)
def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  with tf.name_scope('gradient_average') as scope:
      for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
          # Add 0 dimension to the gradients to represent the tower.
          expanded_g = tf.expand_dims(g, 0)

          # Append on a 'tower' dimension which we will average over below.
          grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
  return average_grads


    
# Construct model

# define saver. it will include global step
# print("run id %s" % ID)
# 
# both the saver and print need to be done after variables are
# created by call to alexnet

# (orm) in case of Ctrl+C, save current model then exit
ctrlc_received = False 
original_sigint = signal.getsignal(signal.SIGINT)
def record_signal(signum, frame):
    global ctrlc_received
    signal.signal(signal.SIGINT, original_sigint)
    print("Control-C received. will save before next iteration and exit... Ctrl-C again will kill it immediately")
    ctrlc_received = True

signal.signal(signal.SIGINT, record_signal)

# start with an empty graph and make everything add stuff to it.
with tf.Graph().as_default():
    #tf.device("/cpu:0"):
    # loader used for training exclusively
    # loader used for validation metrics
    # (val_imgs, val_labels) = loader.input_pipeline(
    #     ['miniplaces_256_val.h5.tfrecords'],
    #     LoadParams(batch_size=PARAMS.eval_batch_size,
    #                load_size=PARAMS.load_size,
    #                fine_size=PARAMS.fine_size,
    #                data_mean=PARAMS.data_mean,
    #                random_distort=False,
    #                collection_name='eval_summaries',
    #                shuffle_window=PARAMS.eval_batch_size))

    # # loader used for running model on training set (but for evaluation)
    # (train_metrics_imgs, train_metrics_labels) = loader.input_pipeline(
    #     ['miniplaces_256_train.h5.tfrecords'],
    #     LoadParams(batch_size=PARAMS.batch_size,
    #                load_size=PARAMS.load_size,
    #                fine_size=PARAMS.fine_size,
    #                data_mean=PARAMS.data_mean,
    #                random_distort=False,
    #                collection_name='train_metrics_load',
    #                shuffle_window=3*PARAMS.batch_size))    
    
    # define logger params
    summary_writer_train = tf.train.SummaryWriter(LOGDIR+'/train_' + ID, graph=tf.get_default_graph())

    summary_writer_training = tf.train.SummaryWriter(LOGDIR+'/training_' + ID, graph=tf.get_default_graph())

    summary_writer_eval = tf.train.SummaryWriter(LOGDIR+'/eval_'+ ID, graph=tf.get_default_graph())

    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False, dtype=tf.int32)

    #start strong first epoch, never go below .0001?
    epoch_cutoffs = [ 1,    3,     5,     10]
    values =         [.002, .001, .0005, .0002, .0001]
    boundaries = [c*EPOCH_SIZE for c in epoch_cutoffs]
    learning_rate = tf.train.piecewise_constant(global_step, boundaries, values, name='learning_rate')

    # tf Graph input placeholders for each tower
    def input_placeholder(name):
        return (name,
                {'images':tf.placeholder(tf.float32,
                            [None, PARAMS.fine_size, PARAMS.fine_size, PARAMS.c],
                                         name=str(name) + '_images'),
                 
                 'labels':tf.placeholder(tf.int64, None, name=str(name) + '_label')})
    
    keep_dropout = tf.placeholder(tf.float32, name='dropout_rate')
    # shared dropout setting

    tower_grads = []
    opt = model.optimizer(learning_rate)


    # for i in range(FLAGS.num_gpus):
    #     with tf.device('/gpu:%d' %i ):
            # NB this is a name scope, not a variable scope.
    with tf.name_scope('tower_0') as scope:
        (train_imgs, train_labels)  = loader.input_pipeline(
            ['miniplaces_256_train.h5.tfrecords'],
            LoadParams(batch_size=PARAMS.batch_size,
                       load_size=PARAMS.load_size,
                       fine_size=PARAMS.fine_size,
                       data_mean=PARAMS.data_mean,
                       random_distort=True,
                       collection_name='train_load',
                       shuffle_window=PARAMS.shuffle_window))

        xs = tf.identity(train_imgs)
        ys = tf.identity(train_labels)
        kd = tf.identity(keep_dropout)

        loss = tower_loss(scope, xs, ys, 0.5, scope)

        # stuff after here that calls get variable
        # will reuse variables of the same name
        # rather than make a new one
        tf.get_variable_scope().reuse_variables()
        # based on cifar. somehow only the last tower summaries are here
        #summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
        grads = opt.compute_gradients(loss)
        # which are these?
        #TODO summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
        tower_grads.append(grads)

    # this assumes the conv1 layer is done.
    # Visualize conv1 features
    with tf.variable_scope('conv1') as scope_conv:
        tf.get_variable_scope().reuse_variables()
        weights = tf.get_variable('weights')
        grid_x = 12
        grid_y = 8   # to get a square grid for 64 conv1 features
        grid = put_kernels_on_grid (weights, grid_y, grid_x)
        conv_summary = tf.image_summary('conv1/features', grid, max_images=1)

    # train_metrics_logits = model.model_run(train_metrics_imgs,
    #                                        train_metrics_labels)
    
    # val_logits = model.model_run(val_imgs, val_labels)

    # TODO: monitor learning rate of Adam?
    grads = average_gradients(tower_grads)
    lrsum = tf.scalar_summary('learning_rate', learning_rate)

    TRAINING_SUMMARIES = 'training_summaries'
    # #TODO: monitor histogram of gradients
    # # right now complains about placeholders
    # for grad,var in grads:
    #     tag = var.op.name + '/gradients'
    #     gradsum = tf.histogram_summary(tag, grad, name=tag+'_histogram')
    #     tf.add_to_collection(TRAINING_SUMMARIES, gradsum)

    # # log historgram of trainable variables.
    # before_update = {}
    # for var in tf.trainable_variables():
    #     ivar = tf.identity(var)
    #     before_update[var.op.name] = tf.identity(var)

    # # now apply merged gradients to model variables
    # with tf.control_dependencies(before_update.values()):
    train_op = opt.apply_gradients(grads)

    # this code attempts to measure the amount of change in our weights after each iteration
    # these operations depend on the train op:
    #   It's need is motivated by observing that the weights in conv1 seem to just stop changing altogether.
    #   after a couple thousand iterations.
    #   I would like to measure this more precisely to see where/when are the changes taking place
    #   To implement this, we need to read the values of the same variables
    #   before and after the train operation in order to compute the changes
    #   the way I see to do this is using control dependencies (assuming I understand what that means)
    # TODO: it would be interesting to visualize first layer deltas as well.
    # with tf.control_dependencies([train_op]):
    #     for vafter in tf.trainable_variables():
    #         vbefore = before_update[vafter.op.name]
    #         # update_to_weight_ratio = | after - before | / max(|before|,|after|, epsilon) 
    #         vafter = tf.to_double(vafter)
    #         vbefore = tf.to_double(vbefore)
    #         epsilon = tf.constant(0.00001, dtype=tf.double)
    #         delta = vafter - vbefore

    #         base = tf.maximum(tf.abs(vbefore), tf.abs(vafter))
    #         update_to_weight = tf.to_float(tf.truediv(tf.abs(delta),
    #                                       tf.maximum(base,epsilon)))

    #         tag = vafter.op.name + '/update_weight_ratio'
    #         uwr = tf.histogram_summary(tag, update_to_weight, name=tag + '_histogram')
            
    #         tf.add_to_collection(TRAINING_SUMMARIES, uwr)

    # this has no dependency on training
    for var in tf.trainable_variables():
        tag = var.op.name
        hs = tf.histogram_summary(var.op.name, var, name=tag+'_histogram')
        tf.add_to_collection('eval_summaries', hs)
        
    # log population stats.
    # note: adding this here means we only add summary once.
    # otherwise, summary gets added every time that code runs.
    for var in tf.get_collection('pop'):
        hs = tf.histogram_summary(var.op.name, var, name=var.op.name + '/histogram_summary')
        tf.add_to_collection('eval_summaries', hs)

    print("variables declared prior to saver and init:")
    for v in tf.all_variables():
        print(v.name, v)
    print_param_sizes()
    saver = tf.train.Saver()
    init = tf.initialize_all_variables() 

    # perf eval.
    # train_metrics_target = performance_metrics(train_metrics_logits,
    #                                            train_metrics_labels, 'train_metrics')
    
    # eval_metrics_target = performance_metrics(val_logits, val_labels, 'eval_summaries')
    
    # Launch the graph
    with tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement, allow_soft_placement=True)) as sess:
        tf.train.start_queue_runners(sess, coord=None, daemon=True, start=True, collection='queue_runners')

        # Initialization
        if len(start_from)>1:
            saver.restore(sess, cf)
            print("restored state from %s" % (cf,))
            step = sess.run(global_step)
        else:
            print("starting from random state...")
            sess.run(init)
            step = 0

        print("Initial step is %d" % step)

        iter_start = None
        while step < FLAGS.training_iters:
            last_iter_start = iter_start
            iter_start  = time.time()
            if step > 0 and step % FLAGS.profile_step == 0:
                print("step %d profile: train: %0.1fs. total: %0.1fs" %
                      (step-1,
                       train_end - train_start,
                       iter_start - last_iter_start))
            # Load a batch for each gpu.
            # batches = []
            # load_start = time.time()
            # for i in range(FLAGS.num_gpus):
            #     images_batch, labels_batch = loader_train.next_batch(PARAMS.batch_size)
            #     batches.append({'images':images_batch, 'labels':labels_batch})

            
            #if False and step % step_display == 0:                    
                # start = time.time()
                # def run_test(target, feed_dict, name, step, writer):
                #     res = sess.run(target, feed_dict=feed_dict, )
                #     print("-Iter " + str(step) + ", %s Loss= " % name + \
                #       "{:.4f}".format(res['loss']) + ", Error Top1 = " + \
                #       "{:.2f}".format(res['top1']) + ", Top5 = " + \
                #       "{:.2f}".format(res['top5']))
                #     writer.add_summary(res['all_summaries'], step)
                
                # print('[%s] Starting metrics run:' %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                # print('--logidr=%s' % LOGDIR)
                
                # train_metrics_target['all_summaries'] = tf.merge_summary(
                #     tf.get_collection('train_metrics') + tf.get_collection('train_metrics_load')
                # )
                
                # run_test(train_metrics_target,
                #          feed_dict={ keep_dropout: 1.},
                #          name='Training',
                #          step=step,
                #          writer=summary_writer_train)

                # eval_summaries = tf.get_collection('eval_summaries') 
                # eval_metrics_target['all_summaries'] = tf.merge_summary(eval_summaries)
                # # run val on larger batches to denoise print output a bit?
                # run_test(eval_metrics_target,
                #          feed_dict={ keep_dropout: 1.},
                #          name='Validation',
                #          step=step,
                #          writer=summary_writer_eval)
                # end = time.time()
                # print("Metrics run took %.1f seconds" %( end - start))

            # Run optimization op (backprop)

            train_start = time.time()
            if FLAGS.timeline_trace:
                run_metadata = tf.RunMetadata()
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            else:
                run_metadata = None
                options = None

            (_, summ) = sess.run([train_op, tf.merge_summary(tf.get_collection('train_load'))],
                                 feed_dict={keep_dropout:PARAMS.dropout},
                                 options=options,
                                 run_metadata=run_metadata)
            summary_writer_training.add_summary(summ, global_step=step)
            train_end = time.time()
            step+=1
            
            if FLAGS.timeline_trace:
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                trace_file = open('timeline_%s.ctf.json' % step, 'w')
                trace_file.write(trace.generate_chrome_trace_format())
            
            # Checkpoint
            if ctrlc_received or (step % step_save == 0) or (step == FLAGS.training_iters):
                print("Saving model as of before step %d..." %(step))
                saver.save(sess, path_save, global_step=step)
                print("Model saved.")
                if ctrlc_received:
                    print("Exiting after Ctrl+C")
                    os.kill(os.getpid(), signal.SIGINT)



        print("Optimization Finished!")

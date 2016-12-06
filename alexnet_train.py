import os, datetime
import numpy as np
import tensorflow as tf
from DataLoader import *
import time
import signal
import sys
import io
import alexnet
import re
from tensorflow.python.client import timeline
            
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
detected_gpus = len(device_lib.list_local_devices()) - 1
tf.app.flags.DEFINE_integer('num_gpus', detected_gpus, """How many GPUs to use. Defaults to using all detected gpus""")

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


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


# Dataset Parameters
batch_size = 200
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
learning_rate = 0.008
dropout = 0.5 # Dropout, probability to keep units
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

def make_summary(mets):
    summaries = []
    for (name,node) in mets.items():
        summaries.append(tf.scalar_summary(name, node))
    return dict(list(mets.items()) + [('summary',tf.merge_summary(summaries))])

def topkerror(logits, y, k):
   bools = tf.nn.in_top_k(logits, y, k)
   topkaccuracy  = tf.reduce_mean(tf.cast(bools, tf.float32))
   return 1 - topkaccuracy

def performance_metrics(logits, y):
    loss = model.loss(logits, y)
    top1err = topkerror(logits, y, 1)
    top5err = topkerror(logits, y, 5)
    return {'loss':loss, 'top1err':top1err, 'top5err':top5err}

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

  # remove scoped prefix so tensorboard shows nicer stuff
  loss_name = re.sub('tower_[0-9]*/', '', total_loss.name)

  # add a summary per tower
  tf.scalar_summary(loss_name +' (raw)', total_loss)
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

# def prep_example(example):
#     image_raw = example.features.feature['image_raw'].bytes_list.value[0]
#     height = example.features.feature['height'].int64_list.value
#     shape = [f['height'][0], f['width'][0], f['depth'][0]]
#     tf.constant(f['image_raw'][0])

# def read_input(filename_queue):
#     reader = tf.TFRecordReader()
#     key, record_string = reader.read(filename_queue)
#     example, label = tf.parse_example(record_string)
#     processed_example = prep_example(example)
#     return processed_example, label

# def input_pipeline(filenames, batch_size, num_epochs=None):
#     filename_queue = tf.train.string_input_producer(
#         filenames, num_epochs=num_epochs, shuffle=True)
#     example, label = read_my_file_format(filename_queue)
#     # min_after_dequeue defines how big a buffer we will randomly sample
#     #   from -- bigger means better shuffling but slower start up and more
#     #   memory used.
#     # capacity must be larger than min_after_dequeue and the amount larger
#     #   determines the maximum we will prefetch.  Recommendation:
#     #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
#     min_after_dequeue = 10000
#     capacity = min_after_dequeue + 3 * batch_size
#     example_batch, label_batch = tf.train.shuffle_batch(
#     [example, label], batch_size=batch_size, capacity=capacity,
#     min_after_dequeue=min_after_dequeue)
#     return example_batch, label_batch
    
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
with tf.Graph().as_default(), tf.device("/cpu:0"):
    # define logger params
    summary_writer_train = tf.train.SummaryWriter(LOGDIR+'/train_' + ID, graph=tf.get_default_graph())
    summary_writer_eval = tf.train.SummaryWriter(LOGDIR+'/eval_'+ ID, graph=tf.get_default_graph())
    
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False, dtype=tf.int64)


    # tf Graph input placeholders for each tower
    def input_placeholder(name):
        return (name, {'images':tf.placeholder(tf.float32, [None, fine_size, fine_size, c]), 'labels':tf.placeholder(tf.int64, None)})
    
    placeholders = dict([input_placeholder(n) for n in range(FLAGS.num_gpus)])

    (_, evald) = input_placeholder('val')
    val_images_placeholder = evald['images']
    val_labels_placeholder = evald['labels']
    
    keep_dropout = tf.placeholder(tf.float32) # shared dropout setting

    tower_grads = []
    opt = model.optimizer(learning_rate)

    for i in range(FLAGS.num_gpus):
        with tf.device('/gpu:%d' %i ):
            # NB this is a name scope, not a variable scope.
            with tf.name_scope('%s_%d' % ("tower", i)) as scope:
                xs = tf.identity(placeholders[i]['images'])
                ys = tf.identity(placeholders[i]['labels'])
                kd = tf.identity(keep_dropout)
                loss = tower_loss(scope, xs, ys, kd, scope)

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


    # use the same variables to construct the evaluation graph.
    # note. runnable model must be constructed after training ones for now
    eval_logits = model.model_run(val_images_placeholder, local_scope_name='eval')


    # TODO: monitor learning rate of Adam?
    grads = average_gradients(tower_grads)
    summaries = []

    # TODO: monitor histogram of gradients
    # for grad,var in grads:
    #     if grad is not None:
    #         summaries.append(tf.histogram_summary(var.op.name + '/gradients', grad))


    # TODO: add decay / track decay to model parameters?
    

    # now apply merged gradients to model variables
    train_op = opt.apply_gradients(grads, global_step=global_step)

    # TODO: log historgram of trainable variables.
    for var in tf.trainable_variables():
      summaries.append(tf.histogram_summary(var.op.name, var))

    # not sure if I need to say 'all_variables' (ie, does that include
    # things outside the current graph)
    saver_vars = tf.all_variables()
    saver_dict = dict(map(lambda v: (v.name, v.dtype), saver_vars))
    for (k,v) in saver_dict.items():
        print(k, v)

    saver = tf.train.Saver()
    init = tf.initialize_all_variables() 

    # perf eval.
    metrics = performance_metrics(eval_logits, val_labels_placeholder)
    summ_train = make_summary(metrics)
    summ_eval = make_summary(metrics)

    print_param_sizes()

    # Launch the graph
    # softplacement = allow some opts to not be in the gpu if TF
    # prefers not to.
    summaries.append(summ_train['summary'])
    summary_op = tf.merge_summary(summaries)
    # overwrite
    summ_train['summary'] = summary_op 
    with tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement, allow_soft_placement=True)) as sess:
        # Initialization
        if len(start_from)>1:
            saver.restore(sess, cf)
            print("restored state from %s" % (cf,))
        else:
            print("starting from random state...")
            sess.run(init)

        step = sess.run(global_step) # usable by python code
        print("Initial step is %d" % step)

        iter_start = None
        while step < FLAGS.training_iters:
            last_iter_start = iter_start
            iter_start  = time.time()
            if step > 0 and step % 13 == 0:
                print("step %d profile: load: %0.1fs. train: %0.1fs. total: %0.1fs" % (step-1,
                                                                                       load_end - load_start,
                                                                                       train_end - train_start,
                                                                                       iter_start - last_iter_start))

            # Load a batch for each gpu.
            # TODO use the queueing ops from TF so it happens asynchronously
            batches = []
            load_start = time.time()
            for i in range(FLAGS.num_gpus):
                images_batch, labels_batch = loader_train.next_batch(batch_size)
                batches.append({'images':images_batch, 'labels':labels_batch})
            load_end = time.time()
            
            if step % step_display == 0:
                print('[%s]:' %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                start = time.time()
                def run_test(target, feed_dict, name, step, writer):
                    res = sess.run(target, feed_dict=feed_dict)
                    print("-Iter " + str(step) + ", %s Loss= " % name + \
                      "{:.4f}".format(res['loss']) + ", Error Top1 = " + \
                      "{:.2f}".format(res['top1err']) + ", Top5 = " + \
                      "{:.2f}".format(res['top5err']))
                    writer.add_summary(res['summary'], step)

                run_test(summ_train,
                         feed_dict={val_images_placeholder: batches[0]['images'], val_labels_placeholder: batches[0]['labels'], keep_dropout: 1.},
                         name='Training',
                         step=step,
                         writer=summary_writer_train)

                # run val on larger batches to denoise print output a bit?
                images_batch_val, labels_batch_val = loader_val.next_batch(4*batch_size)    
                run_test(summ_eval,
                         feed_dict={val_images_placeholder: images_batch_val, val_labels_placeholder: labels_batch_val, keep_dropout: 1.},
                         name='Validation',
                         step=step,
                         writer=summary_writer_eval)
                end = time.time()
                print("Validation run took %.1f seconds" %( end - start))

            # Run optimization op (backprop)
            feed_dict = {}
            for i in range(FLAGS.num_gpus):
                feed_dict[placeholders[i]['images']] = batches[i]['images']
                feed_dict[placeholders[i]['labels']] = batches[i]['labels']
                
            feed_dict[keep_dropout] = dropout

            train_start = time.time()
            if FLAGS.timeline_trace:
                run_metadata = tf.RunMetadata()
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            else:
                run_metadata = None
                options = None
                
            (_, step) = sess.run([train_op, global_step],
                                feed_dict=feed_dict,
                                options=options,
                                run_metadata=run_metadata)
            
            train_end = time.time()

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

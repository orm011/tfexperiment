import tensorflow as tf
import time
import signal
import sys
import io
import os, datetime
from common import LoadParams

LOADER='loader'

def preprocess(image, load_params):
    # imitate DataLoader.py code
    _original_image = image
    image = tf.identity(image, name='raw_image')

    if load_params.random_distort:
        image = tf.random_crop(image, [load_params.fine_size,
                                       load_params.fine_size, 3])
        image = tf.image.random_flip_left_right(image)
    else:
        image = tf.image.resize_image_with_crop_or_pad(
            image, load_params.fine_size, load_params.fine_size)

    # for logging
    _padded = tf.image.resize_image_with_crop_or_pad(
        image, load_params.load_size, load_params.load_size)

    before_after = tf.concat(1, [_original_image, _padded], name='before_after')

    # now convert to float. 
    image = tf.to_float(image) / 255.
    image = image - load_params.data_mean

    output = tf.identity(image, name='preprocessed_image')
    return [before_after, output]
    

def read_places_data(filename_queue, load_params):
    
     reader = tf.TFRecordReader()
     _, serialized_example = reader.read(filename_queue)
     features = tf.parse_single_example(
         serialized_example,
         features={
             'image_raw': tf.FixedLenFeature([], tf.string),
             'label': tf.FixedLenFeature([], tf.int64),
         }
     )

     image = tf.decode_raw(features['image_raw'], tf.uint8)
     label = tf.cast(features['label'], tf.int32)
     image = tf.reshape(image, shape=[load_params.load_size,
                                      load_params.load_size, 3])

     before_after, processed_example = preprocess(image, load_params)

     # also forward the input image, so that we can picture it.
     return [before_after, processed_example, label]

def input_pipeline(filenames, load_params):
    
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=None, shuffle=True)
    
    before_after, processed, label = read_places_data(filename_queue, load_params)
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size

    # if we are randomizing, shuffle among at least 5000 elements
    # about 1MB per element x input output => 1k queue => 2GB
    min_after_dequeue = load_params.shuffle_window

    # always have enough batches available + shuffle window.
    capacity = min_after_dequeue + 3 * load_params.batch_size
    before_after_batch, processed_batch, label_batch = tf.train.shuffle_batch(
        [before_after, processed, label],
        batch_size=load_params.batch_size,
        num_threads=1,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    
    # would like to see before/after.
    with tf.name_scope('loader'):
        tf.add_to_collection(LOADER, tf.image_summary('before_after', before_after_batch))
        tf.add_to_collection(LOADER, tf.image_summary('processed', processed_batch))
    
    return processed_batch, label_batch


# can run this to test / see what the loader does, after a change
if __name__ == '__main__':
    lp = LoadParams(batch_size=4, load_size=256, fine_size=224,
                data_mean=[0.45834960097,0.44674252445,0.41352266842],
                random_distort=True, shuffle_window=1000)

    p = input_pipeline(['../h5data/miniplaces_256_val.h5.tfrecords'], lp)

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        tf.train.start_queue_runners(sess, coord=None, daemon=True, start=True, collection='queue_runners')
    
        writer = tf.train.SummaryWriter('logs/images', graph=tf.get_default_graph())
        loader_summ = tf.merge_summary(tf.get_collection(LOADER))        
    
        step = 0
        while True:
            print("Start")
            res,summ = sess.run([p,loader_summ])
            writer.add_summary(summ, global_step=step)
            print("Done. sleeping")
            time.sleep(20)
            step += 1

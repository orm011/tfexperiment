import tensorflow as tf
import h5py

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input', '',
                            """Input file in h5 format""")

#following:
# https://github.com/tensorflow/tensorflow/blob/r0.12/tensorflow/examples/how_tos/reading_data/convert_to_records.py
#transforms our h5 files to a .tfrecord file readable by
# tf builtin ops. (this way, we can  use the parallel load ops
# they already provide)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

"""Converts a labeled dataset to tfrecords."""
# TODO: support unlabeled ones as well
def convert(h5_filename):
    h5f  = h5py.File(h5_filename, "r")
    im_set = h5f['images']
    lab_set = h5f['labels']

    num_examples = im_set.shape[0]
    assert(num_examples == lab_set.shape[0])
    print("found %d entries in %s\n" % (num_examples, h5_filename))
    
    rows = im_set.shape[1]
    cols = im_set.shape[2]
    depth = im_set.shape[3]
    print("dimensions: h %d x w %d x depth %d"%(rows, cols, depth))

    filename = h5_filename + '.tfrecords'
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = im_set[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(lab_set[index])),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':    
    convert(FLAGS.input)

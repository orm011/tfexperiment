import os
import numpy as np
import scipy.misc
import h5py
np.random.seed(123)
import time
import queue, threading
from queue import Queue
import common

# loading data from .h5
class DataLoaderH5(object):
    def __init__(self, **kwargs):
        self.load_size = int(kwargs['load_size'])
        self.fine_size = int(kwargs['fine_size'])
        self.data_mean = np.array(kwargs['data_mean'])
        self.randomize = kwargs['randomize']
        self.batch_size = int(kwargs['batch_size'])
        self.buffered_batches = int(kwargs['buffered_batches'])
        
        # read data info from lists
        f = h5py.File(kwargs['data_h5'], "r")
        self.im_set = f['images']
        self.cat_set = f['categories']
        self.attr_set = f['attributes']

        self.num = self.im_set.shape[0]
        assert self.im_set.shape[0]==self.cat_set.shape[0]
        assert self.cat_set.shape[0] == self.attr_set.shape[0]
        
        assert self.im_set.shape[1]==self.load_size, 'Image size error!'
        assert self.im_set.shape[2]==self.load_size, 'Image size error!'
        print(('# Images found:'), self.num)

        # up to 100k integers.
        self.shuffle_array = np.array(range(self.num))
        self._reinit()

        # now start the loader thread
        self.data_queue = Queue(self.buffered_batches) # a queue with some buffer space (# num batches queued)
        self.loader_thread = threading.Thread(target=self.load_task).start()

    #the data-loading task
    def load_task(self):
        while True:
            category_batch = np.zeros((self.batch_size))
            attribute_batch = np.zeros((self.batch_size, common.PARAMS.num_scene_attributes))
            images_batch = np.zeros((self.batch_size, self.fine_size, self.fine_size, 3)) 

            for i in range(self.batch_size):
                actual_idx = self.shuffle_array[self._idx]
                image = self.im_set[actual_idx]
                category = self.cat_set[actual_idx]
                attributes = self.attr_set[actual_idx]

                image = 2*(image.astype(np.float32)/255. - self.data_mean)
                if self.randomize:
                    flip = np.random.random_integers(0, 1)
                    if flip>0:
                        image = image[:,::-1,:]
                        offset_h = np.random.random_integers(0, self.load_size-self.fine_size)
                        offset_w = np.random.random_integers(0, self.load_size-self.fine_size)
                else:
                    offset_h = (self.load_size-self.fine_size)//2 #(orm: use int division python3)
                    offset_w = (self.load_size-self.fine_size)//2 #(orm: use int division python3)

                out_image = image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size, :]

                images_batch[i, ...] = out_image
                category_batch[i] = category
                attribute_batch[i,...] = attributes

                # reinit in the middle of the loop if necessary
                self._idx += 1
                if self._idx == self.num:
                    self._reinit()
            
            self.data_queue.put((images_batch,category_batch, attribute_batch), True)

    def _reinit(self):
        print('started shuffling array')
        print("before: ", self.shuffle_array[:10])
        start = time.time()
        np.random.shuffle(self.shuffle_array)
        end = time.time()
        print("after: " , self.shuffle_array[:10])
        print('shuffled array in %.5f' % (end - start))
        self._idx = 0
        
    def next_batch(self):
        (images_batch,category_batch,attribute_batch) = self.data_queue.get(True)        
        return images_batch, category_batch, attribute_batch
    
    def size(self):
        return self.num

    def reset(self):
        self._idx = 0

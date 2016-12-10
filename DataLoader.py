import os
import numpy as np
import scipy.misc
import h5py
np.random.seed(123)
import time
import queue, threading
from queue import Queue

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
        self.lab_set = f['labels']

        self.num = self.im_set.shape[0]
        assert self.im_set.shape[0]==self.lab_set.shape[0], '#images and #labels do not match!'
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

            labels_batch = np.zeros(self.batch_size)
            images_batch = np.zeros((self.batch_size, self.fine_size, self.fine_size, 3)) 

            for i in range(self.batch_size):
                actual_idx = self.shuffle_array[self._idx]
                image = self.im_set[actual_idx]
                label = self.lab_set[actual_idx]

                image = image.astype(np.float32)/255. - self.data_mean
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
                labels_batch[i, ...] = label

                # reinit in the middle of the loop if necessary
                self._idx += 1
                if self._idx == self.num:
                    self._reinit()
            
            self.data_queue.put((images_batch,labels_batch), True)

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
        (images_batch,labels_batch) = self.data_queue.get(True)        
        return images_batch, labels_batch
    
    def size(self):
        return self.num

    def reset(self):
        self._idx = 0

# Loading data from disk
class DataLoaderDisk(object):
    def __init__(self, **kwargs):

        self.load_size = int(kwargs['load_size'])
        self.fine_size = int(kwargs['fine_size'])
        self.data_mean = np.array(kwargs['data_mean'])
        self.randomize = kwargs['randomize']
        self.data_root = os.path.join(kwargs['data_root'])

        # read data info from lists
        self.list_im = []
        self.list_lab = []
        with open(kwargs['data_list'], 'r') as f:
            for line in f:
                path, lab =line.rstrip().split(' ')
                self.list_im.append(os.path.join(self.data_root, path))
                self.list_lab.append(int(lab))
        self.list_im = np.array(self.list_im, np.object)
        self.list_lab = np.array(self.list_lab, np.int64)
        self.num = self.list_im.shape[0]
        print(('# Images found:'), self.num)

        # permutation
        perm = np.random.permutation(self.num) 
        self.list_im = self.list_im[perm]
        self.list_lab = self.list_lab[perm]

        self._idx = 0
        
    def next_batch(self, batch_size):
        images_batch = np.zeros((batch_size, self.fine_size, self.fine_size, 3)) 
        labels_batch = np.zeros(batch_size)
        for i in range(batch_size):
            image = scipy.misc.imread(self.list_im[self._idx])
            image = scipy.misc.imresize(image, (self.load_size, self.load_size))
            image = image.astype(np.float32)/255.
            image = image - self.data_mean
            if self.randomize:
                flip = np.random.random_integers(0, 1)
                if flip>0:
                    image = image[:,::-1,:]
                offset_h = np.random.random_integers(0, self.load_size-self.fine_size)
                offset_w = np.random.random_integers(0, self.load_size-self.fine_size)
            else:
                offset_h = (self.load_size-self.fine_size)//2 #(orm: use int division python3)
                offset_w = (self.load_size-self.fine_size)//2 #(orm: use int division python3)

            images_batch[i, ...] =  image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size, :]
            labels_batch[i, ...] = self.list_lab[self._idx]
            
            self._idx += 1
            if self._idx == self.num:
                self._idx = 0
        
        return images_batch, labels_batch
    
    def size(self):
        return self.num

    def reset(self):
        self._idx = 0

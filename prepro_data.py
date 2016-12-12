import os
import numpy as np
import h5py
import scipy.misc
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('datapath', '.',
                            """Location of train/ val/ and test/ folders and their .txt descriptors""")

def createH5(params):

        # create output h5 file
        output_h5 = '%s_%d_%s.h5' %(params['name'], params['img_resize'], params['split'])
        f_h5 = h5py.File(output_h5, "w")

        # read data info from lists
        list_im = []
        list_cat = []
        list_attr = []
        with open(params['data_list'], 'r') as f:
            for line in f:
                attrs = line.rstrip().split(' ')
                list_im.append(os.path.join(params['data_root'], attrs[0]))
                elt0 = int(attrs[1])
                list_cat.append(elt0)
                attrs = list(map(float, attrs[2:]))
                list_attr.append(attrs)

                
        list_im = np.array(list_im, np.object)
        list_cat = np.array(list_cat, np.uint8)
        list_attr = np.array(list_attr, np.float)

        N = list_im.shape[0]
        print(('# Images found:'), N)
        
        # permutation
        perm = np.random.permutation(N) 
        list_im = list_im[perm]
        list_cat = list_cat[perm]
        list_attr = list_attr[perm]

        im_set = f_h5.create_dataset("images", (N,params['img_resize'],params['img_resize'],3), dtype='uint8') # space for resized images
        f_h5.create_dataset("categories", dtype='uint8', data=list_cat)
        f_h5.create_dataset("attributes", dtype='float', data=list_attr)

        for i in range(N):
                image = scipy.misc.imread(list_im[i])
                assert image.shape[2]==3, 'Channel size error!'
                image = scipy.misc.imresize(image, (params['img_resize'],params['img_resize']))

                im_set[i] = image

                if i % 1000 == 0:
                        print('processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N))

        f_h5.close()


if __name__=='__main__':
        params_train = {
                'name': 'miniplaces',
                'split': 'train',
                'img_resize': 256,
                'data_root': '%s' % FLAGS.datapath,
                'data_list': '%s/train_joint.txt' % FLAGS.datapath
        }

        params_val = {
                'name': 'miniplaces',
                'split': 'val',
                'img_resize': 256,
                'data_root': '%s/' % FLAGS.datapath,
                'data_list': '%s/val_joint.txt' % FLAGS.datapath
        }

        params_test = {
                'name': 'miniplaces',
                'split': 'test',
                'img_resize': 256,
                'data_root': '%s/' % FLAGS.datapath,
                'data_list': '%s/test.txt' % FLAGS.datapath
        }

        createH5(params_train)
        createH5(params_val)
        #createH5(params_test)

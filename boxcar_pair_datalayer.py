import sys
sys.path.append('/tmp3/pcjeff/caffe/python/')
import caffe
import numpy as np
import yaml
from multiprocessing import Process, Queue
from datetime import datetime
import cv2
import os
import random

BoxcarFile = '/tmp3/pcjeff/boxcar/BoxCars21k/BoxCars.npy'
PREFECTH = True

def caffe_phase_trans(index):
    if index == 0:
        return 'train'
    elif index == 1:
        return 'test'
    else:
        return None

def readImg(img_path, img_scale, crop_bb=None):
    im = cv2.imread(img_path)
    if crop_bb:
        ymin, xmin, ymax, xmax = crop_bb
        im = im[xmin:xmax, ymin:ymax, :]

    im = cv2.resize(im, img_scale)
    im[:,:, 0] -= 103
    im[:,:, 1] -= 117
    im[:,:, 2] -= 123
    im = np.swapaxes(im, 0, 2)

    return im
    
class BoxCarPairDataLayer(caffe.Layer):

    def readInData(self, BoxcarFile):
        self.task = 'classification'
        self.level = 'medium'
        info_dict = np.load(BoxcarFile).tolist()
        self._classes = info_dict[self.task][self.level]['typesMapping']

        for sample_index, label_index in info_dict[self.task][self.level][caffe_phase_trans(self.phase)]:
            label = self._classes.keys()[self._classes.values().index(label_index)]
            for vehicle in info_dict['samples'][sample_index]['vehicleSamples']:
                path = vehicle['path']
                BB = list(vehicle['2DBB'])
                if BB[2] <= BB[0]:
                    BB[2] = BB[2] + 80
                if BB[3] <= BB[1]:
                    BB[3] = BB[3] + 80

                BB = tuple(BB)
                self.imageset.append({'path':path, 'label_index':label_index, 'label':label, '2DBB': BB})
   

    def _shuffle_db_inds(self):
        self._perm = np.random.permutation(np.arange(len(self.imageset)))
        self._cur = 0

    def _get_next_batch_inds(self):
        if self._cur + self.batch_size >= len(self.imageset):
            self._shuffle_db_inds()

        db_inds = self._perm[self._cur: self._cur + self.batch_size]
        self._cur += self.batch_size

        return db_inds

    def _get_minibatch(self):

        if PREFECTH:
            return self._blob_queue.get()
        else:
            db_inds = self._get_next_batch_inds()
            blobs = {'surveillance_data': [], 'web_data':[], 'class_label': []}

            for index in db_inds:
                im = readImg( os.path.join(self.img_dir, self.imageset[index]['path']), (self.img_width, self.img_height)\
                        self.imageset[index]['2DBB'])
                label_index = self.imageset[index]['label_index']
                #find web data img 
                web_data_dir = os.path.join('/tmp3/pcjeff/boxcar/cropped_webimg/', self.imageset[index]['label'].replace(' ', ''))
                web_im = self.readImg(random.choice(os.listdir(web_data_dir)))
                
                blobs['surveillance_data'].append(im)
                blobs['class_label'].append(label_index)
                blobs['web_data'].append(web_im)

            blobs['surveillance_data'] = np.array(blobs['surveillance_data'])
            blobs['web_data'] = np.array(blobs['web_data'])
            blobs['class_label'] = np.array(blobs['class_label'])
            return blobs

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)

        self.img_width = layer_params['width']
        self.img_height = layer_params['height']
        self.batch_size = layer_params['batch_size']
        self.img_dir = '/tmp3/pcjeff/boxcar/BoxCars21k/'
        self._name_to_top_map = {}
        self.imageset = []

        self._name_to_top_map['surveillance_data'] = 0
        self._name_to_top_map['web_data'] = 1
        self._name_to_top_map['class_label'] = 2

        print 'Reading Data...'
        self.readInData(BoxcarFile)
        self._shuffle_db_inds()

        if PREFECTH:
            self._blob_queue = Queue(10)
            self._prefetch_process = BlobFetcher(self._blob_queue, self.imageset, self.batch_size, self.img_width, self.img_height)
            self._prefetch_process.daemon = True
            self._prefetch_process.start()

            def cleanup():
                print 'Terminating BlobFetcher'
                self._prefetch_process.terminate()
                self._prefetch_process.join()

            import atexit
            atexit.register(cleanup)

        top[self._name_to_top_map['surveillance_data']].reshape(self.batch_size, 3, self.img_width, self.img_height)
        top[self._name_to_top_map['web_data']].reshape(self.batch_size, 3, self.img_width, self.img_height)
        top[self._name_to_top_map['class_label']].reshape(self.batch_size,1,1)
        
        assert len(top) == len(self._name_to_top_map)


    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        blobs = self._get_minibatch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            top[top_ind].reshape(*(blob.shape))
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        pass

class BlobFetcher(Process):

    def __init__(self, queue, db, batch_size, img_width, img_height):
        super(BlobFetcher, self).__init__()
        self._queue = queue
        self._db = db
        # train set or test set list (paths and label)
        self._perm = None
        self._cur = 0
        self.img_dir = '/tmp3/pcjeff/boxcar/BoxCars21k/'
        self._queue = queue
        self._batch_size = batch_size
        self.img_width = img_width
        self.img_height = img_height
        self._shuffle_db_inds()

    def _shuffle_db_inds(self):
        self._perm = np.random.permutation(np.arange(len(self._db)))
        self._cur = 0

    def _get_next_batch_inds(self):
        if self._cur + self._batch_size >= len(self._db):
            self._shuffle_db_inds()

        db_inds = self._perm[self._cur:self._cur+self._batch_size]
        self._cur += self._batch_size

        return db_inds

    def _get_minibatch(self):
        db_inds = self._get_next_batch_inds()
        blobs = {'surveillance_data': [], 'web_data':[], 'class_label': []}

        for index in db_inds:
            im = readImg( os.path.join(self.img_dir, self._db[index]['path']), (self.img_width, self.img_height))
            label_index = self._db[index]['label_index']
            #find web data img 
            web_data_dir = os.path.join('/tmp3/pcjeff/boxcar/cropped_webimg/', self._db[index]['label'].replace(' ', ''))
            img_name = random.choice(os.listdir(web_data_dir))
            web_im = self.readImg(os.path.join(web_data_dir, img_name))
            
            blobs['surveillance_data'].append(im)
            blobs['class_label'].append(label_index)
            blobs['web_data'].append(web_im)

        blobs['surveillance_data'] = np.array(blobs['surveillance_data'])
        blobs['web_data'] = np.array(blobs['web_data'])
        blobs['class_label'] = np.array(blobs['class_label'])
        return blobs

    def run(self):
        print 'BlobFetcher started'
        while True:
            blobs = self._get_minibatch()
            self._queue.put(blobs)




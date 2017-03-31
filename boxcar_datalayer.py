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
CROP_IMG = True
ALIGN_POSE = False

def caffe_phase_trans(index):
    if index == 0:
        return 'train'
    elif index == 1:
        return 'val'
    else:
        return None

def readSRInputImg(img_path, img_scale,crop_bb=None):
    im = cv2.imread(img_path)
    if crop_bb:
        ymin, xmin, ymax, xmax = crop_bb
        im = im[xmin:xmax, ymin:ymax, :]

    im = cv2.resize(im, img_scale)
    # substract 127.5 (255/2) and times 2/255 to shift the pixel value between [-1,1]
    # for sr network input

    im = im.astype(np.float)
    im[...] -= 127.5
    im[...] *= 0.0078125
    im = np.swapaxes(im, 0, 2)

    return im

def readInputImg(img_path, img_scale,crop_bb=None):
    im = cv2.imread(img_path)
    if crop_bb:
        ymin, xmin, ymax, xmax = crop_bb
        im = im[xmin:xmax, ymin:ymax, :]

    im = cv2.resize(im, img_scale)
    # not substract the imagenet RGB mean value

    im = np.swapaxes(im, 0, 2)

    return im

def Prune_BB(BB):
    BB = list(BB)
    if BB[2] <= BB[0] or BB[2] - BB[0] < 30:
        BB[2] += 80
    else:
        BB[2] += 20
    if BB[3] <= BB[1] or BB[3] - BB[1] < 30:
        BB[3] += 40
    else:
        BB[3] += 20
    BB = tuple(BB)

    return BB

class BoxCarDataLayer(caffe.Layer):

    def readInData(self, BoxcarFile):
        self.task = 'classification'
        self.level = 'medium'
        info_dict = np.load(BoxcarFile).tolist()
        self._classes = info_dict[self.task][self.level]['typesMapping']
        
        get_this_data = True

        for sample_index, label_index in info_dict[self.task][self.level][caffe_phase_trans(self.phase)]:
            label = self._classes.keys()[self._classes.values().index(label_index)]
            if self.USE_LESS_DATA and caffe_phase_trans(self.phase) == 'train':
                if get_this_data:
                    # random choose if we are going to pick this sample
                    get_this_data = False
                    continue
                else:
                    get_this_data = True

                vehicle = info_dict['samples'][sample_index]['vehicleSamples'][0]
                path = vehicle['path']
                BB = Prune_BB(vehicle['2DBB'])
                self.imageset.append({'path':path, 'label_index':label_index, 'label':label, '2DBB': BB})
            else:
                for vehicle in info_dict['samples'][sample_index]['vehicleSamples']:
                    path = vehicle['path']
                    BB = Prune_BB(vehicle['2DBB'])
                    self.imageset.append({'path':path, 'label_index':label_index, 'label':label, '2DBB': BB})
   
    def _get_minibatch(self):
        return self._blob_queue.get()
        
    def print_setting(self):
        print 'sur_width: {}'.format(self.sur_width)
        print 'sur_height: {}'.format(self.sur_height)
        print 'web_width: {}'.format(self.web_width)
        print 'web_height: {}'.format(self.web_height)
        print 'img_batch_size: {}'.format(self.batch_size)
        print 'USE_LESS_DATA: {}'.format(self.USE_LESS_DATA)
        print 'USE_PAIR_DATA: {}'.format(self.USE_PAIR_DATA)

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)

        self.sur_width = layer_params['sur_width']
        self.sur_height = layer_params['sur_height']
        self.web_width = layer_params['web_width']
        self.web_height = layer_params['web_height']
        self.batch_size = layer_params['batch_size']
        self.USE_LESS_DATA = layer_params['USE_LESS_DATA']
        self.USE_PAIR_DATA = layer_params['USE_PAIR_DATA']
        self.IS_SR_NET = layer_params['IS_SR_NET']
        self.img_dir = '/tmp3/pcjeff/boxcar/BoxCars21k/'
        self._name_to_top_map = {}
        self.imageset = []
        
        self.print_setting()

        if self.USE_PAIR_DATA:
            self._name_to_top_map['surveillance_data'] = 0
            self._name_to_top_map['web_data'] = 1
            self._name_to_top_map['class_label'] = 2
        else:
            self._name_to_top_map['data'] = 0
            self._name_to_top_map['class_label'] = 1

        print 'Reading Data...'
        self.readInData(BoxcarFile)

        # Set up blob queue
        self._blob_queue = Queue(10)
        self._prefetch_process = BlobFetcher(self._blob_queue, self.imageset, self.batch_size,\
                (self.sur_width, self.sur_height), (self.web_width, self.web_height), self.USE_PAIR_DATA, self.IS_SR_NET)
        self._prefetch_process.start()

        def cleanup():
            print 'Terminating BlobFetcher'
            self._prefetch_process.terminate()
            self._prefetch_process.join()

        import atexit
        atexit.register(cleanup)

        if self.USE_PAIR_DATA:
            top[self._name_to_top_map['surveillance_data']].reshape(self.batch_size, 3, self.sur_width, self.sur_height)
            top[self._name_to_top_map['web_data']].reshape(self.batch_size, 3, self.web_width, self.web_height)
            top[self._name_to_top_map['class_label']].reshape(self.batch_size,1,1)
        else:
            top[self._name_to_top_map['data']].reshape(self.batch_size, 3, self.sur_width, self.sur_height)
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

    def __init__(self, queue, db, batch_size, sur_scale, web_scale, use_pair_data=False, is_sr=True):
        super(BlobFetcher, self).__init__()
        self._queue = queue
        self._db = db
        # train set or test set list (paths and label)
        self._perm = None
        self._cur = 0
        self.img_dir = '/tmp3/pcjeff/boxcar/BoxCars21k/'
        self._queue = queue
        self._batch_size = batch_size
        self.sur_scale = sur_scale
        self.web_scale = web_scale
        self.USE_PAIR_DATA = use_pair_data
        self.IS_SR_NET = is_sr
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
        blobs = {'data': [], 'class_label': []}

        for index in db_inds:
            BB = self._db[index]['2DBB'] if CROP_IMG else None

            if self.IS_SR_NET:
                im = readSRInputImg( os.path.join(self.img_dir, self._db[index]['path']), self.sur_scale, BB)
            else:
                im = readInputImg( os.path.join(self.img_dir, self._db[index]['path']), self.sur_scale, BB)

            label_index = self._db[index]['label_index']
            #flip im or not
            #TODO
            blobs['data'].append(im)
            blobs['class_label'].append(label_index)

        blobs['data'] = np.array(blobs['data'])
        blobs['class_label'] = np.array(blobs['class_label'])
        return blobs

    def _get_pair_data_minibatch(self):
        db_inds = self._get_next_batch_inds()
        blobs = {'surveillance_data': [], 'web_data':[], 'class_label': []}
    
        for index in db_inds:
            BB = self._db[index]['2DBB'] if CROP_IMG else None
            if self.IS_SR_NET:
                im = readSRInputImg( os.path.join(self.img_dir, self._db[index]['path']), self.sur_scale, BB)
            else:
                im = readInputImg( os.path.join(self.img_dir, self._db[index]['path']), self.sur_scale, BB)

            label_index = self._db[index]['label_index']
            #find web data img 
            if ALIGN_POSE:
                pass
                # the pose in boxcar are front-side
            else:
                web_data_dir = os.path.join('/tmp3/pcjeff/boxcar/cropped_webimg/', self._db[index]['label'].replace(' ', ''))
                img_name = random.choice(os.listdir(web_data_dir))
                if self.IS_SR_NET:
                    web_im = readSRInputImg( os.path.join(web_data_dir, img_name), self.web_scale)
                else:
                    web_im = readInputImg( os.path.join(web_data_dir, img_name), self.web_scale )
           
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
            if self.USE_PAIR_DATA:
                blobs = self._get_pair_data_minibatch()
            else:
                blobs = self._get_minibatch()
            self._queue.put(blobs)

        print 'img_batch_size: {}'.format(self.batch_size)
        print 'USE_LESS_DATA: {}'.format(self.USE_LESS_DATA)
        print 'USE_PAIR_DATA: {}'.format(self.USE_PAIR_DATA)

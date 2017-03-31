import sys
import csv
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import utils
import argparse
import glob
import random
import h5py
from xml.etree.ElementTree import parse, Element
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from utils import getCaffeClassifier, getCaffeOutput
from operator import itemgetter
from collections import defaultdict
from collections import OrderedDict

def parse_args():

    parser = argparse.ArgumentParser(description='Boxcar dataset Testing')
    parser.add_argument('--prototxt', dest='prototxt', help='deploy prototxt for caffemodel', type=str)
    parser.add_argument('--caffemodel', dest='caffemodel', help='caffemodel', type=str)
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id', default=0, type=int)
    parser.add_argument('--sr', dest='is_sr', help='use the input size for sr network or not', 
            action='store_true', default=False)
    args = parser.parse_args()

    return args

def readSRInputImg(img_path, crop_bb=None):
    im = cv2.imread(img_path)
    if crop_bb:
        ymin, xmin, ymax, xmax = crop_bb
        im = im[xmin:xmax, ymin:ymax, :]

    im = cv2.resize(im, (114, 114))
    im = im.astype(np.float)
    im[...] -= 127.5
    im[...] *= 0.0078125
    im = np.swapaxes(im, 0, 2)

    return im

def readInputImg(img_path, crop_bb=None):
    im = cv2.imread(img_path)
    if crop_bb:
        ymin, xmin, ymax, xmax = crop_bb
        im = im[xmin:xmax, ymin:ymax, :]

    im = cv2.resize(im, (227, 227))
    im = np.swapaxes(im, 0, 2)

    return im

def indices(l, val):
    retval = []
    last = 0
    while val in l[last:]:
        i = l[last:].index(val)
        retval.append(last + i)
        last += i + 1

    return retval

def class_acc(y_true, y_pred, cls):

    index = indices(y_true, cls)
    if len(index) == 0:
        return 0, 0
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true, y_pred = y_true[index], y_pred[index]
    
    tp = [1 for k in range(y_pred.shape[0]) if y_true[k]==y_pred[k]]
    tp = np.sum(tp)
    return tp/float(y_pred.shape[0]), y_true.shape[0]

def dump_mis(cls_true, cls_pred, cls_name):
    
    conf_mat = defaultdict(int)
    for label, pred in zip(cls_true, cls_pred):
        if label == cls_name and label != pred:
            conf_mat[pred] += 1

    return conf_mat

def eval_on_boxcar(args):
    cls_true = []
    cls_pred = []

    net = getCaffeClassifier(args.prototxt, args.caffemodel, args.gpu_id)

    Imgset_file = '/tmp3/pcjeff/boxcar/BoxCars21k/BoxCars.npy'
    dataset = np.load(Imgset_file).tolist()
    task = 'classification'
    level = 'medium'
    phase = 'test'
    class_info = dataset[task][level]['typesMapping']

    for sample_index, label_index in dataset[task][level][phase]:
        for vehicle in dataset['samples'][sample_index]['vehicleSamples']:
            path = os.path.join('/tmp3/pcjeff/boxcar/BoxCars21k/', vehicle['path'])
            BB = list(vehicle['2DBB'])
            BB = utils.Prune_boxcar_BB(BB)

            if args.is_sr:
                im = readSRInputImg(path, BB)
            else:
                im = readInputImg(path, BB)
            cls_score = getCaffeOutput(im, net, 'boxcar_cls_score')
            cls_index = np.argmax(cls_score)

            cls_true.append(label_index)
            cls_pred.append(cls_index)

            if len(cls_true) % 100 == 0:
                print '{} images proccessed'.format(len(cls_true))
            if len(cls_true) % 1000 == 0 and len(cls_true) > 0:
                print 'Class Accuracy:', accuracy_score(cls_true, cls_pred)

    
    print 'Overall Accuracy:', accuracy_score(cls_true, cls_pred)
    print '======================  acc for each class  =========================='

    acc_list = []
    for i in range(len(class_info)):
        cls_name = class_info.keys()[class_info.values().index(i)]
        acc, num = class_acc(cls_true, cls_pred, i)
        if num > 0:
            acc_list.append(acc)
        print 'class {}, {}, {}'.format(cls_name, acc, num)

    acc_list = np.array(acc_list)
    print 'Averge Accuracy: {}'.format(np.sum(acc_list)/acc_list.shape[0])
    '''
    print class_info.keys()[class_info.values().index(74)]
    for k,v in sorted(dump_mis(cls_true, cls_pred, 74).items(), key=itemgetter(1)):
        cls_name = class_info.keys()[class_info.values().index(k)]
        print cls_name, v
    '''

def eval_on_web_data(args):
    cls_true = [] 
    cls_pred = []

    net = getCaffeClassifier(args.prototxt, args.caffemodel, args.gpu_id)
    
    Imgset_file = '/tmp3/pcjeff/boxcar/BoxCars21k/BoxCars.npy'
    dataset = np.load(Imgset_file).tolist()
    class_info = dataset['classification']['medium']['typesMapping']

    test_file = '/tmp3/pcjeff/boxcar/imgset/web_crop_test_label.txt'
    root_dir = '/tmp3/pcjeff/boxcar/'
    with open(test_file, 'r') as f:
        for line in f:
            img_path, label_index = line.strip().split()
            if not os.path.exists(os.path.join(root_dir, img_path)):
                continue

            im = readImg(os.path.join(root_dir, img_path))
            cls_score = getCaffeOutput(im, net, 'boxcar_cls_score')
            cls_index = np.argmax(cls_score)

            cls_true.append(int(label_index))
            cls_pred.append(cls_index)

            if len(cls_true) % 100 == 0:
                print '{} images proccessed'.format(len(cls_true))
            if len(cls_true) % 1000 == 0 and len(cls_true) > 0:
                print 'Class Accuracy:', accuracy_score(cls_true, cls_pred)


    print 'Overall Accuracy:', accuracy_score(cls_true, cls_pred)
    print '======================  acc for each class  =========================='

    acc_list = []
    for i in range(len(class_info)):
        cls_name = class_info.keys()[class_info.values().index(i)]
        acc, num = class_acc(cls_true, cls_pred, i)
        if num > 0:
            acc_list.append(acc)
        print 'class {}, {}, {}'.format(cls_name, acc, num)

    acc_list = np.array(acc_list)
    print 'Averge Accuracy: {}'.format(np.sum(acc_list)/acc_list.shape[0])

if __name__ == '__main__':
    args = parse_args()
    eval_on_boxcar(args)

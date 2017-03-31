import sys
import caffe
import numpy as np
import yaml
import cv2
import os
import random
import argparse
import utils

def parse_args():

    parser = argparse.ArgumentParser(description='Boxcar dataset Testing')
    parser.add_argument('--prototxt', dest='prototxt', help='deploy prototxt for caffemodel', type=str)
    parser.add_argument('--caffemodel', dest='caffemodel', help='caffemodel', type=str)
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id', default=0, type=int)
    parser.add_argument('--outdir', dest='output_dir', help='output_dir for imgs', type=str)
    args = parser.parse_args()

    return args


def get_network(prototxt, caffemodel, gpu_id):
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)

    return caffe.Net(prototxt, caffemodel, caffe.TEST)

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

def restoreSROutput2Img(im):
    im[...] /= 0.0078125
    im[...] += 127.5
    im = np.swapaxes(im, 0, 2)

    return im

def get_img(net, im, layername):

    net.blobs['data'].data[0, :, ...] = im
    out = net.forward()
    output = net.blobs[layername].data
    return np.squeeze(output, axis=(0,))

def main(args):
    Imgset_file = '/tmp3/pcjeff/boxcar/BoxCars21k/BoxCars.npy'
    dataset = np.load(Imgset_file).tolist()
    task = 'classification'
    level = 'medium'
    phase = 'test'
    class_info = dataset[task][level]['typesMapping']

    net = get_network(args.prototxt, args.caffemodel, args.gpu_id)

    for sample_index, label_index in dataset[task][level][phase]:
        for vehicle in dataset['samples'][sample_index]['vehicleSamples']:
            path = os.path.join('/tmp3/pcjeff/boxcar/BoxCars21k/', vehicle['path'])
            BB = list(vehicle['2DBB'])
            BB = utils.Prune_boxcar_BB(BB)

            im = readSRInputImg(path, BB)
            sr_im = restoreSROutput2Img(get_img(net, im, 'gen'))
            utils.writeImg2dir(args.output_dir, '_'.join(vehicle['path'].split('/')), sr_im)

def get_origin_img(args):
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
            
            im = readInputImg(path, BB)
            im = np.swapaxes(im, 0, 2)
            utils.writeImg2dir(args.output_dir, '_'.join(vehicle['path'].split('/')), im)
 
if __name__ == '__main__':
    args = parse_args()  
    #get_img(args)
    main(args)

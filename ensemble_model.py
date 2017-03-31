import numpy as np
import random
import lmdb
import sys
sys.path.append('/tmp3/pcjeff/caffe/python/')
import caffe
import cv2
from collections import defaultdict


def construct_solver(solver_prototxt, pretrain_model=None):
    solver = caffe.SGDSolver(solver_prototxt)
    if pretrain_model:
        solver.net.copy_from(pretrain_model)

    return solver.net

def deploy_model(deploy_prototxt, caffemodel):
    return caffe.Net(deploy_prototxt, caffemodel, caffe.TEST)

def copy_params(source_model, target_model, filename=None):
    #load params in vgg16 to siamese
    for layer_name in source_model.params:
        if layer_name in target_model.params:
            print layer_name
            for i in range(len(target_model.params[layer_name])):
                target_model.params[layer_name][i].data[...] = source_model.params[layer_name][i].data.copy()

    if filename:
        print 'Wrote to caffemodel {}'.format(filename)
        target_model.save(filename)
    else:
        return target_model

def copy2pairlayer(source_model, target_model, filename):
    for layer_name in source_model.params:
        if layer_name + '_p' in target_model.params:
            print layer_name + '_p'
            target_model.params[layer_name + '_p'][0].data[...] = source_model.params[layer_name][0].data.copy()
            target_model.params[layer_name + '_p'][1].data[...] = source_model.params[layer_name][1].data.copy()

    print 'Wrote to caffemodel {}'.format(filename)
    target_model.save(filename)


def replace_layer(source_model, target_model, layer_names, filename=None):
    """insert fc6 from sia_vgg to vgg16"""
    for layer_name in layer_names:
        print layer_name
        target_model.params[layer_name][0].data[...] = source_model.params[layer_name][0].data.copy()
        target_model.params[layer_name][1].data[...] = source_model.params[layer_name][1].data.copy()
    
    if filename:
        print 'Wrote to caffemodel {}'.format(filename)
        target_model.save(filename)
    else:
        return target_model

if __name__ == '__main__':

    prototxt = '/tmp3/pcjeff/caffe/models/sr/sr_id_deploy.prototxt'
    caffemodel = '/tmp3/pcjeff/FSRCNN/Train/sr_114_227_iter_25000.caffemodel'
    sr_net = deploy_model(prototxt, caffemodel)

    prototxt = '/tmp3/pcjeff/caffe/models/bvlc_reference_caffenet/deploy.prototxt'
    caffemodel = '/tmp3/pcjeff/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    caffenet = deploy_model(prototxt, caffemodel)

    prototxt = '/tmp3/pcjeff/caffe/models/bvlc_reference_caffenet/deconv_solver.prototxt'
    net = construct_solver(prototxt)

    net = copy_params(sr_net, net)
    net = copy_params(caffenet, net, '/tmp3/pcjeff/FSRCNN/sr_caffenet.caffemodel')


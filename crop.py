import numpy as np
import os
import cv2

BoxcarFile = '/tmp3/pcjeff/boxcar/BoxCars21k/BoxCars.npy'
USE_LESS_DATA = True
BOXCAR_DIR = '/tmp3/pcjeff/boxcar/BoxCars21k/'
OUTPUT_DIR = '/tmp3/pcjeff/boxcar/cropped_boxcarimg/'

def readInData():
    task = 'classification'
    level = 'medium'
    info_dict = np.load(BoxcarFile).tolist()
    _classes = info_dict[task][level]['typesMapping']
    imageset = []

    for sample_index, label_index in info_dict[task][level]['train']:
        label = _classes.keys()[_classes.values().index(label_index)]
        if USE_LESS_DATA:
            vehicle = info_dict['samples'][sample_index]['vehicleSamples'][0]
            path = vehicle['path']
            BB = Prune_BB(vehicle['2DBB'])
            imageset.append({'path':path, 'label_index':label_index, 'label':label, '2DBB': BB})
        else:
            for vehicle in info_dict['samples'][sample_index]['vehicleSamples']:
                path = vehicle['path']
                BB = Prune_BB(vehicle['2DBB'])
                imageset.append({'path':path, 'label_index':label_index, 'label':label, '2DBB': BB})

    return imageset

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

def crop_img(imageset):
    name_counter = 0

    for blob in imageset:
        im_path = os.path.join(BOXCAR_DIR, blob['path'])
        im = cv2.imread(im_path)
        BB = Prune_BB(blob['2DBB'])

        ymin, xmin, ymax, xmax = BB
        im = im[xmin:xmax, ymin:ymax, :]

        filename = os.path.join(OUTPUT_DIR, str(name_counter) + '.jpg')
        print 'Write img: {}'.format(filename)
        cv2.imwrite(filename, im)

        name_counter+=1

if __name__ == '__main__':
    imageset = readInData()
    crop_img(imageset)



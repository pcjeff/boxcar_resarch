import numpy as np 


BoxcarFile = '/tmp3/pcjeff/boxcar/BoxCars21k/BoxCars.npy'
task = 'classification'
level = 'medium'

info_dict = np.load(BoxcarFile).tolist()
CLASSES = info_dict[task][level]['typesMapping']

CLASSES = dict(((k.replace(' ',''), v) for k,v in CLASSES.iteritems()))

print CLASSES


with open('./crop_test.txt') as f:
    for line in f:
        car_type = line.strip().split('/')[1]
        print line.strip(), CLASSES[car_type]

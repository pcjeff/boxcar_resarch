import sys
import random

def readAllImglist(filename, dataset=[]):
    with open(filename, 'r') as f:
        for line in f:
            img_name = line.strip()
            dataset.append(img_name)

def write2File(dataset, Trainfile='train.txt', Testfile='test.txt'):
    with open(Trainfile, 'a+') as Trainf, open(Testfile, 'a+') as Testf:
        for img_name in dataset:
            choice = random.choice('0123')
            if choice != '0':
                print >>Trainf, img_name
            else:
                print >>Testf, img_name

if __name__ == '__main__':
    dataset = []
    filename = './valid_crop_all.txt'
    Trainfile = './crop_train.txt'
    Testfile = './crop_test.txt'

    readAllImglist(filename, dataset)
    write2File(dataset, Trainfile, Testfile)


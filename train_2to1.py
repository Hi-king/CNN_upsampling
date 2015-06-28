#!/usr/bin/env python
"""
superresolution

"""
import cv2
import random
import argparse
import numpy
import chainer
import chainer.optimizers
import chainer.cuda
import sys
import os
import pickle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import models

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("trains")
#parser.add_argument("tests")
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

PATCH_SHAPE = (15, 15, 3)

if args.model == "simple3layer":
    model = models.simple3layer.Model(PATCH_SHAPE)
elif args.model == "conv3layer":
    model = models.conv3layer.Model(PATCH_SHAPE)
elif args.model == "conv3layer_large":
    PATCH_SHAPE = models.conv3layer_large.Model.PATCH_SHAPE
    model = models.conv3layer_large.Model()
else:
    exit(1)

if args.gpu >= 0:
    chainer.cuda.init(args.gpu)
    model.to_gpu()


def read_image_label(trains_dir, skip=10, num=10000):
    class BigImg(object):
        def __init__(self, img):
            print img.shape
            img_big_width = numpy.concatenate(
                (numpy.fliplr(img), img, numpy.fliplr(img)),
                axis=1)
            self.img_big = numpy.concatenate(
                (numpy.flipud(img_big_width), img_big_width, numpy.flipud(img_big_width)),
                axis=0)
            self.original_shape = img.shape

        def getpatch(self, y, x):
            PATCH_SHAPE = (15, 15, 3)
            input_noisy = self.img_big[
                self.original_shape[0]+y-(PATCH_SHAPE[0]-1)/2:self.original_shape[0]+y+(PATCH_SHAPE[0]+1)/2,
                self.original_shape[1]+x-(PATCH_SHAPE[1]-1)/2:self.original_shape[1]+x+(PATCH_SHAPE[1]+1)/2
            ]
            return input_noisy

    trains = []
    now_i = 41
    target_i = now_i+skip/2
    next_i = now_i+skip
    print "{}/{:04d}.png".format(trains_dir, now_i)
    nowimg = BigImg(cv2.imread("{}/{:04d}.png".format(trains_dir, now_i)))
    print "{}/{:04d}.png".format(trains_dir, next_i)
    nextimg = BigImg(cv2.imread("{}/{:04d}.png".format(trains_dir, next_i)))
    print "{}/{:04d}.png".format(trains_dir, target_i)
    targetimg = cv2.imread("{}/{:04d}.png".format(trains_dir, target_i))

    for i in xrange(num):
        target_y = random.randint(0, nowimg.original_shape[0]-1)
        target_x = random.randint(0, nowimg.original_shape[1]-1)

        target_bgr = targetimg[target_y, target_x]

        input_concat = numpy.concatenate(
            (nowimg.getpatch(target_y, target_x), nextimg.getpatch(target_y, target_x)),
            axis = 2
        )

        yield input_concat, target_bgr

# train
data = []
label = []
for i, (img, bgr) in enumerate(read_image_label(args.trains, num=10000000)):
    #model.train(img.transpose((2,0,1)), bgr.reshape((3,1)))
    if i % 1000 == 1:
        data = numpy.array(data)
        label = numpy.array(label)
        if args.gpu >= 0:
            data = chainer.cuda.to_gpu(numpy.array(data, dtype=numpy.float32))
            label = chainer.cuda.to_gpu(numpy.array(label, dtype=numpy.float32))

        error = model.train(data, label)
        print("error\t{}\t{}".format(i, error))
        data = []
        label = []
    if i % 100000 == 1:
        with open("output/{}_{}.dump".format(args.model, i), "w+") as f:
            pickle.dump(model, f)
    else:
        data.append(img.transpose((2,0,1)))
        label.append(bgr.reshape(3,1))

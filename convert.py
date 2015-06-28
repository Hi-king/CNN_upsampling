#!/usr/bin/env python
import chainer
import argparse
import cv2
import numpy
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import models
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("modelpath")
parser.add_argument("inputs")
parser.add_argument("outputs")
parser.add_argument("base_index", type=int)
parser.add_argument("skip", type=int)
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

if args.gpu >= 0:
    chainer.cuda.init(args.gpu)

with open(args.modelpath) as f:
    model = pickle.load(f)
PATCH_SHAPE = model.PATCH_SHAPE


#PATCH_SHAPE = (9, 9, 3)

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
        input_noisy = self.img_big[
            self.original_shape[0]+y-(PATCH_SHAPE[0]-1)/2:self.original_shape[0]+y+(PATCH_SHAPE[0]+1)/2,
            self.original_shape[1]+x-(PATCH_SHAPE[1]-1)/2:self.original_shape[1]+x+(PATCH_SHAPE[1]+1)/2
        ]
        return input_noisy


# test
skip = args.skip
now_i = args.base_index
next_i = now_i+skip
print "{}/{:04d}.png".format(args.inputs, now_i)
nowimg = BigImg(cv2.imread("{}/{:04d}.png".format(args.inputs, now_i)))
print "{}/{:04d}.png".format(args.inputs, next_i)
nextimg = BigImg(cv2.imread("{}/{:04d}.png".format(args.inputs, next_i)))

zero_img = numpy.zeros(nowimg.original_shape)

xys = []
inputs = []
i = 0
lasti = nowimg.original_shape[1] * nowimg.original_shape[0] -1
for x in xrange(nowimg.original_shape[1]):
    for y in xrange(nowimg.original_shape[0]):
        i += 1
        if i % 1000 == 0:
            print i
        input_concat = numpy.concatenate(
            (nowimg.getpatch(y, x), nextimg.getpatch(y, x)),
            axis = 2
        )
        inputs.append(input_concat.transpose(2,0,1))
        xys.append((x,y))


        if i % 1000 == 0 or i == lasti:
            data = numpy.array(inputs)
            print data.shape
            #data = numpy.array(inputs[:100000])
            #print data.shape

            if args.gpu >= 0:
                data = chainer.cuda.to_gpu(numpy.array(data, dtype=numpy.float32))
                bgrs = model.predict(data)

            for (x,y), bgr in zip(xys, bgrs):
                bgr = numpy.array(chainer.cuda.to_cpu(bgr), dtype=int)
                #print(bgr)
                zero_img[y, x] = bgr

            cv2.imwrite(
                "{}/{:04d}.png".format(args.outputs, now_i+skip/2),
                zero_img)
                
            xys = []
            inputs = []

cv2.imwrite(
    "{}/{:04d}.png".format(args.outputs, now_i+skip/2),
    zero_img)


#!/usr/bin/env python
import chainer
import argparse
import cv2
import numpy
import sys
import os
import random


parser = argparse.ArgumentParser()
parser.add_argument("imga")
parser.add_argument("imgb")
args = parser.parse_args()

a = cv2.imread(args.imga)
b = cv2.imread(args.imgb)

num = 100
error = 0
for _ in xrange(num):
    target_y = random.randint(0, a.shape[0]-1)
    target_x = random.randint(0, a.shape[1]-1)
    diff = (a[target_y, target_x] - b[target_y, target_x])
    se_vec = numpy.dot(diff, diff)
    rse = numpy.sqrt(numpy.sum(se_vec))
    error += rse
print(error / num)

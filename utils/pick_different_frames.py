#!/usr/bin/env python
import argparse
import os
import PIL.Image
import numpy
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("raw_imgs_dir")
parser.add_argument("output_dir")
parser.add_argument("threshould", type=int)
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)


def different_frames(dirpath):
    prev_frame = None
    for basename in os.listdir(dirpath):
        filename = dirpath + "/" + basename
        img = numpy.asarray(
            PIL.Image.open(open(filename)),
            dtype=numpy.float32)
        if prev_frame is None:
            yield filename
        else:
            diff = numpy.linalg.norm(img - prev_frame)
            print diff
            if diff > args.threshould:
                yield filename
        prev_frame = img

for i, filename in enumerate(different_frames(args.raw_imgs_dir)):
    ext = os.path.splitext(filename)[1]
    shutil.copy(
        filename,
        "{}/{:04d}{}".format(args.output_dir, i, ext)
    )

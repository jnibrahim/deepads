# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "../python"))
import argparse
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def list_image(root, recursive, exts):
    print("Listing images in {}".format(root))
    if recursive:
        folders = {}
        for path, dirs, files in os.walk(root, followlinks=True):
            dirs.sort()
            files.sort()
            for fname in files:
                fpath = os.path.join(path, fname)
                suffix = os.path.splitext(fname)[1].lower()
                if os.path.isfile(fpath) and (suffix in exts):
                    if path not in folders:
                        folders[path] = len(folders)
                    yield os.path.relpath(fpath, root)
    else:
        for fname in sorted(os.listdir(root)):
            fpath = os.path.join(root, fname)
            suffix = os.path.splitext(fname)[1].lower()
            if os.path.isfile(fpath) and (suffix in exts):
                yield os.path.relpath(fpath, root)


def preprocess_image(image_file_path, output_path):
    # Randomly rotate, resize, sheaer, zoom, and flip original image up to 20 copies
    datagen = ImageDataGenerator(rotation_range=40,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest')

    img = load_img(image_file_path)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape) # this is a Numpy array with shape (1, 3, 150, 150)

    save_prefix = os.path.basename(image_file_path).split('.')[0]
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=output_path, save_prefix=save_prefix, save_format='jpg'):
        i += 1
        if i > 20:
            break  # otherwise the generator would loop indefinitely


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess images for training purpose')
    # Required arguements
    parser.add_argument('source', help='source directory of the training images')
    parser.add_argument('output', help='output directory of preprocessed images')

    # Argument for image processing
    igroup = parser.add_argument_group('arguments for image processing')
    igroup.add_argument('--exts', nargs='+', default=['.jpeg', '.jpg', '.png'],
                        help='list of acceptable image extensions.')

    ogroup = parser.add_argument_group('control arguments')
    ogroup.add_argument('--num-thread', type=int, default=1,
                        help='number of threads used for image process jobs.')
    ogroup.add_argument('--recursive', type=bool, default=False,
                        help='if ture recursively walk through sub-directories. Otherwise only walk images in the source directory.')

    args = parser.parse_args()
    args.source = os.path.abspath(args.source)
    args.output = os.path.abspath(args.output)
    return args


if __name__ == '__main__':
    args = parse_args()

    image_list = list_image(args.source, args.recursive, args.exts)
    image_list = list(image_list)


    for image_path in image_list:
        file_path = os.path.join(args.source, image_path)
        preprocess_image(file_path, args.output)

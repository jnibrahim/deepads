
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "../python"))
import argparse
import cv2
import numpy as np
import time
import traceback
import uuid

try:
    import multiprocessing
except ImportError:
    multiprocessing = None


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


def find_faces(image_ndarray, classifier):
    faces = classifier.detectMultiScale(image_ndarray,
                                        scaleFactor=1.3,
                                        minNeighbors=5,
                                        minSize=(50, 50),
                                        flags = cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        yield (x, y , w, h)


def faces_detection(q_out, i, fpath, face_cascade):
    try:
        image = cv2.imread(fpath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = find_faces(gray, face_cascade)
        faces = list(faces)
        q_out.put((i, fpath, image, faces))
    except:
        traceback.print_exc()
        print("Failed to process image file: {}".format(fpath))
        q_out.put((i, fpath, None, None))


def save_faces(image_ndarray, faces, file_path, output_path, width, height):
    file_name = os.path.basename(file_path)
    file_prefix = os.path.splitext(file_name)[0]

    for (x, y, w, h) in faces:
        try:
            face = image_ndarray[y:y+h, x:x+w]
            if (width != 0) or (height != 0):
                resizeW = width if width > 0 else face.shape[1]
                resizeH = height if height > 0 else face.shape[0]
                face = cv2.resize(face, (resizeW, resizeH), interpolation = cv2.INTER_CUBIC)
            file_suffix = uuid.uuid5(uuid.NAMESPACE_DNS, "{}_{}+{}-{}+{}".format(file_path, x, y ,w, h))
            file_name = "{}_{}.jpg".format(file_prefix, file_suffix)
            output_file_path = os.path.join(output_path, file_name)
            print("Save face to: {}".format(output_file_path))
            cv2.imwrite(output_file_path, face, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        except Exception as e:
            traceback.print_exc()
            print("Failed to write face images for {}".format(file_path))
            continue


def read_worker(source, face_cascade, q_in, q_out):
    while True:
        deq = q_in.get()
        if deq is None:
            break
        i, image_path = deq
        fpath = os.path.join(source, image_path)
        print("Read worker {} - detecting face in {}".format(i, fpath))
        faces_detection(q_out, i, fpath, face_cascade)


def write_worker(q_out, output_path, width, height):
    more = True
    while more:
        deq = q_out.get()
        if deq is not None:
            i, image_path, image, faces = deq
            error = False
            if image is None:
                print("Failed to read image: {}".format(image_path))
                error = True
            if faces is None:
                print("Failed to detect faces for image: {}".format(image_path))
                error = True
            if error != True:
                num_face = len(faces)
                if num_face > 0:
                    print("Writing {} face {} for image {}, path: {}".format(num_face, 'file' if num_face < 2 else 'files', i, image_path))
                    save_faces(image, faces, image_path, output_path, width, height)
        else:
            more = False

def im2faces_single_process(image_list, face_cascade,
                            source, output,
                            width, height):
    print('multiprocessing not available, fall back to single threaded processing')
    try:
        import Queue as queue
    except ImportError:
        import queue
    queue_out = queue.Queue()
    count = 0
    for i, image_path in enumerate(image_list):
        try:
            fpath = os.path.join(source, image_path)
            print("Find face in {}".format(fpath))
            faces_detection(queue_out, i, fpath, face_cascade)
            if queue_out.empty():
                continue
            _, image_path, image, faces = queue_out.get()
            if image is None:
                print("Failed to read image: {}".format(image_path))
                continue
            if faces is None:
                print("Failed to detect faces for image: {}".format(image_path))
                continue
            save_faces(image, faces, image_path, output, width, height)
        except Exception as e:
            traceback.print_exc()
            continue


def im2faces_multi_process(image_list, face_cascade,
                           num_thread, source, output,
                           width, height):
       input_num_thread = int(num_thread / 2)
       output_num_thread = num_thread - input_num_thread
       queue_out = [multiprocessing.Queue(1024) for i in range(input_num_thread)]
       queue_in = [multiprocessing.Queue(1024) for i in range(output_num_thread)]

       read_process = []

       queue_out_index = 0
       for i in range(input_num_thread):
           process = multiprocessing.Process(target=read_worker,
                                             args=(source, face_cascade, queue_in[i], queue_out[queue_out_index % output_num_thread]))
           read_process.append(process)
           queue_out_index += 1

       for rp in read_process:
           rp.start()

       write_process = [multiprocessing.Process(target=write_worker,
                                                  args=(queue_out[i], output, width, height)) \
                                                  for i in range(output_num_thread)]
       for wp in write_process:
           wp.start()

       for i, image_path in enumerate(image_list):
           queue_in[i % len(queue_in)].put((i, image_path))

       for q_in in queue_in:
           q_in.put(None)
       for rp in read_process:
           rp.join()

       for q_out in queue_out:
           q_out.put(None)
       for wp in write_process:
           wp.join()

def im2faces(source, output,
             cascade_classifier_path,
             exts=['.jpeg', '.jpg', '.JPG', '.png'], width=0, height=0,
             num_thread=1, recursive=False):
    image_list = list_image(source, recursive, exts)
    image_list = list(image_list)
    print("Listed {} images".format(len(image_list)))
    face_cascade = cv2.CascadeClassifier(cascade_classifier_path)

    try:
        output_path = os.path.abspath(output)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    except Exception as e:
        traceback.print_exc
        print ("Failed to create directory of ".format(output_path))
        return

    start = time.time()
    if num_thread > 1 and multiprocessing is not None:
        im2faces_multi_process(image_list, face_cascade,
                               num_thread, source, output_path,
                               width, height)
    else:
        im2faces_single_process(image_list, face_cascade,
                                source, output_path,
                                width, height)
    process_time = time.time() - start
    print("Process time: {:0.2f}s".format(process_time))



def parse_args():
    parser = argparse.ArgumentParser(description='Prepare facial recognition training image datasets with pre-trained face detection models')
    # Required arguements
    parser.add_argument('source', help='source directory of the training image datasets')
    parser.add_argument('output', help='output directory of prepared image dataset')

    # Models and prediction
    mgroup = parser.add_argument_group('Model and prediction')
    mgroup.add_argument('--cascade_classifier_path', required=True,
                        help='path to haar cascade classifier provided by opencv.')

    # Argument for image processing
    igroup = parser.add_argument_group('arguments for image processing')
    igroup.add_argument('--exts', nargs='+', default=['.jpeg', '.jpg', '.png'],
                        help='list of acceptable image extensions.')
    igroup.add_argument('--width', type=int, default=0,
                        help = 'width of the resulting image of faces, default: 0 - won\'t change width.')
    igroup.add_argument('--height', type=int, default=0,
                        help = 'height of the resulting image of faces, default: 0 - won\'t change height.')

    ogroup = parser.add_argument_group('control arguments')
    ogroup.add_argument('--num_thread', type=int, default=1,
                        help='number of threads used for image process jobs.')
    ogroup.add_argument('--recursive', type=bool, default=False,
                        help='if ture recursively walk through sub-directories. Otherwise only walk images in the source directory.')

    args = parser.parse_args()
    args.source = os.path.abspath(args.source)
    args.output = os.path.abspath(args.output)
    return args


if __name__ == '__main__':
    args = parse_args()
    im2faces(args.source, args.output,
             args.cascade_classifier_path,
             args.exts, args.width, args.height,
             args.num_thread, args.recursive)

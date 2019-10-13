from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import flask
from flask import request

import tensorflow as tf
from scipy import misc
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import facenet
import detect_face
import os
from os.path import join as pjoin
import sys
import time
import copy
import math
import pickle
import re
import logging
from sklearn.svm import SVC
from sklearn.externals import joblib
import random
from time import sleep

app = flask.Flask(__name__)
# app.config["DEBUG"] = True

# log = logging.getLogger('werkzeug')
# log.setLevel(logging.ERROR)

def YUVtoRGB(byteArray):

    e = 320*240
    Y = byteArray[0:e]
    Y = np.reshape(Y, (240, 320))

    s = e
    V = byteArray[s::2]
    V = np.repeat(V, 2, 0)
    V = np.reshape(V, (120, 320))
    V = np.repeat(V, 2, 0)

    U = byteArray[s+1::2]
    U = np.repeat(U, 2, 0)
    # 38398 -> 38400
    U = np.append(U, [0, 0])
    U = np.reshape(U, (120, 320))
    U = np.repeat(U, 2, 0)

    RGBMatrix = (np.dstack([Y, U, V])).astype(np.uint8)
    RGBMatrix = cv2.cvtColor(RGBMatrix, cv2.COLOR_YUV2RGB, 3)

    return RGBMatrix


print('Creating networks and loading parameters')
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, 'det')

        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 3
        batch_size = 1000
        image_size = 182
        input_image_size = 160

        # HumanNames = ['Human_a','Human_b','Human_c','...','Human_h']    #train human name

        print('Loading feature extraction model')
        modeldir = '20170511-185253/20170511-185253.pb'
        facenet.load_model(modeldir)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        classifier_filename = 'classifier/my_classifier.pkl'
        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)
            print('load classifier file-> %s' % classifier_filename_exp)

        regex = re.compile('(.*)(\.txt|\.zip)')
        HumanNames = [x for x in class_names if not regex.match(x)]

        c = 0

        print('Start Recognition!')
        prevTime = 0

        correctCount = 0
        total = 0

        @app.route('/api/yuv', methods=['POST'])
        def predict():

            frame = YUVtoRGB(list(request.data))

            frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)

            # rotate 2700 degree
            (h, w) = frame.shape[:2]
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, 270, 1.0)
            frame = cv2.warpAffine(frame, M, (h, w))

            curTime = time.time()    # calc fps
            timeF = frame_interval

            if (c % timeF == 0):
                find_results = []

                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                frame = frame[:, :, 0:3]
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]

                if nrof_faces > 0:
                    print('Detected_FaceNum: %d' % nrof_faces)
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]

                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    bb = np.zeros((nrof_faces,4), dtype=np.int32)

                    for i in range(nrof_faces):
                        emb_array = np.zeros((1, embedding_size))

                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]

                        # inner exception
                        if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                            print('face is inner of range!')
                            continue

                        cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                        cropped[i] = facenet.flip(cropped[i], False)
                        scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                        scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                               interpolation=cv2.INTER_CUBIC)
                        scaled[i] = facenet.prewhiten(scaled[i])
                        scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                        feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                        predictions = model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face

                        if best_class_probabilities > 0.8:
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20
                            for H_i in HumanNames:
                                if HumanNames[best_class_indices[0]] == H_i:
                                    result_names = HumanNames[best_class_indices[0]] + " "+ str(best_class_probabilities * 100) + "%"
                                    print(result_names)
                                    return str(result_names)
            return ""


@app.route('/', methods=['GET'])
def home():
    return "<h1>Hello Flask!</h1>"

@app.route('/align')
def align():
    output_dir_path = 'aligned'
    output_dir = os.path.expanduser(output_dir_path)
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    datadir = 'images'
    dataset = facenet.get_dataset(datadir)
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, './det')

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    margin = 44
    image_size = 182

    # Add a random key to the filename to allow alignment using multiple processes
    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)
    print('Goodluck')

    with open(bounding_boxes_filename, "w") as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned = 0
        for cls in dataset:
            output_class_dir = os.path.join(output_dir, cls.name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
            imgCount = 0
            for image_path in cls.image_paths:
                if imgCount > 30:
                    break
                nrof_images_total += 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                output_filename = os.path.join(output_class_dir, filename + '.png')
                print(image_path)
                if not os.path.exists(output_filename):
                    try:
                        img = misc.imread(image_path)
                        print('read data dimension: ', img.ndim)
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(image_path, e)
                        print(errorMessage)
                    else:
                        if img.ndim < 2:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))
                            continue
                        if img.ndim == 2:
                            img = facenet.to_rgb(img)
                            print('to_rgb data dimension: ', img.ndim)
                        img = img[:, :, 0:3]
                        print('after data dimension: ', img.ndim)

                        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                        nrof_faces = bounding_boxes.shape[0]
                        print('detected_face: %d' % nrof_faces)
                        if nrof_faces == 1: # only one face in image
                            det = bounding_boxes[:, 0:4]
                            img_size = np.asarray(img.shape)[0:2]
                            # if nrof_faces > 1:
                            #     bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                            #     img_center = img_size / 2
                            #     offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                            #                          (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                            #     offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                            #     index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                            #     det = det[index, :]
                            det = np.squeeze(det)
                            bb_temp = np.zeros(4, dtype=np.int32)

                            bb_temp[0] = det[0]
                            bb_temp[1] = det[1]
                            bb_temp[2] = det[2]
                            bb_temp[3] = det[3]

                            cropped_temp = img[bb_temp[1]:bb_temp[3], bb_temp[0]:bb_temp[2], :]
                            try:
                                scaled_temp = misc.imresize(cropped_temp, (image_size, image_size), interp='bilinear')
                                nrof_successfully_aligned += 1
                                misc.imsave(output_filename, scaled_temp)
                                text_file.write('%s %d %d %d %d\n' % (output_filename, bb_temp[0], bb_temp[1], bb_temp[2], bb_temp[3]))
                                imgCount += 1
                            except ValueError as e:
                                pass
                        else:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))

    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)

@app.route('/classify')
def classify():
    with tf.Graph().as_default():

        with tf.Session() as sess:

            datadir = 'aligned'
            dataset = facenet.get_dataset(datadir)
            paths, labels = facenet.get_image_paths_and_labels(dataset)
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))

            print('Loading feature extraction model')
            modeldir = '20170511-185253/20170511-185253.pb'
            facenet.load_model(modeldir)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            batch_size = 1000
            image_size = 160
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

            classifier_filename = 'classifier/my_classifier.pkl'
            classifier_filename_exp = os.path.expanduser(classifier_filename)

            # Train classifier
            print('Training classifier')
            model = SVC(kernel='linear', probability=True)
            model.fit(emb_array, labels)

            # Create a list of class names
            class_names = [cls.name.replace('_', ' ') for cls in dataset]
            print(class_names)
            
            # Saving classifier model
            with open(classifier_filename_exp, 'wb') as outfile:
                pickle.dump((model, class_names), outfile)
            print('Saved classifier model to file "%s"' % classifier_filename_exp)
            print('Goodluck')

app.run(host='0.0.0.0', threaded=True)
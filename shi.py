from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage
from firebase_admin import db
import os, sys, json, shutil

def align(src, dir):

    output_dir_path = dir
    output_dir = os.path.expanduser(output_dir_path)
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    datadir = src
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

def classify(src, dir):
    with tf.Graph().as_default():

        with tf.Session() as sess:

            datadir = src
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

            classifier_filename = dir
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

# 引用私密金鑰
# path/to/serviceAccount.json 請用自己存放的路徑
cred = credentials.Certificate('/home/chengzu/facenetServer/shi.json')

firebase_admin.initialize_app(cred, {
    'databaseURL' : 'https://flutter-firebase-ffdb2.firebaseio.com/',
    'storageBucket': 'flutter-firebase-ffdb2.appspot.com'
})

# 初始化firestore
DB = firestore.client()

bucket = storage.bucket()
Pictures_dir = '/home/chengzu/facenetServer/Pictures/'

blobs = bucket.list_blobs()

#for blob in blobs:
#    print(blob.name)
#    #print(Pictures_dir + blob.name.split("/",1)[0])
#    if not os.path.exists(Pictures_dir + blob.name.split("/",1)[0]):
#        os.mkdir(Pictures_dir + blob.name.split("/",1)[0])

#    blob.download_to_filename(Pictures_dir + blob.name)  # Download


print('--------Download Sucessful--------')

ref = db.reference('classes')

dir = '/home/chengzu/facenetServer/image/'
src = '/home/chengzu/facenetServer/Pictures/'
align_dir = '/home/chengzu/facenetServer/align/'
classifier_dir = '/home/chengzu/facenetServer/classify/'

for key in ref.get(shallow=True):#shallow=True
    if(ref.child(key).child("students").get(shallow=True)):
        print(key+':')
        dl_dir = dir + key
        print(dl_dir)
        if os.path.exists(dl_dir):
            shutil.rmtree(dl_dir)#刪除資料夾
        os.mkdir(dl_dir)
        for j in ref.child(key).child("students").get(shallow=True):
            print(ref.child(key).child("students").child(j).child("nid").get())
            tempNid = ref.child(key).child("students").child(j).child("nid").get()
            tempDir = dl_dir + '/' + tempNid
            if os.path.exists(tempDir):
                shutil.rmtree(tempDir)#刪除資料夾
            shutil.copytree(src + tempNid, tempDir)
        align(dl_dir, align_dir + key)
        classify(align_dir + key, classifier_dir + key + '.pkl')
        print('-----' + key + '-----')

#for key in ref.get(shallow=True):
#    print(key+':')
#    dl_dir = dir + key
#    print(dl_dir)
#    if os.path.exists(dl_dir):
#        shutil.rmtree(dl_dir)#刪除資料夾
#    os.mkdir(dl_dir)#/home/chengzu/facenetServer/image/classno
#    for j in ref.child(key).get(shallow=True):
#        if j != 0:
#            print(ref.child(key).child('students').child(j).get())
#            tempNid = ref.child(key).child(j).get()
#            tempDir = dl_dir + '/' + tempNid
#            shutil.copytree(src + tempNid, tempDir)
#    align(dl_dir, align_dir + key)
#    classify(align_dir + key, classifier_dir + key + '.pkl')
#    print('-----' + key + '-----')


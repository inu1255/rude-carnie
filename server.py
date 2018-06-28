#!/usr/bin/env python
# coding=utf-8

from flask import Flask, request, jsonify
from scipy.misc import imread, imsave

app = Flask(__name__)

from datetime import datetime
import math
import time
from data import inputs
import numpy as np
import tensorflow as tf
from model import select_model, get_checkpoint
from utils import standardize_image
import os
import json
import csv
from face import FaceDetectorDlib
import inception_resnet_v1
import cv2

RESIZE_FINAL = 227
GENDER_LIST =['M','F']
AGE_LIST = [1,5,20,18,28,40,50,80]
# AGE_LIST = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']

def make_multi_crop_batch(image):
    crops = []
    image = tf.image.resize_images(image, (RESIZE_FINAL, RESIZE_FINAL))

    crop = tf.image.resize_images(image, (RESIZE_FINAL, RESIZE_FINAL))
    crops.append(standardize_image(crop))
    crops.append(standardize_image(tf.image.flip_left_right(crop)))

    corners = [ (0, 0) ]
    for corner in corners:
        ch, cw = corner
        cropped = tf.image.crop_to_bounding_box(image, ch, cw, RESIZE_FINAL, RESIZE_FINAL)
        crops.append(standardize_image(cropped))
        flipped = standardize_image(tf.image.flip_left_right(cropped))
        crops.append(standardize_image(flipped))
        
    image_batch = tf.stack(crops)
    return image_batch

def get_label(batch_results, age=True):
    label_list = AGE_LIST if age else GENDER_LIST
    output = batch_results[0]
    batch_sz = batch_results.shape[0]

    for i in range(1, batch_sz):
        output = output + batch_results[i]
    
    output /= batch_sz
    best = np.argmax(output)
    return label_list[best]

class Predict:
    def __init__(self, checkpoint_path='./checkpoints/age', label_list=AGE_LIST):
        self.checkpoint_path = checkpoint_path
        self.label_list = label_list
        self.init()

    def init(self):
        self.graph=tf.Graph() #为每个类(实例)单独创建一个graph
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess=tf.Session(graph=self.graph, config=config)
        with self.sess.as_default():
            with self.graph.as_default():
                self.images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
                model_fn = select_model('inception')
                nlabels = len(self.label_list)
                logits = model_fn(nlabels, self.images, 1, False)
                model_checkpoint_path, global_step = get_checkpoint(self.checkpoint_path)
                self.saver = tf.train.Saver()
                self.saver.restore(self.sess, model_checkpoint_path) #从恢复点恢复参数
                self.softmax_output = tf.nn.softmax(logits)

    def predict(self, aligned_image):
        with self.sess.as_default():
            with self.graph.as_default():
                aligned_image = make_multi_crop_batch(aligned_image)
                init = tf.global_variables_initializer()
                batch_results = self.sess.run(self.softmax_output, feed_dict={self.images: aligned_image.eval()})
                label_list = self.label_list
                output = batch_results[0]
                batch_sz = batch_results.shape[0]

                for i in range(1, batch_sz):
                    output = output + batch_results[i]
                
                output /= batch_sz
                best = np.argmax(output)
                return label_list[best]

class PredictAge:
    def __init__(self, checkpoint_path):
        self.graph=tf.Graph() #为每个类(实例)单独创建一个graph
        self.sess=tf.Session(graph=self.graph) #创建新的sess
        with self.sess.as_default():
            with self.graph.as_default():
                self.sess = tf.Session()
                self.images_pl = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input_image')
                images = tf.map_fn(lambda frame: tf.reverse_v2(frame, [-1]), self.images_pl) #BGR TO RGB
                images_norm = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), images)
                age_logits, gender_logits, _ = inception_resnet_v1.inference(images_norm, keep_probability=0.8, phase_train=False, weight_decay=1e-5)
                # gender = tf.argmax(tf.nn.softmax(gender_logits), 1)
                age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
                self.age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
                init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
                self.sess.run(init_op)
                saver = tf.train.Saver()
                ckpt = tf.train.get_checkpoint_state(checkpoint_path)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(self.sess, ckpt.model_checkpoint_path)
                    print("restore and continue training!")
                    print(ckpt.model_checkpoint_path)
    def predict(self, aligned_images):
        with self.sess.as_default():
            with self.graph.as_default():
                aligned_images = np.array([cv2.resize(image, (160, 160)) for image in aligned_images])
                ages = self.sess.run(self.age, feed_dict={self.images_pl: aligned_images})
                return ages

# age = Predict('./checkpoints/age', AGE_LIST)
age = PredictAge('./checkpoints/age1')
gender = Predict('./checkpoints/gender', GENDER_LIST)
detector = FaceDetectorDlib('data/shape_predictor_68_face_landmarks.dat')

@app.route('/detection', methods=['POST','GET'])
def detection():
    f = None
    if request.method == 'POST':
        f = request.files.get('f')
    else:
        f = request.args.get('f')
    if f is None:
        return jsonify({'no': 400, "msg": "缺少文件参数"})
    try:
        image = imread(f, mode='RGB')
        # image = cv2.imread(f, cv2.IMREAD_COLOR)
    except Exception as e:
        print(e)
        return jsonify({'no': 404, 'msg': '找不到文件'})
    aligned_images, XY = detector.run(image)
    data = []
    if aligned_images is not None and len(aligned_images)>0:
        ages = age.predict(aligned_images)
        for i in range(len(aligned_images)):
            aligned_image = aligned_images[i]
            # data.append({'rect':XY[i],'age':age.predict(aligned_image),'gender':gender.predict(aligned_image)})
            data.append({'rect':XY[i],'age':float(ages[i]),'gender':gender.predict(aligned_image)})

    return jsonify({'no':200,'data':data})

if __name__ == '__main__':
    app.run('127.0.0.1', 3009)
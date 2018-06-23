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
        self.sess=tf.Session(graph=self.graph) #创建新的sess
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

age = Predict('./checkpoints/age', AGE_LIST)
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
    for i in range(len(aligned_images)):
        aligned_image = aligned_images[i]
        data.append({'rect':XY[i],'age':age.predict(aligned_image),'gender':gender.predict(aligned_image)})
    return jsonify({'no':200,'data':data})

if __name__ == '__main__':
    app.run('127.0.0.1', 3009)
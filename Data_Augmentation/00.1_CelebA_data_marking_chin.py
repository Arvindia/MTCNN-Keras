# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 13:53:00 2019

@author: TMaysGGS
"""

'''Last updated on 06/11/2019 17:00'''
import os
import cv2
import dlib
import time
import pickle as pkl

image_path = r'E:\Datasets\CelebA\Img\img_celeba'
output_path = r'E:\Datasets\CelebA\Img\img_celeba_dlibed'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

annotation_list = []
for img_name in os.listdir(image_path):
    
    # 47342 images have been processed, so the name of the last image processed is '047342.jpg'.
    if int(img_name[0: 6] <= 47342):
        continue
    
    annotation = [img_name]
    img = cv2.imread(os.path.join(image_path, img_name))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    start = time.time()
    dets = detector(gray, 1)
    end = time.time()
    for face in dets:
        shape = predictor(img, face)  # 寻找人脸的68个标定点
        '''
        # 左眼外眼角
        cv2.circle(img, (shape.parts()[36].x, shape.parts()[36].y), 2, (0, 255, 0), 1)
        # 右眼外眼角
        cv2.circle(img, (shape.parts()[45].x, shape.parts()[45].y), 2, (0, 255, 0), 1)
        '''
        # 下巴
        cv2.circle(img, (shape.parts()[8].x, shape.parts()[8].y), 2, (0, 255, 0), 1)
        '''
        cv2.imshow("image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        # annotation.extend([shape.parts()[36].x, shape.parts()[36].y, shape.parts()[45].x, shape.parts()[45].y, shape.parts()[8].x, shape.parts()[8].y])
        annotation.extend([shape.parts()[8].x, shape.parts()[8].y])
    annotation_list.append(annotation)
    
    cv2.imwrite(os.path.join(output_path, img_name), img)
    print("Total time spent on detecting face landmarks is " + str(end - start) + "s.")

# 下巴：8；左眼外眼角：36；右眼外眼角：45；鼻尖：30；左嘴角：48；右嘴角：54

file = open(r'E:\Datasets\CelebA\Img\chin_annotations.pkl', 'wb+')
pkl.dump(annotation_list, file)
file.close()

with open('E:\Datasets\CelebA\Img\chin_annotations.pkl', 'rb') as f:
    chin_annos_loaded = pkl.load(f)
    
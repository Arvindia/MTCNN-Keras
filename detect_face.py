#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 02:20:29 2019

@author: tmaysggs
"""

'''Last updated on 07/06/2019 10:36'''
import time
import cv2 
import numpy as np
from keras.models import load_model

import utils

Pnet = load_model('./Models/Pnet.h5')
Rnet = load_model('./Models/Rnet.h5')
Onet = load_model('./Models/Onet.h5')
NewOnet = load_model('./Models/NewOnet.h5')

def detect_face(img, thresholds): 
    
    '''P-Net Prediction'''
    temp = (img.copy() - 127.5) / 127.5 
    orig_h, orig_w, orig_c = temp.shape
    # 生成用于制作图像金字塔的缩放比例列表
    scales = utils.calculate_scales(temp)
    Pnet_outputs = []
    t0 = time.time()
    
    # 生成图像金字塔列表并逐一预测结果
    for scale in scales:
        scale_h = int(orig_h * scale)
        scale_w = int(orig_w * scale)
        scaled_img = cv2.resize(temp, (scale_w, scale_h)) # OpenCV中宽在前
        input_img = scaled_img.reshape(1, *scaled_img.shape) # reshape to (1, scale_h, scale_w, orig_c)
        pred = Pnet.predict(input_img) # pred is a list of 2 arrays with the shapes (1, ?, ?, 2) & (1, ?, ?, 4)
        Pnet_outputs.append(pred) 
    img_num = len(scales)
    
    rectangles_list = []
    for i in range(img_num):
        prob = Pnet_outputs[i][0][0][:, :, 1] # 是“人脸”的置信度，对应前面(1, ?, ?, 2)中的(?, ?)
        roi = Pnet_outputs[i][1][0] # 人脸框的坐标偏移比例，对应前面(1, ?, ?, 4)中的(?, ?, 4)
        
        out_h, out_w = prob.shape # 每个点的值对应一个12 x 12框是否有”人“的置信度
        out_side = max(out_h, out_w) # ???
        
        prob = np.swapaxes(prob, 0, 1) 
        roi = np.swapaxes(roi, 0, 2) # shape变为(4, ?, ?)
        rectangles = utils.pnet_detect_face(prob, roi, out_side, 1 / scales[i], orig_w, orig_h, thresholds[0])
        rectangles_list.extend(rectangles) # 每个rectangles包含(num, x1, y1, x2, y2, score)
    
    rectangles_list = utils.NMS(rectangles_list, 0.7, 'iou') 
    
    t1 = time.time()
    print("Time for P-Net is " + str(t1 - t0))
    
    if len(rectangles_list) == 0:
        return rectangles_list
    
    '''R-Net Prediction'''
    cropping_count = 0 # 记录对该张图片的裁取次数 
    Rnet_inputs = [] 
    
    for rectangle in rectangles_list:
        cropped_img = temp[int(rectangle[1]): int(rectangle[3]), int(rectangle[0]): int(rectangle[2])]
        scaled_img = cv2.resize(cropped_img, (24, 24))
        Rnet_inputs.append(scaled_img)
        cropping_count += 1
        
    Rnet_inputs = np.array(Rnet_inputs)
    Rnet_outputs = Rnet.predict(Rnet_inputs)
    prob = Rnet_outputs[0]
    roi = Rnet_outputs[1]
    prob = np.array(prob)
    roi = np.array(roi)
    
    rectangles_list = utils.rnet_detect_face(prob, roi, rectangles_list, orig_w, orig_h, thresholds[1])
    
    t2 = time.time()
    print("Time for R-Net is " + str(t2 - t1))
    
    if len(rectangles_list) == 0:
        return rectangles_list
    
    '''O-Net Prediction'''
    cropping_count = 0
    Onet_inputs = []
    
    for rectangle in rectangles_list:
        cropped_img = temp[int(rectangle[1]): int(rectangle[3]), int(rectangle[0]): int(rectangle[2])] 
        scaled_img = cv2.resize(cropped_img, (48, 48))
        Onet_inputs.append(scaled_img)
        cropping_count += 1
        
    Onet_inputs = np.array(Onet_inputs)
    Onet_outputs = Onet.predict(Onet_inputs)
    prob = Onet_outputs[0]
    roi = Onet_outputs[1]
    pts = Onet_outputs[2]
    
    rectangles = utils.onet_detect_face(prob, roi, pts, rectangles_list, orig_w, orig_h, thresholds[2])
    
    t3 = time.time()
    print("Time for O-Net is " + str(t3 - t2))
    
    return rectangles

def detect_face_with_chin(img, thresholds): 
    
    '''P-Net Prediction'''
    temp = (img.copy() - 127.5) / 127.5 
    orig_h, orig_w, orig_c = temp.shape
    # 生成用于制作图像金字塔的缩放比例列表
    scales = utils.calculate_scales(temp)
    Pnet_outputs = []
    t0 = time.time()
    
    # 生成图像金字塔列表并逐一预测结果
    for scale in scales:
        scale_h = int(orig_h * scale)
        scale_w = int(orig_w * scale)
        scaled_img = cv2.resize(temp, (scale_w, scale_h)) # OpenCV中宽在前
        input_img = scaled_img.reshape(1, *scaled_img.shape) # reshape to (1, scale_h, scale_w, orig_c)
        pred = Pnet.predict(input_img) # pred is a list of 2 arrays with the shapes (1, ?, ?, 2) & (1, ?, ?, 4)
        Pnet_outputs.append(pred) 
    img_num = len(scales)
    
    rectangles_list = []
    for i in range(img_num):
        prob = Pnet_outputs[i][0][0][:, :, 1] # 是“人脸”的置信度，对应前面(1, ?, ?, 2)中的(?, ?)
        roi = Pnet_outputs[i][1][0] # 人脸框的坐标偏移比例，对应前面(1, ?, ?, 4)中的(?, ?, 4)
        
        out_h, out_w = prob.shape # 每个点的值对应一个12 x 12框是否有”人“的置信度
        out_side = max(out_h, out_w) # ???
        
        prob = np.swapaxes(prob, 0, 1) 
        roi = np.swapaxes(roi, 0, 2) # shape变为(4, ?, ?)
        rectangles = utils.pnet_detect_face(prob, roi, out_side, 1 / scales[i], orig_w, orig_h, thresholds[0])
        rectangles_list.extend(rectangles) # 每个rectangles包含(num, x1, y1, x2, y2, score)
    
    rectangles_list = utils.NMS(rectangles_list, 0.7, 'iou') 
    
    t1 = time.time()
    print("Time for P-Net is " + str(t1 - t0))
    
    if len(rectangles_list) == 0:
        return rectangles_list
    
    '''R-Net Prediction'''
    cropping_count = 0 # 记录对该张图片的裁取次数 
    Rnet_inputs = [] 
    
    for rectangle in rectangles_list:
        cropped_img = temp[int(rectangle[1]): int(rectangle[3]), int(rectangle[0]): int(rectangle[2])]
        scaled_img = cv2.resize(cropped_img, (24, 24))
        Rnet_inputs.append(scaled_img)
        cropping_count += 1
        
    Rnet_inputs = np.array(Rnet_inputs)
    Rnet_outputs = Rnet.predict(Rnet_inputs)
    prob = Rnet_outputs[0]
    roi = Rnet_outputs[1]
    prob = np.array(prob)
    roi = np.array(roi)
    
    rectangles_list = utils.rnet_detect_face(prob, roi, rectangles_list, orig_w, orig_h, thresholds[1])
    
    t2 = time.time()
    print("Time for R-Net is " + str(t2 - t1))
    
    if len(rectangles_list) == 0:
        return rectangles_list
    
    '''O-Net Prediction'''
    cropping_count = 0
    Onet_inputs = []
    
    for rectangle in rectangles_list:
        cropped_img = temp[int(rectangle[1]): int(rectangle[3]), int(rectangle[0]): int(rectangle[2])] 
        scaled_img = cv2.resize(cropped_img, (48, 48))
        Onet_inputs.append(scaled_img)
        cropping_count += 1
        
    Onet_inputs = np.array(Onet_inputs)
    Onet_outputs = NewOnet.predict(Onet_inputs)
    prob = Onet_outputs[0]
    roi = Onet_outputs[1]
    pts = Onet_outputs[2]
    chinpts = Onet_outputs[3]

    rectangles = utils.new_onet_detect_face(prob, roi, pts, chinpts, rectangles_list, orig_w, orig_h, thresholds[2])
    
    t3 = time.time()
    print("Time for O-Net is " + str(t3 - t2))
    
    return rectangles
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 16:26:07 2019

@author: TMaysGGS
"""

'''Last updated on 11/15/2019 14:22'''
import sys
import time
import os
import cv2 
import random 
import argparse 
import pickle as pkl
import numpy as np

sys.path.append("../")
import utils
from MTCNN_models import pnet 

'''Helper functions'''
# P-Net Prediction
def pnet_prediction(img, PNet, thresholds):
    
    temp = img.copy() / 255. 
    orig_h, orig_w, orig_c = temp.shape 
    # 生成用于制作图像金字塔的缩放比例列表
    scales = utils.calculate_scales(temp)
    PNet_outputs = []

    t0 = time.time()
    # 生成图像金字塔列表并逐一预测结果
    for scale in scales:
        scale_h = int(orig_h * scale)
        scale_w = int(orig_w * scale)
        scaled_img = cv2.resize(temp, (scale_w, scale_h)) # OpenCV中宽在前
        input_img = scaled_img.reshape(1, *scaled_img.shape) # reshape to (1, scale_h, scale_w, orig_c)
        pred = PNet.predict(input_img) # pred is a list of 2 arrays with the shapes (1, ?, ?, 2) & (1, ?, ?, 4)
        PNet_outputs.append(pred) 
    img_num = len(scales)
    
    rectangles_list = []
    for i in range(img_num):
        prob = PNet_outputs[i][0][0][:, :, 0] # 是“人脸”的置信度，对应前面(1, ?, ?, 1)中的(?, ?)
        roi = PNet_outputs[i][1][0] # 人脸框的坐标偏移比例，对应前面(1, ?, ?, 4)中的(?, ?, 4)
        
        out_h, out_w = prob.shape # 每个点的值对应一个12 x 12框是否有”人“的置信度
        out_side = max(out_h, out_w) # ???
        
        prob = np.swapaxes(prob, 0, 1) 
        roi = np.swapaxes(roi, 0, 2) # shape变为(4, ?, ?)
        rectangles = utils.pnet_detect_face(prob, roi, out_side, 1 / scales[i], orig_w, orig_h, thresholds[0])
        rectangles_list.extend(rectangles) # 每个rectangles包含(num, x1, y1, x2, y2, score)
    
    rectangles_list = utils.NMS(rectangles_list, 0.7, 'iou') 
    
    t1 = time.time()
    print("Inference time for P-Net is " + str(t1 - t0))

    return rectangles_list

'''Generating hard negatvie samples & saving''' 
def main(args):
    
    IMG_SIZE = args.IMG_SIZE
    print("IMG_SIZE:", IMG_SIZE)
    if IMG_SIZE != 12 and IMG_SIZE != 24 and IMG_SIZE != 48: 
        raise Exception("Image size wrong!")
    IMG_ROOT_DIR = args.IMG_ROOT_DIR 
    if args.DEBUG == 1: 
        DEBUG = True 
    else: 
    	DEBUG = False 

    thresholds = [0.6, 0.6, 0.7]
    
    PNet = pnet(training = False)
    PNet.load_weights('../Models/PNet.h5')
    PNet.summary() 
    
    RECORD_PATH = os.path.join(IMG_ROOT_DIR, 'wider_face_add_lm_10_10_info.pkl') 
    f= open(RECORD_PATH, 'rb') 
    info = pkl.load(f) 
    f.close()
    if DEBUG: 
        info = info[: 100] 
    
    neg_hard_save_dir = r'../Data/' + str(IMG_SIZE) + '/neg_hard' 
    if not os.path.exists(neg_hard_save_dir): 
        os.mkdir(neg_hard_save_dir)
    
    neg_hard_idx = 0 
    neg_hard_list = [] 
    
    for i in range(len(info)):
        
        pic_info = info[i]
        
        img_path = os.path.join(IMG_ROOT_DIR, 'JPEGImages', pic_info[0]) 
        img = cv2.imread(img_path) 
        height, width, channel = img.shape
        
        bboxes = np.array(pic_info[2]) # total bounding boxes in one picture 
        
        if (i + 1) % 1000 == 0: 
            print(str(i + 1) + " pics processed ") 
            print(str(neg_hard_idx) + " hard negative samples generated. ") 
    
        # Generate negative hard samples
        if IMG_SIZE == 12: 
            pred_rectangles_list = pnet_prediction(img, PNet, thresholds) 
        
        if args.AUGMENT_CONTROL > 0: 
            random.shuffle(pred_rectangles_list) 
            pred_rectangles_list = pred_rectangles_list[: args.AUGMENT_CONTROL] 
        
        pred_rectangles = np.array(pred_rectangles_list) 
        pred_boxes = utils.rect2square(pred_rectangles) 
        
        for j in range(len(pred_boxes)):
            
            pred_box = pred_boxes[j] 
            x1, y1, x2, y2 = np.array(pred_box[: 4]).astype(int)
            w = x2 - x1 + 1 
            h = y2 - y1 + 1 
            
            # Drop the box that exceeds the boundary or is too small 
            if w < 20 or h < 20 or x1 < 0 or y1 < 0 or x2 > width - 1 or y2 > height - 1: 
                continue
            
            crop_box = np.array([x1, y1, x2, y2])
            if bboxes.shape[0] == 0:
                iou = 0
            else:
                iou = utils.IoU(crop_box, bboxes)
            
            if np.max(iou) < 0.1: 
                
                cropped_img = img[y1: y2, x1: x2]
                resized_img = cv2.resize(cropped_img, (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_LINEAR)
                saving_path = os.path.join(neg_hard_save_dir, str(neg_hard_idx) + '.jpg')
                success = cv2.imwrite(saving_path, resized_img)
                if not success: 
                    raise Exception("Neg picture " + str(neg_hard_idx) + " saving failed. ") 
                
                img_name = os.path.join('neg_hard', str(neg_hard_idx) + '.jpg')
                label = 0 
                roi = np.array([-1] * 4)
                landmark = np.array([-1] * 12)
                
                neg_hard_list.append([img_name, label, roi, landmark]) 
                
                neg_hard_idx = neg_hard_idx + 1
                    
    neg_hard_anno_path = r'../Data/' + str(IMG_SIZE) + '/neg_hard_record.pkl' 
    
    file = open(neg_hard_anno_path, 'wb+') 
    pkl.dump(neg_hard_list, file)
    file.close() 
    
    print("File saving done. ") 

def parse_arguments(argv):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--IMG_SIZE', type = int, help = 'The input image size', default = 12)
    parser.add_argument('--IMG_ROOT_DIR', type = str, help = 'The image data directory', default = r'../Data/wider_face_add_lm_10_10') 
    parser.add_argument('--DEBUG', type = int, help = 'Debug mode or not', default = 0) 
    parser.add_argument('--AUGMENT_CONTROL', type = int, help = 'Limit the number of augmented pics', default = 100) 
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    
    main(parse_arguments(sys.argv[1:]))
    
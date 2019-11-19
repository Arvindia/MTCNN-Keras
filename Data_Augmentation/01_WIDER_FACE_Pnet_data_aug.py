# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 16:26:07 2019

@author: TMaysGGS
"""

'''Last updated on 11/20/2019 01:10'''
import sys
import os
import cv2 
import pickle as pkl
import numpy as np

sys.path.append("..")
from utils import IoU

IMG_SIZE = 12 
BASE_NUM = 3 
PROB_THRESH = 0.15
RECORD_PATH = r'../Data/WIDER FACE/WIDER_FACE_info.pkl'
f= open(RECORD_PATH, 'rb') 
info = pkl.load(f) 
f.close()

save_dir = r'../Data/' + str(IMG_SIZE)
pos_save_dir = r'../Data/' + str(IMG_SIZE) + '/pos' 
part_save_dir = r'../Data/' + str(IMG_SIZE) + '/part' 
neg_save_dir = r'../Data/' + str(IMG_SIZE) + '/neg' 
if not os.path.exists(save_dir): 
    os.mkdir(save_dir)
if not os.path.exists(pos_save_dir): 
    os.mkdir(pos_save_dir)
if not os.path.exists(part_save_dir): 
    os.mkdir(part_save_dir)
if not os.path.exists(neg_save_dir): 
    os.mkdir(neg_save_dir)

neg_idx = 0 
pos_idx = 0 
part_idx = 0

neg_list = [] 
pos_list = [] 
part_list = [] 
for i in range(len(info)):
    
    pic_info = info[i]
    
    img = cv2.imread(pic_info[0]) 
    height, width, channel = img.shape
    
    bboxes = np.array(pic_info[1]) # total bounding boxes in one picture 
    probs = np.array(pic_info[2])
    
    if (i + 1) % 1000 == 0: 
        print(str(i + 1) + " pics processed ") 
        print(str(pos_idx) + " positive, " + str(neg_idx) + " negative and " + str(part_idx) + " part-face samples generated. ") 

    # Generate negative background samples
    neg_num = 0
    while neg_num < 50 * BASE_NUM: 
        
        new_size = np.random.randint(IMG_SIZE, min(width, height) / 2) 
        new_x1 = np.random.randint(0, width - new_size) 
        new_y1 = np.random.randint(0, height - new_size) 
        new_x2 = new_x1 + new_size 
        new_y2 = new_y1 + new_size 
        crop_box = np.array([new_x1, new_y1, new_x2, new_y2]) 
        iou = IoU(crop_box, bboxes) 
        
        if np.max(iou) < 0.3: 
            
            cropped_img = img[new_y1: new_y2, new_x1: new_x2] 
            resized_img = cv2.resize(cropped_img, (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_LINEAR) 
            saving_path = os.path.join(neg_save_dir, str(neg_idx) + '.jpg') 
            success = cv2.imwrite(saving_path, resized_img)
            if not success: 
                raise Exception("Neg picture " + str(neg_idx) + " saving failed. ") 
            
            img_name = os.path.join('neg', str(neg_idx) + '.jpg') 
            label = 0 
            roi = np.array([-1] * 4)
            landmark = np.array([-1] * 12)
            
            neg_list.append([img_name, label, roi, landmark]) 
            
            neg_num = neg_num + 1
            neg_idx = neg_idx + 1 
            
    for j in range(len(bboxes)):
        
        bbox = bboxes[j]
        x1, y1, x2, y2 = np.squeeze(bbox)
        w = x2 - x1 + 1
        h = y2 - y1 + 1 
        prob = probs[j]
        
        # Drop the box that exceeds the boundary or is too small 
        if max(w, h) < 40 or x1 < 0 or y1 < 0 or prob < PROB_THRESH: 
            continue 
        
        # Generate negative samples around face area 
        neg_num = 0
        accident_control = 0
        while neg_num < 10 * BASE_NUM: # 6
            
            # Avoid dead loop for some faces around the corner 
            if accident_control > 20: 
                break 
            
            new_size = np.random.randint(IMG_SIZE, min(width, height) / 2) # 12 ~ shorter length 
            delta_x = np.random.randint(max(-new_size, -x1), w) 
            delta_y = np.random.randint(max(-new_size, -y1), h) 
            new_x1 = int(max(0, x1 + delta_x)) 
            new_y1 = int(max(0, y1 + delta_y)) 
            new_x2 = new_x1 + new_size 
            new_y2 = new_y1 + new_size 
            
            if new_x2 > width or new_y2 > height: 
                accident_control = accident_control + 1 
                continue 
            
            neg_num = neg_num + 1
            
            crop_box = np.array([new_x1, new_y1, new_x2, new_y2]) 
            iou = IoU(crop_box, bboxes) 
            
            if np.max(iou) < 0.3: 
                
                cropped_img = img[new_y1: new_y2, new_x1: new_x2] 
                resized_img = cv2.resize(cropped_img, (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_LINEAR) 
                saving_path = os.path.join(neg_save_dir, str(neg_idx) + '.jpg') 
                success = cv2.imwrite(saving_path, resized_img) 
                if not success: 
                    raise Exception("Neg picture " + str(neg_idx) + " saving failed. ") 
                    
                img_name = os.path.join('neg', str(neg_idx) + '.jpg') 
                label = 0 
                roi = np.array([-1] * 4) 
                landmark = np.array([-1] * 12)
                
                neg_list.append([img_name, label, roi, landmark]) 
                
                neg_idx = neg_idx + 1 

        # Generate positive & part-face samples 
        pos_part_num = 0 
        accident_control = 0
        while pos_part_num < 40 * BASE_NUM: # 25
            
            if accident_control > 20: 
                break
            
            new_size = np.random.randint(int(min(w, h) * 0.8), np.ceil(max(w, h) * 1.25)) 
            
            # When w = 4, range is -0.8~0.8, np.random.randint(-0.8, 0.8) is equal to np.random.randint(-0, -1), which will raise error.
            # Thus enlarge the range of delta x/y by 1
            
            delta_x = np.random.randint(-w * 0.2 - 1, w * 0.2 + 1) 
            delta_y = np.random.randint(-h * 0.2 - 1, h * 0.2 + 1) 
            new_x1 = int(max(x1 + w / 2 - new_size / 2 + delta_x, 0)) 
            new_y1 = int(max(y1 + h / 2 - new_size / 2 + delta_y, 0)) 
            new_x2 = new_x1 + new_size 
            new_y2 = new_y1 + new_size 
            
            if new_x2 > width or new_y2 > height: 
                accident_control = accident_control + 1
                continue 
            
            crop_box = np.array([new_x1, new_y1, new_x2, new_y2]) 
            iou = IoU(crop_box, bbox.reshape(1, -1)) # bbox is one of bboxes so it has the shape (4, ) and needs to be reshape to (1, 4)  
            
            if iou >= 0.65: 
                
                cropped_img = img[new_y1: new_y2, new_x1: new_x2] 
                resized_img = cv2.resize(cropped_img, (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_LINEAR) 
                offset_x1 = (x1 - new_x1) / float(new_size) 
                offset_y1 = (y1 - new_y1) / float(new_size) 
                offset_x2 = (x2 - new_x2) / float(new_size) 
                offset_y2 = (y2 - new_y2) / float(new_size) 
                
                saving_path = os.path.join(pos_save_dir, str(pos_idx) + '.jpg') 
                success = cv2.imwrite(saving_path, resized_img) 
                if not success: 
                    raise Exception("Pos picture " + str(pos_idx) + " saving failed. ") 
                
                img_name = os.path.join('pos', str(pos_idx) + '.jpg') 
                label = 1
                roi = np.array([offset_x1, offset_y1, offset_x2, offset_y2]) 
                landmark = np.array([-1] * 12)
                
                pos_list.append([img_name, label, roi, landmark]) 
                
                pos_idx = pos_idx + 1 
                
            elif iou >= 0.4: 
                
                cropped_img = img[new_y1: new_y2, new_x1: new_x2] 
                resized_img = cv2.resize(cropped_img, (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_LINEAR) 
                offset_x1 = (x1 - new_x1) / float(new_size) 
                offset_y1 = (y1 - new_y1) / float(new_size) 
                offset_x2 = (x2 - new_x2) / float(new_size) 
                offset_y2 = (y2 - new_y2) / float(new_size) 
                
                saving_path = os.path.join(part_save_dir, str(part_idx) + '.jpg') 
                success = cv2.imwrite(saving_path, resized_img) 
                if not success: 
                    raise Exception("Part picture " + str(part_idx) + " saving failed. ") 
                
                img_name = os.path.join('part', str(part_idx) + '.jpg') 
                label = -1
                roi = np.array([offset_x1, offset_y1, offset_x2, offset_y2]) 
                landmark = np.array([-1] * 12)
                
                part_list.append([img_name, label, roi, landmark]) 
                
                part_idx = part_idx + 1 
            
            pos_part_num = pos_part_num + 1 
                
neg_anno_path = r'../Data/' + str(IMG_SIZE) + '/neg_record.pkl' 
pos_anno_path = r'../Data/' + str(IMG_SIZE) + '/pos_record.pkl' 
part_anno_path = r'../Data/' + str(IMG_SIZE) + '/part_record.pkl' 

file = open(neg_anno_path, 'wb+') 
pkl.dump(neg_list, file)
file.close() 

file = open(pos_anno_path, 'wb+') 
pkl.dump(pos_list, file)
file.close() 

file = open(part_anno_path, 'wb+') 
pkl.dump(part_list, file)
file.close() 

print("File saving done. ") 

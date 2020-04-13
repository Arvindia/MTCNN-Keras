# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 09:49:52 2019

@author: TMaysGGS
"""

"""Last updated on 2019.11.07 16:07""" 
import os 
import pickle as pkl 
import xml.etree.ElementTree as ET

DATA_DIR = r'../Data/wider_face_add_lm_10_10' 
IMG_DIR = os.path.join(DATA_DIR, 'JPEGImages')
ANNOTATION_DIR = os.path.join(DATA_DIR, 'Annotations') 
CLASS_DICT = {"BACKGROUND": 0, "face": 1} 

def get_annotations(annotation_file_path): 
    
    objects = ET.parse(annotation_file_path).findall("object") 
    
    labels = [] 
    boxes = [] 
    is_difficult = [] 
    for obj in objects: 
        class_name = obj.find('name').text.lower().strip() 
        if class_name in CLASS_DICT: 
            
            labels.append(CLASS_DICT[class_name])
            
            bbox = obj.find('bndbox') 
            # Origin Comment: VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text) - 1 
            y1 = float(bbox.find('ymin').text) - 1 
            x2 = float(bbox.find('xmax').text) - 1 
            y2 = float(bbox.find('ymax').text) - 1 
            boxes.append([x1, y1, x2, y2]) 
            
            is_difficult_str = obj.find('difficult').text 
            is_difficult.append(int(is_difficult_str) if is_difficult_str else 0) 
    
    return labels, boxes, is_difficult 

img_name_list = os.listdir(IMG_DIR) 
info = [] 
for img_name in img_name_list: 
    
    img_clean_name = img_name[: -4] 
    annotation_file_name = img_clean_name + '.xml' 
    annotation_file_path = os.path.join(ANNOTATION_DIR, annotation_file_name)
    if os.path.exists(annotation_file_path): 
        
        labels, boxes, is_difficult = get_annotations(annotation_file_path) 
        
        info.append([img_name, labels, boxes, is_difficult]) 

record_path = '../Data/wider_face_add_lm_10_10/wider_face_add_lm_10_10_info.pkl' 
file = open(record_path, 'wb+') 
pkl.dump(info, file)
file.close() 

"""Verify the number of hard-detected faces"""
# Actually 0 in this data set 
import numpy as np 

total_face_count = 0 
difficult_face_count = 0
for img_info in info:  
    is_difficult_info = img_info[3]
    total_face_count = total_face_count + len(is_difficult_info) 
    difficult_face_count = difficult_face_count + np.sum(is_difficult_info) 
    if 1 in is_difficult_info: 
        print("1")
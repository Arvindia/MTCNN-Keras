# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 14:20:42 2019

@author: TMaysGGS
"""

import cv2
import pickle as pkl

# Load the bbox info
RECORD_PATH = r'../Data/WIDER FACE/WIDER_FACE_info.pkl'
f= open(RECORD_PATH, 'rb') 
bbox_info = pkl.load(f) 
f.close()
assert(len(bbox_info) == 12880)

# Load the prob info
f = open(r'../Data/WIDER FACE/prob.txt')
prob_info = f.readlines()
f.close()
assert(len(prob_info) == 12880)

# Integration
new_info_list = []
for i in range(len(bbox_info)):
    
    temp_path0 = bbox_info[i][0]
    temp_bboxes = bbox_info[i][1: ]
    img_from_bbox_info = cv2.imread(temp_path0)
    
    temp_prob_info = prob_info[i].split()
    temp_path1 = '../Data/WIDER FACE' + temp_prob_info[0][6: ]
    img_from_prob_info = cv2.imread(temp_path1)
    assert(img_from_bbox_info.all() == img_from_prob_info.all())
    
    temp_probs = []
    for j in range(len(temp_prob_info[1: ])):
        temp_probs.append(float(temp_prob_info[1: ][j]))
        
    new_info_list.append([temp_path0, temp_bboxes, temp_probs])

NEW_RECORD_PATH = r'../Data/WIDER FACE/WIDER_FACE_info.pkl'
file = open(NEW_RECORD_PATH, 'wb+') 
pkl.dump(new_info_list, file)
file.close() 
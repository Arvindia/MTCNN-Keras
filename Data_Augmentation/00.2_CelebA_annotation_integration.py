# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 01:10:03 2019

@author: TMaysGGS
"""

'''Last updated on 08/07/2019 01:27'''
'''Importing the libraries'''
import pickle as pkl
import numpy as np

'''Loading annotations'''
# Read the CelebA bounding box annotations
with open(r'../Data/CelebA/Anno/list_bbox_celeba.txt', 'rb') as f:
    temp_info = f.readlines()[2: ]
for i in range(len(temp_info)):
    pic_info = temp_info[i].split()
    
# Load the chinmarks annotations
with open(r'../Data/CelebA/Anno/chinmarks_celeba.pkl', 'rb') as f:
    chinmarks = pkl.load(f)

'''Integrating the annotations'''
# Re-order the chinmarks list
index_list = []
for i in range(len(chinmarks)):
    
    anno = chinmarks[i]
    img_name = anno[0]
    index = int(img_name[ : 6])
    index_list.append(index)

reorder_list = np.argsort(np.array(index_list))

aranged_chinmarks = []
for i in range(len(reorder_list)):
    
    aranged_chinmarks.append(chinmarks[reorder_list[i]])


# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 16:15:38 2019

@author: TMaysGGS
"""

'''Last updated on 11/14/2019 16:49'''
import sys 

sys.path.append('../')
from MTCNN_models import pnet, rnet, onet

PNet_train_with_lm_path = r'PNet_train_with_lm.h5' 
def convert_training_model_without_landmark(PNet_train_with_lm_path): 
    
    PNet_train_without_lm = pnet(training = True, train_with_landmark = False) 
    
    PNet_train_with_lm = pnet(training = True, train_with_landmark = True) 
    PNet_train_with_lm.load_weights(PNet_train_with_lm_path) 
    
    temp_weights_list = []
    for layer in PNet_train_with_lm.layers:
        
        temp_layer = PNet_train_with_lm.get_layer(layer.name)
        temp_weights = temp_layer.get_weights()
        temp_weights_list.append(temp_weights)
    
    for i in range(len(PNet_train_without_lm.layers) - 3):
        
        PNet_train_without_lm.get_layer(PNet_train_without_lm.layers[i].name).set_weights(temp_weights_list[i]) 
    
    return PNet_train_without_lm 

PNet_train_without_lm = convert_training_model_without_landmark(PNet_train_with_lm_path) 
PNet_train_without_lm.save('PNet_train_without_lm.h5')

TRAIN_WITH_LM = False 
train_model_path = 'PNet_trained_without_lm_epoch_1_iter_600.h5' 
inference_model_path = 'PNet.h5' 
def convert_model_for_inference(train_with_lm, train_model_path, inference_model_path): 
    
    PNet_train = pnet(training = True, train_with_landmark = train_with_lm) 
    PNet = pnet(training = False)  
    
    PNet_train.load_weights(train_model_path) 
    # PNet_train.summary() 

    temp_weights_list = []
    for layer in PNet_train.layers:
        
        temp_layer = PNet_train.get_layer(layer.name)
        temp_weights = temp_layer.get_weights()
        temp_weights_list.append(temp_weights)
    
    for i in range(len(PNet.layers)):
        
        PNet.get_layer(PNet.layers[i].name).set_weights(temp_weights_list[i])
    
    PNet.save(inference_model_path) 
    
    return PNet_train, PNet 

PNet_train, PNet = convert_model_for_inference(TRAIN_WITH_LM, train_model_path, inference_model_path) 

'''
import numpy as np
import cv2

X = np.random.rand(1, 12, 12, 3) 
preds1 = PNet_train_with_lm.predict(X) 
preds2 = PNet_train_without_lm.predict(X) 

preds1 = PNet_train.predict(X) 
preds2 = PNet.predict(X) 

image = cv2.imread(r'../Data/12/pos/3.jpg')
X = image.reshape(1, 12, 12, 3).astype(np.float) / 255.
Y_pred = PNet_train_with_lm.predict(X)

'''

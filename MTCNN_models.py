# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 09:59:45 2019

@author: TMaysGGS
"""

'''Last updated on 04/09/2020 15:40'''
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, PReLU, MaxPooling2D, Reshape, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.regularizers import l2

'''P-Net'''
def pnet_train1(train_with_landmark = False):
    
    X = Input(shape = (12, 12, 3), name = 'Pnet_input')
    
    M = Conv2D(10, 3, strides = 1, padding = 'valid', kernel_initializer = glorot_normal, kernel_regularizer = l2(0.00001), name = 'Pnet_conv1')(X)
    M = PReLU(shared_axes = [1, 2], name = 'Pnet_prelu1')(M)
    M = MaxPooling2D(pool_size = 2, name = 'Pnet_maxpool1')(M) # default 'pool_size' is 2!!! 
    
    M = Conv2D(16, 3, strides = 1, padding = 'valid', kernel_initializer = glorot_normal, kernel_regularizer = l2(0.00001), name = 'Pnet_conv2')(M)
    M = PReLU(shared_axes= [1, 2], name = 'Pnet_prelu2')(M)
    
    M = Conv2D(32, 3, strides = 1, padding = 'valid', kernel_initializer = glorot_normal, kernel_regularizer = l2(0.00001), name = 'Pnet_conv3')(M)
    M = PReLU(shared_axes= [1, 2], name = 'Pnet_prelu3')(M)
    
    Classifier_conv = Conv2D(1, 1, activation = 'sigmoid', name = 'Pnet_classifier_conv', kernel_initializer = glorot_normal)(M)
    Bbox_regressor_conv = Conv2D(4, 1, name = 'Pnet_bbox_regressor_conv', kernel_initializer = glorot_normal)(M)
    Landmark_regressor_conv = Conv2D(12, 1, name = 'Pnet_landmark_regressor_conv', kernel_initializer = glorot_normal)(M)
    
    Classifier = Reshape((1, ), name = 'Pnet_classifier')(Classifier_conv)
    Bbox_regressor = Reshape((4, ), name = 'Pnet_bbox_regressor')(Bbox_regressor_conv) 
    if train_with_landmark: 
        Landmark_regressor = Reshape((12, ), name = 'Pnet_landmark_regressor')(Landmark_regressor_conv)
        Pnet_output = Concatenate()([Classifier, Bbox_regressor, Landmark_regressor]) 
        model = Model(X, Pnet_output) 
    else:
        Pnet_output = Concatenate()([Classifier, Bbox_regressor])
        model = Model(X, Pnet_output)
    
    return model

def pnet_train2(train_with_landmark = False):
    
    X = Input(shape = (12, 12, 3), name = 'Pnet_input')
    
    M = Conv2D(10, 3, strides = 1, padding = 'valid', use_bias = False, kernel_initializer = glorot_normal, kernel_regularizer = l2(0.00001), name = 'Pnet_conv1')(X)
    M = BatchNormalization(axis = -1, name = 'Pnet_bn1')(M)
    M = PReLU(shared_axes = [1, 2], name = 'Pnet_prelu1')(M)
    M = MaxPooling2D(pool_size = 2, name = 'Pnet_maxpool1')(M) # default 'pool_size' is 2!!! 
    
    M = Conv2D(16, 3, strides = 1, padding = 'valid', use_bias = False, kernel_initializer = glorot_normal, kernel_regularizer = l2(0.00001), name = 'Pnet_conv2')(M)
    M = BatchNormalization(axis = -1, name = 'Pnet_bn2')(M)
    M = PReLU(shared_axes= [1, 2], name = 'Pnet_prelu2')(M)
    
    M = Conv2D(32, 3, strides = 1, padding = 'valid', use_bias = False, kernel_initializer = glorot_normal, kernel_regularizer = l2(0.00001), name = 'Pnet_conv3')(M)
    M = BatchNormalization(axis = -1, name = 'Pnet_bn3')(M)
    M = PReLU(shared_axes= [1, 2], name = 'Pnet_prelu3')(M)
    
    Classifier_conv = Conv2D(1, 1, activation = 'sigmoid', name = 'Pnet_classifier_conv', kernel_initializer = glorot_normal)(M)
    Bbox_regressor_conv = Conv2D(4, 1, name = 'Pnet_bbox_regressor_conv', kernel_initializer = glorot_normal)(M)
    Landmark_regressor_conv = Conv2D(12, 1, name = 'Pnet_landmark_regressor_conv', kernel_initializer = glorot_normal)(M)
    
    Classifier = Reshape((1, ), name = 'Pnet_classifier')(Classifier_conv)
    Bbox_regressor = Reshape((4, ), name = 'Pnet_bbox_regressor')(Bbox_regressor_conv) 
    if train_with_landmark: 
        Landmark_regressor = Reshape((12, ), name = 'Pnet_landmark_regressor')(Landmark_regressor_conv)
        Pnet_output = Concatenate()([Classifier, Bbox_regressor, Landmark_regressor]) 
        model = Model(X, Pnet_output) 
    else:
        Pnet_output = Concatenate()([Classifier, Bbox_regressor])
        model = Model(X, Pnet_output)
    
    return model

def pnet_train3(train_with_landmark = False):
    
    X = Input(shape = (12, 12, 3), name = 'Pnet_input')
    
    M = Conv2D(10, 3, strides = 1, padding = 'valid', kernel_initializer = glorot_normal, kernel_regularizer = l2(0.00001), name = 'Pnet_conv1')(X)
    M = PReLU(shared_axes = [1, 2], name = 'Pnet_prelu1')(M)
    M = MaxPooling2D(pool_size = 2, name = 'Pnet_maxpool1')(M) 
    
    M = Conv2D(16, 3, strides = 1, padding = 'valid', kernel_initializer = glorot_normal, kernel_regularizer = l2(0.00001), name = 'Pnet_conv2')(M)
    M = PReLU(shared_axes= [1, 2], name = 'Pnet_prelu2')(M)
    
    M = Conv2D(32, 3, strides = 1, padding = 'valid', kernel_initializer = glorot_normal, kernel_regularizer = l2(0.00001), name = 'Pnet_conv3')(M)
    M = PReLU(shared_axes= [1, 2], name = 'Pnet_prelu3')(M)
    
    Classifier_conv = Conv2D(2, 1, activation = 'softmax', name = 'Pnet_classifier_conv', kernel_initializer = glorot_normal)(M)
    Bbox_regressor_conv = Conv2D(4, 1, name = 'Pnet_bbox_regressor_conv', kernel_initializer = glorot_normal)(M)
    
    Classifier = Reshape((2, ), name = 'Pnet_classifier')(Classifier_conv)
    Bbox_regressor = Reshape((4, ), name = 'Pnet_bbox_regressor')(Bbox_regressor_conv)
    
    if train_with_landmark:
        Landmark_regressor_conv = Conv2D(12, 1, name = 'Pnet_landmark_regressor_conv', kernel_initializer = glorot_normal)(M)
        Landmark_regressor = Reshape((12, ), name = 'Pnet_landmark_regressor')(Landmark_regressor_conv)
        Pnet_output = Concatenate()([Classifier, Bbox_regressor, Landmark_regressor])
    else:
        Pnet_output = Concatenate()([Classifier, Bbox_regressor])
    
    model = Model(X, Pnet_output)
    
    return model

def pnet():
    
    X = Input(shape = (None, None, 3), name = 'Pnet_input')
    
    M = Conv2D(10, 3, strides = 1, padding = 'valid', name = 'Pnet_conv1')(X)
    M = PReLU(shared_axes = [1, 2], name = 'Pnet_prelu1')(M)
    M = MaxPooling2D(pool_size = 2, name = 'Pnet_maxpool1')(M) # default 'pool_size' is 2!!! 
    
    M = Conv2D(16, 3, strides = 1, padding = 'valid', name = 'Pnet_conv2')(M)
    M = PReLU(shared_axes= [1, 2], name = 'Pnet_prelu2')(M)
    
    M = Conv2D(32, 3, strides = 1, padding = 'valid', name = 'Pnet_conv3')(M)
    M = PReLU(shared_axes= [1, 2], name = 'Pnet_prelu3')(M)
    
    Classifier_conv = Conv2D(1, 1, activation = 'sigmoid', name = 'Pnet_classifier_conv')(M)
    Bbox_regressor_conv = Conv2D(4, 1, name = 'Pnet_bbox_regressor_conv')(M)
    
    model = Model(X, [Classifier_conv, Bbox_regressor_conv])
    
    return model

'''R-Net'''
def rnet(training = False, train_with_landmark = False):
    
    X = Input(shape = (24, 24, 3), name = 'Rnet_input')
    
    M = Conv2D(28, 3, strides = 1, padding = 'valid', name = 'Rnet_conv1')(X)
    M = PReLU(shared_axes=[1, 2], name = 'Rnet_prelu1')(M)
    M = MaxPooling2D(pool_size = 3, strides = 2, padding = 'same', name = 'Rnet_maxpool1')(M)
    
    M = Conv2D(48, 3, strides = 1, padding = 'valid', name = 'Rnet_conv2')(M)
    M = PReLU(shared_axes = [1, 2], name = 'Rnet_prelu2')(M)
    M = MaxPooling2D(pool_size = 3, strides = 2, name = 'Rnet_maxpool2')(M)
    
    M = Conv2D(64, 2, strides = 1, padding = 'valid', name = 'Rnet_conv3')(M)
    M = PReLU(shared_axes = [1, 2], name = 'Rnet_prelu3')(M)
    
    M = Flatten(name = 'Rnet_flatten')(M)
    M = Dense(128, name = 'Rnet_fc')(M)
    M = PReLU(name = 'Rnet_prelu4')(M)
    
    Classifier = Dense(1, activation = 'sigmoid', name = 'Rnet_classifier')(M)
    Bbox_regressor = Dense(4, name = 'Rnet_bbox_regressor')(M)
    
    if training and train_with_landmark: 
        Landmark_regressor = Dense(12, name = 'Rnet_landmark_regressor')(M)
        Rnet_output = Concatenate()([Classifier, Bbox_regressor, Landmark_regressor]) 
        model = Model(X, Rnet_output) 
    elif training and not train_with_landmark: 
        Rnet_output = Concatenate()([Classifier, Bbox_regressor, Landmark_regressor]) 
        model = Model(X, Rnet_output)
    else:
        model = Model(X, [Classifier, Bbox_regressor])
    
    return model

'''O-Net'''
def onet(training = False):
    
    X = Input(shape = (48, 48, 3), name = 'Onet_input')

    M = Conv2D(32, 3, strides = 1, padding = 'valid', name = 'Onet_conv1')(X)
    M = PReLU(shared_axes = [1, 2], name = 'Onet_prelu1')(M)
    M = MaxPooling2D(pool_size = 3, strides = 2, padding = 'same', name = 'Onet_maxpool1')(M)
        
    M = Conv2D(64, 3, strides = 1, padding = 'valid', name = 'Onet_conv2')(M)
    M = PReLU(shared_axes = [1, 2], name = 'Onet_prelu2')(M)
    M = MaxPooling2D(pool_size = 3, strides = 2, padding = 'valid', name = 'Onet_maxpool2')(M)
        
    M = Conv2D(64, 3, strides = 1, padding = 'valid', name = 'Onet_conv3')(M)
    M = PReLU(shared_axes = [1,2], name = 'Onet_prelu3')(M)
    M = MaxPooling2D(pool_size = 2, padding = 'valid', name = 'Onet_maxpool3')(M)
    
    M = Conv2D(128, 2, strides = 1, padding = 'valid', name = 'Onet_conv4')(M)
    M = PReLU(shared_axes = [1, 2], name='Onet_prelu4')(M)
    
    M = Flatten(name = 'Onet_flatten')(M)
    M = Dense(256, name = 'Onet_fc') (M)
    M = PReLU(name = 'Onet_prelu5')(M)
    
    Classifier = Dense(2, activation = 'softmax', name='Onet_classifier')(M)
    Bbox_regressor = Dense(4, name = 'Onet_bbox_regressor')(M)
    Landmark_regressor = Dense(12, name = 'Onet_landmark_regressor')(M)
    
    if training:
        Onet_output = Concatenate()([Classifier, Bbox_regressor, Landmark_regressor])
        
        model = Model(X, Onet_output)
    else:
        model = Model(X, [Classifier, Bbox_regressor, Landmark_regressor])
    
    return model

'''
PNet = pnet(False) 
PNet.summary()

PNet = pnet(True) 
PNet.summary()

RNet = rnet(False) 
RNet.summary()

RNet = rnet(True) 
RNet.summary()

ONet = onet(False) 
ONet.summary()

ONet = onet(True) 
ONet.summary()
'''
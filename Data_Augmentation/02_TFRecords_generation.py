# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 00:41:26 2019

@author: TMaysGGS
"""

'''Last updated on 2020.04.03 15:45'''
'''Importing the libraries & setting configurations'''
import os
import random
import sys
import argparse
import pickle as pkl
import numpy as np
import tensorflow as tf

'''Building helper functions'''
def _bytes_feature(value):
    
#    if not isinstance(value, list):
#        value = [value]
#    return tf.train.Feature(bytes_list = tf.train.BytesList(value = value))
    '''Returns a bytes_list from a string / byte. '''
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList will not unpack a string from an EagerTensor. 
    
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def _float_feature(value):
    
    '''Returns a float_list from a float / double. '''
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list = tf.train.FloatList(value = value))

def _int64_feature(value):
    
    '''Returns an int64_list from bool / enum / int / uint. '''
    if not isinstance(value, list):
        value = [value]
    
    return tf.train.Feature(int64_list = tf.train.Int64List(value = value))

def convert_image_info_to_tfexample(img_dir, anno):
    
    img_path = os.path.join(img_dir, anno[0])
    label = int(anno[1])
    roi = anno[2]
    if not isinstance(roi, list):
        roi = list(roi)
    landmark = anno[3]
    if not isinstance(landmark, list):
        landmark = list(landmark)
    
    img_string = open(img_path, 'rb').read()
    img_shape = tf.image.decode_jpeg(img_string).shape
    
    feature = {
            'height': _int64_feature(img_shape[0]),
            'width': _int64_feature(img_shape[1]),
            'depth': _int64_feature(img_shape[2]),
            'label': _int64_feature(label),
            'roi': _float_feature(roi),
            'landmark': _float_feature(landmark),
            'image_raw': _bytes_feature(img_string),
            }
    
    return tf.train.Example(features = tf.train.Features(feature = feature))

def convert_image_info_to_tfexample_for_train(img_dir, anno):
    
    img_path = os.path.join(img_dir, anno[0])
    label = int(anno[1])
    roi = anno[2]
    if not isinstance(roi, list):
        roi = list(roi)
    landmark = anno[3]
    if not isinstance(landmark, list):
        landmark = list(landmark)
    img_info = [label]
    img_info.extend(roi)
    img_info.extend(landmark)
    
    img_string = open(img_path, 'rb').read()
    img_shape = tf.image.decode_jpeg(img_string).shape
    
    feature = {
            'height': _int64_feature(img_shape[0]),
            'width': _int64_feature(img_shape[1]),
            'depth': _int64_feature(img_shape[2]),
            'info': _float_feature(img_info),
            'image_raw': _bytes_feature(img_string),
            }
    
    return tf.train.Example(features = tf.train.Features(feature = feature))

'''TFRecords Generation'''
def main(args):
    
    IMG_SIZE = args.IMG_SIZE
    DEBUG = args.DEBUG
    DATA_PATH = r'../Data/' + str(IMG_SIZE)
    print("Image size: " + str(IMG_SIZE))
    
    # Import the data
    RECORD_NAME_LIST = ['pos_record.pkl', 'neg_record.pkl', 'part_record.pkl','landmark_record.pkl']
    TFRECORD_NAME_PREFIX_LIST = ['pos_data_', 'neg_data_', 'part_data_', 'landmark_data_']
    
    # Make TFRecords
    for i in range(len(RECORD_NAME_LIST)):
        
        print("Start to make " + TFRECORD_NAME_PREFIX_LIST[i][: -1])
        RECORD_PATH = os.path.join(DATA_PATH, RECORD_NAME_LIST[i])
        f= open(RECORD_PATH, 'rb') 
        info = pkl.load(f) 
        f.close()
        random.shuffle(info)
        
        if DEBUG:
            temp = info
            info = temp[: 16384]
            del temp
        print("There are " + str(len(info)) + " images in total. \n")
        
        num_of_tfrecords = int(np.ceil(len(info) / 500000))
        for j in range(num_of_tfrecords):
            TFRECORD_PATH = os.path.join(DATA_PATH, TFRECORD_NAME_PREFIX_LIST[i] + str(j) + '.tfrecord')
            batch_info = info[500000 * j: min(500000 * (j + 1), len(info))]
            # writer = tf.python_io.TFRecordWriter(TFRECORD_PATH)
            with tf.io.TFRecordWriter(TFRECORD_PATH) as writer:
                for k in range(len(batch_info)):
                    example = convert_image_info_to_tfexample_for_train(DATA_PATH, batch_info[k])
                    writer.write(example.SerializeToString())
            # writer.close()
        print(TFRECORD_NAME_PREFIX_LIST[i][: -1] + " generated. ")
    
def parse_arguments(argv):
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--IMG_SIZE", type = int, help = "The size of generated images")
    parser.add_argument("--DEBUG", action = 'store_true', help = "Whether in debug mode")
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    
    main(parse_arguments(sys.argv[1: ]))

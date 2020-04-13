# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 10:42:59 2020

@author: geniu
"""

'''Last updated on 2020.04.09 15:42'''
'''Importing the libraries & setting the configurations'''
import os
import sys
import argparse
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.optimizers import SGD 
from tensorflow.python.keras.optimizer_v2.adam import Adam

sys.path.append(r'../')
from MTCNN_models import pnet_train1, rnet, onet

# os.environ['CUDA_VISIBLE_DEVICES'] = ''

def main(args):
    
    IMG_SIZE = args.IMG_SIZE
    if args.USE_PRETRAINED_MODEL == 0:
        USE_PRETRAINED_MODEL = False
    else:
        USE_PRETRAINED_MODEL = True
    BATCH_SIZE = 1792
    EPOCHS = 100
    DATA_COMPOSE_RATIO = [1 / 7., 3 / 7., 1 / 7., 2 / 7.] # for pos, neg, part-face & landmark data
    SAMPLE_KEEP_RATIO = 0.7
    if args.OPTIMIZER == 'sgd':
        OPTIMIZER = SGD
    else:
        OPTIMIZER = Adam
    if IMG_SIZE == 12:
        print('Training PNet')
        loss_weights = [1., 0.5, 0.5]
        model = pnet_train1(train_with_landmark = not args.TRAIN_WITHOUT_LANDMARK)
        model.summary()
        if USE_PRETRAINED_MODEL:
            model.load_weights('../Models/PNet_train.h5')
    elif IMG_SIZE == 24:
        print('Training RNet')
        loss_weights = [1., 0.5, 0.5]
        model = rnet(training = True, train_with_landmark = not args.TRAIN_WITHOUT_LANDMARK)
        if USE_PRETRAINED_MODEL:
            model.load_weights('../Models/RNet_train.h5')
    elif IMG_SIZE == 48:
        print('Training ONet')
        loss_weights = [1., 0.5, 1.]
        model = onet(training = True, train_with_landmark = not args.TRAIN_WITHOUT_LANDMARK)
        if USE_PRETRAINED_MODEL:
            model.load_weights('../Models/ONet_train.h5')
    else:
        raise Exception("IMG_SIZE must be one of 12, 24 and 48. ")
    
    TFRECORDS_DIR = os.path.join(r'../Data', str(IMG_SIZE))
    POS_TFRECORDS_PATH_LIST = []
    NEG_TFRECORDS_PATH_LIST = []
    PART_TFRECORDS_PATH_LIST = []
    LANDMARK_TFRECORDS_PATH_LIST = []
    for file_name in os.listdir(TFRECORDS_DIR):
        if len(file_name) > 9 and file_name[-9: ] == '.tfrecord':
            if 'pos' in file_name:
                POS_TFRECORDS_PATH_LIST.append(os.path.join(TFRECORDS_DIR, file_name))
            elif 'neg' in file_name:
                NEG_TFRECORDS_PATH_LIST.append(os.path.join(TFRECORDS_DIR, file_name))
            elif 'part' in file_name:
                PART_TFRECORDS_PATH_LIST.append(os.path.join(TFRECORDS_DIR, file_name))
            elif 'landmark' in file_name:
                LANDMARK_TFRECORDS_PATH_LIST.append(os.path.join(TFRECORDS_DIR, file_name))
    
    raw_pos_dataset = tf.data.TFRecordDataset(POS_TFRECORDS_PATH_LIST)
    raw_neg_dataset = tf.data.TFRecordDataset(NEG_TFRECORDS_PATH_LIST)
    raw_part_dataset = tf.data.TFRecordDataset(PART_TFRECORDS_PATH_LIST)
    raw_landmark_dataset = tf.data.TFRecordDataset(LANDMARK_TFRECORDS_PATH_LIST)
    
    image_feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'info': tf.io.FixedLenFeature([17], tf.float32),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        }
    
    def _read_tfrecord(serialized_example):
        
        example = tf.io.parse_single_example(serialized_example, image_feature_description)
        
        img = tf.image.decode_jpeg(example['image_raw'], channels = 3) # RGB rather than BGR!!! 
        img = (tf.cast(img, tf.float32) - 127.5) / 128.
        img_shape = [example['height'], example['width'], example['depth']]
        img = tf.reshape(img, img_shape)
        
        info = example['info']
        
        return img, info
    
    parsed_pos_dataset = raw_pos_dataset.map(_read_tfrecord)
    parsed_neg_dataset = raw_neg_dataset.map(_read_tfrecord)
    parsed_part_dataset = raw_part_dataset.map(_read_tfrecord)
    parsed_landmark_dataset = raw_landmark_dataset.map(_read_tfrecord)
    
    ds = tf.data.experimental.sample_from_datasets([parsed_pos_dataset.repeat(), 
                                                    parsed_neg_dataset.repeat(), 
                                                    parsed_part_dataset.repeat(), 
                                                    parsed_landmark_dataset.repeat()], DATA_COMPOSE_RATIO)
    
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(28672)
    
    '''Building custom loss/cost function'''
    def custom_loss(y_true, y_pred, loss_weights = loss_weights): # Verified
        
        zero_index = K.zeros_like(y_true[:, 0]) 
        ones_index = K.ones_like(y_true[:, 0]) 
        
        # Classifier
        labels = y_true[:, 0] 
        class_preds = y_pred[:, 0] 
        bi_crossentropy_loss = -labels * K.log(class_preds) - (1 - labels) * K.log(1 - class_preds) 
        
        classify_valid_index = tf.where(K.less(y_true[:, 0], 0), zero_index, ones_index) 
        classify_keep_num = K.cast(tf.cast(tf.reduce_sum(classify_valid_index), tf.float32) * SAMPLE_KEEP_RATIO, dtype = tf.int32) 
        # For classification problem, only pick 70% of the valid samples. 
        
        classify_loss_sum = bi_crossentropy_loss * tf.cast(classify_valid_index, bi_crossentropy_loss.dtype) 
        classify_loss_sum_filtered, _ = tf.nn.top_k(classify_loss_sum, k = classify_keep_num) 
        classify_loss = tf.where(K.equal(classify_keep_num, 0), tf.constant(0, dtype = tf.float32), K.mean(classify_loss_sum_filtered)) 
        
        # Bounding box regressor
        rois = y_true[:, 1: 5] 
        roi_preds = y_pred[:, 1: 5] 
        roi_raw_mean_square_error = K.sum(K.square(rois - roi_preds), axis = 1) # mse
        # roi_raw_smooth_l1_loss = K.mean(tf.where(K.abs(rois - roi_preds) < 1, 0.5 * K.square(rois - roi_preds), K.abs(rois - roi_preds) - 0.5)) # L1 Smooth Loss 
        
        roi_valid_index = tf.where(K.equal(K.abs(y_true[:, 0]), 1), ones_index, zero_index) 
        roi_keep_num = K.cast(tf.reduce_sum(roi_valid_index), dtype = tf.int32) 
        
        roi_valid_mean_square_error = roi_raw_mean_square_error * tf.cast(roi_valid_index, roi_raw_mean_square_error.dtype)
        roi_filtered_mean_square_error, _ = tf.nn.top_k(roi_valid_mean_square_error, k = roi_keep_num) 
        roi_loss = tf.where(K.equal(roi_keep_num, 0), tf.constant(0, dtype = tf.float32), K.mean(roi_filtered_mean_square_error)) 
        # roi_valid_smooth_l1_loss = roi_raw_smooth_l1_loss * roi_valid_index
        # roi_filtered_smooth_l1_loss, _ = tf.nn.top_k(roi_valid_smooth_l1_loss, k = roi_keep_num) 
        # roi_loss = K.mean(roi_filtered_smooth_l1_loss) 
        
        # Landmark regressor
        pts = y_true[:, 5: 17] 
        pt_preds = y_pred[:, 5: 17] 
        pts_raw_mean_square_error  = K.sum(K.square(pts - pt_preds), axis = 1) # mse 
        # pts_raw_smooth_l1_loss = K.mean(tf.where(K.abs(pts - pt_preds) < 1, 0.5 * K.square(pts - pt_preds), K.abs(pts - pt_preds) - 0.5)) # L1 Smooth Loss 
        
        pts_valid_index = tf.where(K.equal(y_true[:, 0], -2), ones_index, zero_index) 
        pts_keep_num = K.cast(tf.reduce_sum(pts_valid_index), dtype = tf.int32) 
        
        pts_valid_mean_square_error = pts_raw_mean_square_error * tf.cast(pts_valid_index, tf.float32) 
        pts_filtered_mean_square_error, _ = tf.nn.top_k(pts_valid_mean_square_error, k = pts_keep_num) 
        pts_loss = tf.where(K.equal(pts_keep_num, 0), tf.constant(0, dtype = tf.float32), K.mean(pts_filtered_mean_square_error)) 
        # pts_valid_smooth_l1_loss = pts_raw_smooth_l1_loss * pts_valid_index
        # pts_filtered_smooth_l1_loss, _ = tf.nn.top_k(pts_valid_smooth_l1_loss, k = pts_keep_num) 
        # pts_loss = K.mean(pts_filtered_smooth_l1_loss)
        
        loss = classify_loss * loss_weights[0] + roi_loss * loss_weights[1] + pts_loss * loss_weights[2]
        
        return loss 
        
    '''Training'''
    lr = args.LEARNING_RATE
    model.compile(optimizer = OPTIMIZER(lr = lr), loss = custom_loss)
    model.fit(ds, steps_per_epoch = 1636, epochs = EPOCHS)
    # model.fit(ds, steps_per_epoch = 1636, epochs = EPOCHS, validation_data = ds, validation_steps = 1636)
    model.save(r'../Models/PNet_trained_without_lm.h5')

def parse_arguments(argv):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--IMG_SIZE', type = int, help = 'The input image size', default = 0)
    parser.add_argument('--USE_PRETRAINED_MODEL', type = int, help = 'Use pretrained model or not', default = 0)
    parser.add_argument('--TRAIN_WITHOUT_LANDMARK', action = 'store_false', help = 'Train model with landmark regression or not')
    parser.add_argument('--OPTIMIZER', type = str, help = 'The training optimizer', default = 'adam')
    parser.add_argument('--LEARNING_RATE', type = float, help = 'The initial learning rate', default = 0.001)
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    
    main(parse_arguments(sys.argv[1:]))
    
'''Test
y_true = tf.placeholder(dtype = tf.float32, shape = (None, 17), name = 'y_true')
y_pred = tf.placeholder(dtype = tf.float32, shape = (None, 17), name = 'y_pred')

Y_head = np.random.rand(32, 17) * 2 - 1
for i in range(len(Y_head)):
    Y_head[i][0] = np.squeeze(np.random.rand(1))

Y = np.random.rand(32, 17) * 2 - 1
for i in range(len(Y)):
    Y[i][0] = np.random.choice([0, 1, -1, -2])

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
sess.run(loss, feed_dict = {y_true: Y, y_pred: Y_head})
sess.close()
'''
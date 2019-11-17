# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 11:24:51 2019

@author: TMaysGGS
"""

'''Last updated on 10/14/2019 16:07'''
'''Importing the libraries & setting the configurations'''
import os
import sys
import argparse
import random
import cv2 
import skimage 
import math 
import threading 
import pickle as pkl
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import SGD, Adam 

sys.path.append('../')
from MTCNN_models import pnet, rnet, onet

# os.environ['CUDA_VISIBLE_DEVICES']=''

'''Building helper functions & classes''' 
class CustomThread(threading.Thread):
    def __init__(self, func, args=()):
        super(CustomThread, self).__init__()
        self.func = func
        self.args = args
    def run(self):
        self.ims, self.gts = self.func(*self.args)
    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.ims, self.gts
        except Exception:
            return None
        
def image_random_distort(image): 
    
    B, G, R = cv2.split(image) 
    scale_b = 1.0 + 0.001 * np.random.randint(-200, 201) 
    scale_g = 1.0 + 0.001 * np.random.randint(-200, 201) 
    scale_r = 1.0 + 0.001 * np.random.randint(-200, 201) 

    B = B * scale_b 
    G = G * scale_g 
    R = R * scale_r 
    distorted = cv2.merge([B, G, R]) 
    distorted = np.clip(distorted, 0.0, 255.0) 
    
    if random.choice([0,1]) == 1: 
        distorted = skimage.util.random_noise(distorted, mode = 'gaussian')
    
    return distorted 

def minibatch_data_processing(batch_list, AUG_IMG_PATH, TRAIN_WITH_LANDMARK): 
    
    X_batch_list = []
    Y_batch_list = []
    for k in range(len(batch_list)):
        img_name = batch_list[k][0]
        # print(os.path.join(AUG_IMG_PATH, img_name))
        img = cv2.imread(os.path.join(AUG_IMG_PATH, img_name)) 
        distorted = image_random_distort(img) 
        X_batch_list.append(distorted / 255.)
        label = batch_list[k][1]
        roi = batch_list[k][2] 
        if TRAIN_WITH_LANDMARK: 
            landmark = batch_list[k][3]
            Y_batch_list.append([label, roi[0], roi[1], roi[2], roi[3], 
                                 landmark[0], landmark[1], landmark[2], landmark[3], 
                                 landmark[4], landmark[5], landmark[6], landmark[7], 
                                 landmark[8], landmark[9], landmark[10], landmark[11]]) 
        else: 
            Y_batch_list.append([label, roi[0], roi[1], roi[2], roi[3]]) 
    
    return X_batch_list, Y_batch_list 

'''Reading the data & training the model''' 
def main(args):
    
    IMG_SIZE = args.IMG_SIZE 
    
    if args.USE_PRETRAINED_MODEL == 0: 
        USE_PRETRAINED_MODEL = False 
    else: 
        USE_PRETRAINED_MODEL = True 
    
    if args.TRAIN_WITH_LANDMARK == 0: 
        TRAIN_WITH_LANDMARK = False 
        BATCH_SIZE = 1280 
    else:
        TRAIN_WITH_LANDMARK = True 
        BATCH_SIZE = 1792 
    
    if args.TRAIN_WITH_HARD == 1: 
        TRAIN_WITH_HARD = True 
    
    THREAD_NUM = 4 
    EPOCHS = 100 
    copy_num = 7 if TRAIN_WITH_LANDMARK else 5 
    if IMG_SIZE == 12:
        MODEL = 'PNet'
        loss_weights = [1., 0.5, 0.5]
        model = pnet(training = True, train_with_landmark = TRAIN_WITH_LANDMARK)
        if USE_PRETRAINED_MODEL and TRAIN_WITH_LANDMARK:
            model.load_weights('../Models/PNet_train_with_lm.h5') 
        elif USE_PRETRAINED_MODEL and not TRAIN_WITH_LANDMARK: 
            model.load_weights('../Models/PNet_train_without_lm.h5') 
    elif IMG_SIZE == 24:
        MODEL = 'RNet'
        loss_weights = [1., 0.5, 0.5]
        model = rnet(training = True, train_with_landmark = TRAIN_WITH_LANDMARK)
        if USE_PRETRAINED_MODEL and TRAIN_WITH_LANDMARK:
            model.load_weights('../Models/RNet_train_with_lm.h5') 
        elif USE_PRETRAINED_MODEL and not TRAIN_WITH_LANDMARK: 
            model.load_weights('../Models/RNet_train_without_lm.h5') 
    elif IMG_SIZE == 48:
        MODEL = 'ONet'
        loss_weights = [1., 0.5, 1.]
        model = onet(training = True, train_with_landmark = TRAIN_WITH_LANDMARK)
        if USE_PRETRAINED_MODEL and TRAIN_WITH_LANDMARK:
            model.load_weights('../Models/ONet_train_with_lm.h5') 
        elif USE_PRETRAINED_MODEL and not TRAIN_WITH_LANDMARK: 
            model.load_weights('../Models/ONet_train_without_lm.h5') 
    else:
        raise Exception("IMG_SIZE must be one of 12, 24 and 48. ")
        
    AUG_IMG_PATH = r'../Data/' + str(IMG_SIZE)
    SAMPLE_KEEP_RATIO = 0.7 # Online hard sample mining, used in face/non-face classification task which is adaptive to the training process.
    
    '''Importing the dataset'''
    # Load the records
    POS_RECORD_PATH = r'../Data/' + str(IMG_SIZE) + '/pos_record.pkl'
    PART_RECORD_PATH = r'../Data/' + str(IMG_SIZE) + '/part_record.pkl'
    NEG_RECORD_PATH = r'../Data/' + str(IMG_SIZE) + '/neg_record.pkl' 
    
    f= open(POS_RECORD_PATH, 'rb') 
    pos_info = pkl.load(f) 
    f.close()
    
    f= open(PART_RECORD_PATH, 'rb') 
    part_info = pkl.load(f) 
    f.close()
    
    f= open(NEG_RECORD_PATH, 'rb') 
    neg_info = pkl.load(f) 
    f.close()
    
    if TRAIN_WITH_LANDMARK: 
        LDMK_RECORD_PATH = r'../Data/' + str(IMG_SIZE) + '/landmark_record.pkl' 
        f= open(LDMK_RECORD_PATH, 'rb') 
        ldmk_info = pkl.load(f) 
        f.close()
    
    if TRAIN_WITH_HARD: 
        NEG_HARD_RECORD_PATH = r'../Data/' + str(IMG_SIZE) + '/neg_hard_record.pkl' 
        f = open(NEG_HARD_RECORD_PATH, 'rb') 
        neg_hard_info = pkl.load(f) 
        f.close() 
    
    if not TRAIN_WITH_LANDMARK and not TRAIN_WITH_HARD: 
        # neg: pos: part-face = 3: 1: 1
        lengths = np.array([len(neg_info) / 3, len(pos_info), len(part_info)])
        # Get the dataset index with the smallest amount rate
        min_index = lengths.argsort()[0]
        if min_index == 0:
            smallest_set = neg_info
            branch_batch_size = int(BATCH_SIZE / copy_num * 3)
        elif min_index == 1:
            smallest_set = pos_info
            branch_batch_size = int(BATCH_SIZE / copy_num) 
        elif min_index == 2: 
            smallest_set = part_info
            branch_batch_size = int(BATCH_SIZE / copy_num) 
        else:
            raise Exception("Getting lengths of datasets error") 
    
    elif not TRAIN_WITH_LANDMARK and TRAIN_WITH_HARD: 
        # neg: neg_hard: pos: part-face = 2: 1: 1: 1
        lengths = np.array([len(neg_info) / 2, len(neg_hard_info), len(pos_info), len(part_info)])
        # Get the dataset index with the smallest amount rate
        min_index = lengths.argsort()[0]
        if min_index == 0:
            smallest_set = neg_info
            branch_batch_size = int(BATCH_SIZE / copy_num * 2)
        elif min_index == 1:
            smallest_set = neg_hard_info
            branch_batch_size = int(BATCH_SIZE / copy_num) 
        elif min_index == 2:
            smallest_set = pos_info
            branch_batch_size = int(BATCH_SIZE / copy_num) 
        elif min_index == 3: 
            smallest_set = part_info
            branch_batch_size = int(BATCH_SIZE / copy_num) 
        else:
            raise Exception("Getting lengths of datasets error") 
            
    elif TRAIN_WITH_LANDMARK and not TRAIN_WITH_HARD: 
        # neg: pos: part-face: landmark = 3: 1: 1: 2
        lengths = np.array([len(neg_info) / 3, len(pos_info), len(part_info), len(ldmk_info) / 2])
        # Get the dataset index with the smallest amount rate
        min_index = lengths.argsort()[0]
        if min_index == 0:
            smallest_set = neg_info
            branch_batch_size = int(BATCH_SIZE / copy_num * 3)
        elif min_index == 1:
            smallest_set = pos_info
            branch_batch_size = int(BATCH_SIZE / copy_num)
        elif min_index == 2: 
            smallest_set = part_info
            branch_batch_size = int(BATCH_SIZE / copy_num)
        elif min_index == 3:
            smallest_set = ldmk_info
            branch_batch_size = int(BATCH_SIZE / copy_num * 2)
        else:
            raise Exception("Getting lengths of datasets error") 
        
    else: 
        # neg: neg_hard: pos: part-face: landmark = 2: 1: 1: 1: 2
        lengths = np.array([len(neg_info) / 2, len(neg_hard_info), len(pos_info), len(part_info), len(ldmk_info) / 2])
        # Get the dataset index with the smallest amount rate
        min_index = lengths.argsort()[0]
        if min_index == 0:
            smallest_set = neg_info
            branch_batch_size = int(BATCH_SIZE / copy_num * 2) 
        elif min_index == 1:
            smallest_set = neg_hard_info 
            branch_batch_size = int(BATCH_SIZE / copy_num)
        elif min_index == 2:
            smallest_set = pos_info
            branch_batch_size = int(BATCH_SIZE / copy_num)
        elif min_index == 3: 
            smallest_set = part_info
            branch_batch_size = int(BATCH_SIZE / copy_num)
        elif min_index == 4:
            smallest_set = ldmk_info
            branch_batch_size = int(BATCH_SIZE / copy_num * 2)
        else:
            raise Exception("Getting lengths of datasets error") 
    
    '''Building custom loss/cost function''' 
    if TRAIN_WITH_LANDMARK: 
        
        def custom_loss(y_true, y_pred, loss_weights = loss_weights): # Verified
            
            zero_index = K.zeros_like(y_true[:, 0])
            ones_index = K.ones_like(y_true[:, 0]) 
            
            # Classifier
            labels = y_true[:, 0] 
            class_preds = y_pred[:, 0] 
            bi_crossentropy_loss = -labels * K.log(class_preds) - (1 - labels) * K.log(1 - class_preds)
            
            classify_valid_index = tf.where(K.less(y_true[:, 0], 0), zero_index, ones_index) 
            classify_keep_num = K.cast(tf.reduce_sum(classify_valid_index) * SAMPLE_KEEP_RATIO, dtype = tf.int32) 
            # For classification problem, only pick 70% of the valid samples. 
            
            classify_loss_sum = bi_crossentropy_loss * classify_valid_index 
            classify_loss_sum_filtered, _ = tf.nn.top_k(classify_loss_sum, k = classify_keep_num) 
            classify_loss = K.mean(classify_loss_sum_filtered) 
            
            # Bounding box regressor
            rois = y_true[:, 1: 5] 
            roi_preds = y_pred[:, 1: 5] 
            # roi_raw_mean_square_error = K.sum(K.square(rois - roi_preds), axis = 1) # mse
            roi_raw_smooth_l1_loss = K.mean(tf.where(K.abs(rois - roi_preds) < 1, 0.5 * K.square(rois - roi_preds), K.abs(rois - roi_preds) - 0.5)) # L1 Smooth Loss 
            
            roi_valid_index = tf.where(K.equal(K.abs(y_true[:, 0]), 1), ones_index, zero_index) 
            roi_keep_num = K.cast(tf.reduce_sum(roi_valid_index), dtype = tf.int32) 
            
            # roi_valid_mean_square_error = roi_raw_mean_square_error * roi_valid_index 
            # roi_filtered_mean_square_error, _ = tf.nn.top_k(roi_valid_mean_square_error, k = roi_keep_num) 
            # roi_loss = K.mean(roi_filtered_mean_square_error) 
            roi_valid_smooth_l1_loss = roi_raw_smooth_l1_loss * roi_valid_index
            roi_filtered_smooth_l1_loss, _ = tf.nn.top_k(roi_valid_smooth_l1_loss, k = roi_keep_num) 
            roi_loss = K.mean(roi_filtered_smooth_l1_loss) 
            
            # Landmark regressor
            pts = y_true[:, 5: 17] 
            pt_preds = y_pred[:, 5: 17] 
            # pts_raw_mean_square_error  = K.sum(K.square(pts - pt_preds), axis = 1) # mse 
            pts_raw_smooth_l1_loss = K.mean(tf.where(K.abs(pts - pt_preds) < 1, 0.5 * K.square(pts - pt_preds), K.abs(pts - pt_preds) - 0.5)) # L1 Smooth Loss 
            
            pts_valid_index = tf.where(K.equal(y_true[:, 0], -2), ones_index, zero_index) 
            pts_keep_num = K.cast(tf.reduce_sum(pts_valid_index), dtype = tf.int32) 
            
            # pts_valid_mean_square_error = pts_raw_mean_square_error * pts_valid_index
            # pts_filtered_mean_square_error, _ = tf.nn.top_k(pts_valid_mean_square_error, k = pts_keep_num)
            # pts_loss = K.mean(pts_filtered_mean_square_error)
            pts_valid_smooth_l1_loss = pts_raw_smooth_l1_loss * pts_valid_index
            pts_filtered_smooth_l1_loss, _ = tf.nn.top_k(pts_valid_smooth_l1_loss, k = pts_keep_num) 
            pts_loss = K.mean(pts_filtered_smooth_l1_loss)
            
            loss = classify_loss * loss_weights[0] + roi_loss * loss_weights[1] + pts_loss * loss_weights[2]
            
            return loss 
    
    else: 
        
        def custom_loss(y_true, y_pred, loss_weights = loss_weights): # Verified
            
            zero_index = K.zeros_like(y_true[:, 0])
            ones_index = K.ones_like(y_true[:, 0]) 
            
            # Classifier
            labels = y_true[:, 0] 
            class_preds = y_pred[:, 0] 
            bi_crossentropy_loss = -labels * K.log(class_preds) - (1 - labels) * K.log(1 - class_preds)
            
            classify_valid_index = tf.where(K.less(y_true[:, 0], 0), zero_index, ones_index) 
            classify_keep_num = K.cast(tf.reduce_sum(classify_valid_index) * SAMPLE_KEEP_RATIO, dtype = tf.int32) 
            # For classification problem, only pick 70% of the valid samples. 
            
            classify_loss_sum = bi_crossentropy_loss * classify_valid_index 
            classify_loss_sum_filtered, _ = tf.nn.top_k(classify_loss_sum, k = classify_keep_num) 
            classify_loss = K.mean(classify_loss_sum_filtered) 
            
            # Bounding box regressor
            rois = y_true[:, 1: 5] 
            roi_preds = y_pred[:, 1: 5] 
            # roi_raw_mean_square_error = K.sum(K.square(rois - roi_preds), axis = 1) # mse
            roi_raw_smooth_l1_loss = K.mean(tf.where(K.abs(rois - roi_preds) < 1, 0.5 * K.square(rois - roi_preds), K.abs(rois - roi_preds) - 0.5)) # L1 Smooth Loss 
            
            roi_valid_index = tf.where(K.equal(K.abs(y_true[:, 0]), 1), ones_index, zero_index) 
            roi_keep_num = K.cast(tf.reduce_sum(roi_valid_index), dtype = tf.int32) 
            
            # roi_valid_mean_square_error = roi_raw_mean_square_error * roi_valid_index 
            # roi_filtered_mean_square_error, _ = tf.nn.top_k(roi_valid_mean_square_error, k = roi_keep_num) 
            # roi_loss = K.mean(roi_filtered_mean_square_error) 
            roi_valid_smooth_l1_loss = roi_raw_smooth_l1_loss * roi_valid_index
            roi_filtered_smooth_l1_loss, _ = tf.nn.top_k(roi_valid_smooth_l1_loss, k = roi_keep_num) 
            roi_loss = K.mean(roi_filtered_smooth_l1_loss) 
            
            loss = classify_loss * loss_weights[0] + roi_loss * loss_weights[1] 
            
            return loss 
        
    '''Training'''    
    # Save the model after every epoch
    # from tensorflow.keras.callbacks import ModelCheckpoint 
    # check_pointer = ModelCheckpoint(filepath = 'trained_PNet.h5', verbose = 1, save_best_only = False)
    
    # Stream each epoch results into a .csv file
    from tensorflow.keras.callbacks import CSVLogger
    csv_logger = CSVLogger('training.csv', separator = ',', append = True)
    # append = True append if file exists (useful for continuing training)
    # append = False overwrite existing file

    num_of_iters = len(smallest_set) // branch_batch_size 
    if USE_PRETRAINED_MODEL == False: 
        lr = 0.001 
        optimizer = Adam(lr = lr) 
    else: 
        lr = 0.0001 
        optimizer = SGD(lr = lr) 
    former_loss = 100000
    loss_patience = 500
    for i in range(EPOCHS):
        
        print("Epoch: " + str(i))

        # Re-shuffle all the datasets each epoch
        random.shuffle(neg_info)
        random.shuffle(part_info)
        random.shuffle(pos_info) 
        if TRAIN_WITH_LANDMARK: 
            random.shuffle(ldmk_info)
        if TRAIN_WITH_HARD: 
            random.shuffle(neg_hard_info) 
        
        for j in range(num_of_iters):
            
            print("Iteration: " + str(j) + " in Epoch: " + str(i)) 
            
            
            pos_batch = pos_info[int(j * (BATCH_SIZE / copy_num)): int((j + 1) * (BATCH_SIZE / copy_num))]
            part_batch = part_info[int(j * (BATCH_SIZE / copy_num)): int((j + 1) * (BATCH_SIZE / copy_num))] 
            if TRAIN_WITH_LANDMARK: 
                ldmk_batch = ldmk_info[int(j * (BATCH_SIZE / copy_num)): int((j + 1) * (BATCH_SIZE / copy_num * 2))]
            if TRAIN_WITH_HARD: 
                neg_batch = neg_info[int(j * (BATCH_SIZE / copy_num * 2)): int((j + 1) * (BATCH_SIZE / copy_num * 2))]
                neg_hard_batch = neg_hard_info[int(j * (BATCH_SIZE / copy_num)): int((j + 1) * (BATCH_SIZE / copy_num))] 
            else:
                neg_batch = neg_info[int(j * (BATCH_SIZE / copy_num * 3)): int((j + 1) * (BATCH_SIZE / copy_num * 3))] 
                
            batch_list = []
            batch_list.extend(neg_batch)
            batch_list.extend(pos_batch)
            batch_list.extend(part_batch) 
            if TRAIN_WITH_LANDMARK: 
                batch_list.extend(ldmk_batch) 
            if TRAIN_WITH_HARD: 
                batch_list.extend(neg_hard_batch) 
            random.shuffle(batch_list)
            
            # Multi-thread data processing 
            num_of_batch_imgs = len(batch_list) 
            thread_num = max(1, THREAD_NUM) 
            num_per_thread = math.ceil(float(num_of_batch_imgs) / thread_num) 
            threads = [] 
            X_batch_list = [] 
            Y_batch_list = [] 
            for t in range(thread_num): 
                start_idx = int(num_per_thread * t) 
                end_idx = int(min(num_per_thread * (t + 1), num_of_batch_imgs)) 
                cur_batch_list = batch_list[start_idx: end_idx] 
                cur_thread = CustomThread(minibatch_data_processing, (cur_batch_list, AUG_IMG_PATH, TRAIN_WITH_LANDMARK)) 
                threads.append(cur_thread) 
            for t in range(thread_num): 
                threads[t].start() 
            for t in range(thread_num): 
                cur_processed_imgs, cur_processed_gts = threads[t].get_result() 
                X_batch_list.extend(cur_processed_imgs) 
                Y_batch_list.extend(cur_processed_gts) 
                
            ''' 
            # Single-thread data processing 
            X_batch_list = []
            Y_batch_list = []
            for k in range(len(batch_list)):
                img_name = batch_list[k][0]
                # print(os.path.join(AUG_IMG_PATH, img_name))
                img = cv2.imread(os.path.join(AUG_IMG_PATH, img_name)) 
                distorted = image_random_distort(img) 
                X_batch_list.append(distorted / 255.)
                label = batch_list[k][1]
                roi = batch_list[k][2]
                landmark = batch_list[k][3]
                Y_batch_list.append([label, roi[0], roi[1], roi[2], roi[3], 
                                     landmark[0], landmark[1], landmark[2], landmark[3], 
                                     landmark[4], landmark[5], landmark[6], landmark[7], 
                                     landmark[8], landmark[9], landmark[10], landmark[11]])
            '''
            
            X_batch = np.array(X_batch_list)
            Y_batch = np.array(Y_batch_list)
            
            # Fit the data into the model for training & adjust the learning rate according to training procedure 
            patience_count = 0 
            model.compile(optimizer = optimizer, loss = custom_loss)
            hist = model.fit(x = X_batch, y = Y_batch, epochs = 1, shuffle = True, callbacks = [csv_logger]) # Consider .train_on_batch
            loss = hist.history['loss'][0]
            if loss >= former_loss:
                patience_count = patience_count + 1
            else:
                patience_count = 0
            former_loss = loss
            if patience_count > loss_patience:
                lr = lr * 0.1
                print("Now the learning rate is changed to " + str(lr))
            if j != 0 and j % 1000 == 0:
                lr = lr * 0.5
                print("Now the learning rate is changed to " + str(lr)) 
            
            if j % 200 == 0: 
                if TRAIN_WITH_LANDMARK: 
                    model_name = MODEL + '_trained_with_lm' 
                else:
                    model_name = MODEL + '_trained_without_lm' 
                model_name = model_name + '_epoch_' + str(i) + '_iter_' + str(j) + '.h5'
                print("Saving model: " + model_name)
                model.save(os.path.join(r'../Models', model_name))

def parse_arguments(argv):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--IMG_SIZE', type = int, help = 'The input image size', default = 0)
    parser.add_argument('--USE_PRETRAINED_MODEL', type = int, help = 'Use pretrained model or not', default = 0)
    parser.add_argument('--TRAIN_WITH_LANDMARK', type = int, help = 'Train with landmark or not', default = 1) 
    parser.add_argument('--TRAIN_WITH_HARD', type = int, help = 'Train with hard sample mining or not', default = 0) 
    
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
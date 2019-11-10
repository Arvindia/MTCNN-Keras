# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 01:54:14 2019

@author: TMaysGGS
"""

'''Last updated on 11/09/2019 14:26'''
import argparse
import sys
import os
import cv2
# import random
import pickle as pkl
import numpy as np

sys.path.append("..")
from utils import IoU, flip, rotate

RECORD_PATH = r'../Data/CelebA/Anno/CelebA_annotations.pkl'

def main(args):
     
    IMG_SIZE = args.IMG_SIZE
    print("Image size: " + str(IMG_SIZE))
    
    save_dir = '../Data/' + str(IMG_SIZE)
    landmark_save_dir = '../Data/' + str(IMG_SIZE) + '/landmark'
    record_save_dir = '../Data/' + str(IMG_SIZE) + '/landmark_record.pkl'
    if not os.path.exists(save_dir): 
        os.mkdir(save_dir)
    if not os.path.exists(landmark_save_dir): 
        os.mkdir(landmark_save_dir)
    
    f= open(RECORD_PATH, 'rb') 
    info = pkl.load(f) 
    f.close()
    
    idx = 0
    aug_list = []
    
    for i in range(len(info)):
        
        pic_info = info[i]
        image_name = pic_info[0]
        roi = pic_info[1]
        landmark = pic_info[2]
        
        print("Processing the image: " + image_name)
        print(str(idx) + " images generated. ")
        
        image = cv2.imread(os.path.join('../Data/CelebA/Img/img_celeba', image_name))
        height, width, channel = image.shape
        
        # Crop out the face area
        x1, y1, x2, y2 = roi
        face = image[y1: y2, x1: x2]
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        # Calculate the relative position of each landmark
        transfered_landmark = np.zeros((6, 2))
        for k in range(6):
            transfered_landmark[k][0] = (landmark[k * 2] - x1) / (x2 - x1) 
            transfered_landmark[k][1] = (landmark[k * 2 + 1] - y1) / (y2 - y1) 
        '''Verify
        temp = face.copy()
        for k in range(6):
            cv2.circle(temp, (int(transfered_landmark[k][0] * (x2 - x1)), int(transfered_landmark[k][1] * (y2 - y1))), 2, (0, 255, 0), 1)
        cv2.imshow('t', temp)
        cv2.waitKey()
        cv2.destroyAllWindows() # Verified
        del temp
        '''
        # Save the resized face image
        resized_face = cv2.resize(face, (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_LINEAR)
        saving_name = str(idx) + '.jpg'
        saving_path = os.path.join(landmark_save_dir, saving_name)
        success = cv2.imwrite(saving_path, resized_face)
        if not success:
            raise Exception("Landmark picture " + str(idx) + " saving failed. ")
        aug_list.append(['landmark/' + saving_name, 
                         -2, 
                         np.array([-1, -1, -1, -1]), 
                         np.squeeze(np.array(transfered_landmark.reshape(1, -1)))])
        idx = idx + 1
        
        # Mirror
        flipped_face, flipped_landmark = flip(face, transfered_landmark)
        '''Verify
        temp = flipped_face.copy()
        for k in range(6):
            cv2.circle(temp, (int(flipped_landmark[k][0] * (x2 - x1)), int(flipped_landmark[k][1] * (y2 - y1))), 2, (0, 255, 0), 1)
        cv2.imshow('t', temp)
        cv2.waitKey()
        cv2.destroyAllWindows() # Verified
        del temp
        '''
        # Save the resized face image
        resized_flipped_face = cv2.resize(flipped_face, (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_LINEAR)
        saving_name = str(idx) + '.jpg'
        saving_path = os.path.join(landmark_save_dir, saving_name)
        success = cv2.imwrite(saving_path, resized_flipped_face)
        if not success:
            raise Exception("Landmark picture " + str(idx) + " saving failed. ")
        aug_list.append(['landmark/' + saving_name, 
                         -2, 
                         np.array([-1, -1, -1, -1]), 
                         np.squeeze(np.array(flipped_landmark.reshape(1, -1)))])
        idx = idx + 1
        
        # Drop the faces that are too small or exceed the boundaries
        if max(w, h) < 40 or x1 < 0 or y1 < 0:
            continue
        
        # Augment the cropped face
        for j in range(10):
            
            # Randomly pick a new size & shifts
            new_size = np.random.randint(int(min(w, h) * 0.8), np.ceil(max(w, h) * 1.25)) # new_size = width/height - 1
            delta_x = np.random.randint(-w * 0.15, w * 0.15)
            delta_y = np.random.randint(-h * 0.15, h * 0.15)
            
            new_x1 = int(max(0, x1 + w / 2 - new_size / 2 + delta_x))
            new_y1 = int(max(0, y1 + h / 2 - new_size / 2 + delta_y))
            new_x2 = new_x1 + new_size 
            new_y2 = new_y1 + new_size 
            
            if new_x2 > width or new_y2 > height:
                continue
            
            crop_box = np.array([new_x1, new_y1, new_x2, new_y2])
            iou = IoU(crop_box, np.array(roi).reshape(1, -1))
            
            if iou > 0.65:
                
                cropped_image = image[new_y1: new_y2 + 1, new_x1: new_x2 + 1]
                resized_image = cv2.resize(cropped_image, (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_LINEAR)
                transfered_landmark = np.zeros((6, 2))
                for k in range(6):
                    transfered_landmark[k][0] = (landmark[k * 2] - new_x1) / new_size
                    transfered_landmark[k][1] = (landmark[k * 2 + 1] - new_y1) / new_size
                '''Verify
                temp = cropped_image.copy()
                for k in range(6):
                    cv2.circle(temp, (int(transfered_landmark[k][0] * new_size), int(transfered_landmark[k][1] * new_size)), 2, (0, 255, 0), 1)
                cv2.imshow('t', temp)
                cv2.waitKey()
                cv2.destroyAllWindows() # Verified
                del temp
                '''
                saving_name = str(idx) + '.jpg'
                saving_path = os.path.join(landmark_save_dir, saving_name)
                success = cv2.imwrite(saving_path, resized_image)
                if not success:
                    raise Exception("Landmark picture " + str(idx) + " saving failed. ")
                aug_list.append(['landmark/' + saving_name, 
                                 -2, 
                                 np.array([-1, -1, -1, -1]), 
                                 np.squeeze(np.array(transfered_landmark.reshape(1, -1)))])
                idx = idx + 1
                
                # Mirror 
                # if random.choice([0,1]) == 1:
                flipped_image, flipped_landmark = flip(cropped_image, transfered_landmark)
                '''Verify
                temp = flipped_image.copy()
                for k in range(6):
                    cv2.circle(temp, (int(flipped_landmark[k][0] * new_size), int(flipped_landmark[k][1] * new_size)), 2, (0, 255, 0), 1)
                    cv2.imshow('t', temp)
                    cv2.waitKey()
                    cv2.destroyAllWindows() # Verified
                del temp
                '''
                resized_image = cv2.resize(flipped_image, (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_LINEAR)
                saving_name = str(idx) + '.jpg'
                saving_path = os.path.join(landmark_save_dir, saving_name)
                success = cv2.imwrite(saving_path, resized_image)
                if not success:
                    raise Exception("Landmark picture " + str(idx) + " saving failed. ")
                aug_list.append(['landmark/' + saving_name, 
                                 -2, 
                                 np.array([-1, -1, -1, -1]), 
                                 np.squeeze(np.array(flipped_landmark.reshape(1, -1)))])
                idx = idx + 1
                
                # Anti-Clockwise Rotate 
                # if random.choice([0,1]) == 1:
                # a. Anti-clockwise rotate
                theta = np.random.randint(5, 15)
                rotated_face, rotated_landmark = rotate(image, crop_box, landmark, theta) # rotated_landmark here has not been transfered yet! 
                resized_rotated_face = cv2.resize(rotated_face, (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_LINEAR)
                transfered_rotated_landmark = np.zeros((6, 2))
                for k in range(6):
                    transfered_rotated_landmark[k][0] = (rotated_landmark[k][0] - new_x1) / new_size
                    transfered_rotated_landmark[k][1] = (rotated_landmark[k][1] - new_y1) / new_size
                '''Verify
                temp = rotated_face.copy()
                for k in range(6):
                    cv2.circle(temp, (int(transfered_rotated_landmark[k][0] * new_size), int(transfered_rotated_landmark[k][1] * new_size)), 2, (0, 255, 0), 1)
                    cv2.imshow('t', temp)
                    cv2.waitKey()
                    cv2.destroyAllWindows() # Verified
                del temp
                '''
                saving_name = str(idx) + '.jpg'
                saving_path = os.path.join(landmark_save_dir, saving_name)
                success = cv2.imwrite(saving_path, resized_rotated_face)
                if not success:
                    raise Exception("Landmark picture " + str(idx) + " saving failed. ")
                aug_list.append(['landmark/' + saving_name, 
                                 -2, 
                                 np.array([-1, -1, -1, -1]), 
                                 np.squeeze(np.array(transfered_rotated_landmark.reshape(1, -1)))])
                idx = idx + 1
                
                # b. Anti-clockwise rotate & mirror
                flipped_rotated_face, flipped_transfered_rotated_landmark = flip(rotated_face, transfered_rotated_landmark)
                '''Verify
                temp = flipped_rotated_face.copy()
                for k in range(6):
                    cv2.circle(temp, (int(flipped_transfered_rotated_landmark[k][0] * new_size), int(flipped_transfered_rotated_landmark[k][1] * new_size)), 2, (0, 255, 0), 1)
                    cv2.imshow('t', temp)
                    cv2.waitKey()
                    cv2.destroyAllWindows() # Verified
                del temp
                '''
                resized_flipped_rotated_face = cv2.resize(flipped_rotated_face, (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_LINEAR)
                saving_name = str(idx) + '.jpg'
                saving_path = os.path.join(landmark_save_dir, saving_name)
                success = cv2.imwrite(saving_path, resized_flipped_rotated_face)
                if not success:
                    raise Exception("Landmark picture " + str(idx) + " saving failed. ")
                aug_list.append(['landmark/' + saving_name, 
                                 -2, 
                                 np.array([-1, -1, -1, -1]), 
                                 np.squeeze(np.array(flipped_transfered_rotated_landmark.reshape(1, -1)))])
                idx = idx + 1
            
                # Clockwise Rotate 
                # if random.choice([0,1]) == 1:
                # a. Clockwise rotate
                theta = np.random.randint(5, 15)
                rotated_face, rotated_landmark = rotate(image, crop_box, landmark, -theta) # rotated_landmark here has not been transfered yet! 
                resized_rotated_face = cv2.resize(rotated_face, (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_LINEAR)
                transfered_rotated_landmark = np.zeros((6, 2))
                for k in range(6):
                    transfered_rotated_landmark[k][0] = (rotated_landmark[k][0] - new_x1) / new_size
                    transfered_rotated_landmark[k][1] = (rotated_landmark[k][1] - new_y1) / new_size
                '''Verify
                temp = rotated_face.copy()
                for k in range(6):
                    cv2.circle(temp, (int(transfered_rotated_landmark[k][0] * new_size), int(transfered_rotated_landmark[k][1] * new_size)), 2, (0, 255, 0), 1)
                    cv2.imshow('t', temp)
                    cv2.waitKey()
                    cv2.destroyAllWindows() # Verified
                del temp
                '''
                saving_name = str(idx) + '.jpg'
                saving_path = os.path.join(landmark_save_dir, saving_name)
                success = cv2.imwrite(saving_path, resized_rotated_face)
                if not success:
                    raise Exception("Landmark picture " + str(idx) + " saving failed. ")
                aug_list.append(['landmark/' + saving_name, 
                                 -2, 
                                 np.array([-1, -1, -1, -1]), 
                                 np.squeeze(np.array(transfered_rotated_landmark.reshape(1, -1)))])
                idx = idx + 1
                
                # b. Clockwise rotate & mirror
                flipped_rotated_face, flipped_transfered_rotated_landmark = flip(rotated_face, transfered_rotated_landmark)
                '''Verify
                temp = flipped_rotated_face.copy()
                for k in range(6):
                    cv2.circle(temp, (int(flipped_transfered_rotated_landmark[k][0] * new_size), int(flipped_transfered_rotated_landmark[k][1] * new_size)), 2, (0, 255, 0), 1)
                    cv2.imshow('t', temp)
                    cv2.waitKey()
                    cv2.destroyAllWindows() # Verified
                del temp
                '''
                resized_flipped_rotated_face = cv2.resize(flipped_rotated_face, (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_LINEAR)
                saving_name = str(idx) + '.jpg'
                saving_path = os.path.join(landmark_save_dir, saving_name)
                success = cv2.imwrite(saving_path, resized_flipped_rotated_face)
                if not success:
                    raise Exception("Landmark picture " + str(idx) + " saving failed. ")
                aug_list.append(['landmark/' + saving_name, 
                                 -2, 
                                 np.array([-1, -1, -1, -1]), 
                                 np.squeeze(np.array(flipped_transfered_rotated_landmark.reshape(1, -1)))])
                idx = idx + 1
            
    # Save the augmentation list
    file = open(record_save_dir, 'wb+') 
    pkl.dump(aug_list, file)
    file.close()
    
    print("Processing Finished. ")
    
def parse_arguments(argv):
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--IMG_SIZE", type = int, help = "The size of generated images")
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    
    main(parse_arguments(sys.argv[1: ]))

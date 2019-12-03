#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 01:57:30 2019

@author: tmaysggs
"""

'''Last updated on 12/03/2019 14:47'''
import cv2
import numpy as np

'''计算IoU'''
def IoU(box, boxes):

    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    true_area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    inter_x1 = np.maximum(box[0], boxes[:, 0])
    inter_y1 = np.maximum(box[1], boxes[:, 1])
    inter_x2 = np.minimum(box[2], boxes[:, 2])
    inter_y2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    inter_w = np.maximum(0, inter_x2 - inter_x1 + 1)
    inter_h = np.maximum(0, inter_y2 - inter_y1 + 1)

    intersection = inter_w * inter_h
    union = box_area + true_area - intersection

    iou = intersection / (union + 1e-10)

    return iou

'''求出人脸图像的镜像及其对应的转换后关键点坐标'''
def flip(cropped_image, transfered_landmark):
    
    flipped_image = cv2.flip(cropped_image, 1)
    flipped_landmark = np.asarray([(1 - x, y) for (x, y) in transfered_landmark])
    flipped_landmark[[0, 1]] = flipped_landmark[[1, 0]]
    flipped_landmark[[3, 4]] = flipped_landmark[[4, 3]]
    
    return flipped_image, flipped_landmark

"""求出旋转后的人脸图像及其对应的转换后关键点坐标"""
def rotate(image, crop_box, landmark, theta):
    
    '''
    Arguments:
        image: original image
        landmark: original ground truth landmark
        theta: 旋转角度大小
    Rotation Matrix:
        [[alpha, beta, (1 - alpha) * center.x - beta * center.y], 
        [-beta, alpha, beta * center.x + (1 - alpha) * center.y]]
    where
        alpha = scale * cos(theta)
        beta = scale * sin(theta)
    '''
    new_x1, new_y1, new_x2, new_y2 = crop_box
    center = [(new_x1 + new_x2) / 2, (new_y1 + new_y2) / 2]
    rotation_matrix = cv2.getRotationMatrix2D((center[0], center[1]), theta, 1) # scale = 1
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    rotated_landmark = np.asarray([(rotation_matrix[0][0] * x + rotation_matrix[0][1] * y + rotation_matrix[0][2], 
                                  rotation_matrix[1][0] * x + rotation_matrix[1][1] * y + rotation_matrix[1][2]) for (x, y) in np.array(landmark).reshape(6, 2)])
    '''Verify
    temp = rotated_image.copy()
    for k in range(6):
        cv2.circle(temp, (int(rotated_landmark[k][0]), int(rotated_landmark[k][1])), 2, (0, 255, 0), 1)
        cv2.imshow('t', temp)
        cv2.waitKey()
        cv2.destroyAllWindows() # Verified
    del temp
    '''
    rotated_face = rotated_image[new_y1: new_y2 + 1, new_x1: new_x2 + 1]
    
    return rotated_face, rotated_landmark


'''生成一个缩放比例值列表，该列表中的缩放比例用于缩放图片生成图片金字塔'''
def calculate_scales(img):
    
    temp = img.copy()
    scale = 1.0
    h, w, c = temp.shape
    
    # 过大的图片按比例缩小到窄边为500
    if min(h, w) > 500.0: 
        scale = 500.0 / min(h, w) 
        h = int(h * scale)
        w = int(w * scale)
    
    # 过小的图片按比例扩大到长边为500
    elif max(w, h) < 500.0: 
        scale = 500.0 / max(h, w) 
        h = int(h * scale)
        w = int(w * scale)
    
    # 生成用于缩放图片生成图片金字塔的缩放比例值列表
    scales = []
    factor = 0.709
    factor_count = 0
    min_length = min(h, w)
    while min_length >= 12:
        scales.append(scale * pow(factor, factor_count)) # pow(x, y): x^y 
        min_length *= factor
        factor_count += 1
    
    return scales

'''非极大值抑制：将每个置信度非最高的候选框同置信度最高的候选框比较计算IoU（或IoM）,剔除IoU高于threshold的候选框'''
def NMS(rectangles, threshold, nms_type): # 此处threshold与用于NMS前置任务挑选候选框时所用的threshold无关！！！
    
    if len(rectangles) == 0:
        return rectangles
    
    boxes = np.array(rectangles)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]
    
    true_area = np.multiply(x2 - x1 + 1, y2 - y1 + 1)
    
    # 将scores中的各个概率从小到大排列并输出其在scores中对应的index
    score_order = np.array(scores.argsort())
    # array.argsort() 将array中的元素从小到大排列，提取其对应的index(索引)，然后输出
    
    picked_box_index = []
    while len(score_order) > 0: 
        inter_x1 = np.maximum(x1[score_order[-1]], x1[score_order[0: -1]])
        inter_y1 = np.maximum(y1[score_order[-1]], y1[score_order[0: -1]])
        inter_x2 = np.minimum(x2[score_order[-1]], x2[score_order[0: -1]])
        inter_y2 = np.minimum(y2[score_order[-1]], y2[score_order[0: -1]])
        # index = -1时的候选框概率最高，将其坐标与剩余候选框坐标相比，找出最窄的边缘坐标
        inter_w = np.maximum(0.0, inter_x2 - inter_x1 + 1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1 + 1)
        
        # 计算交集
        intersection = inter_w * inter_h
        
        if nms_type == 'iom': # Intersection over the area of the smallest box (minimum) rather than IoU
            o = intersection / np.minimum(true_area[score_order[-1]], true_area[score_order[0: -1]])
        else: # IoU 
            o = intersection / (true_area[score_order[-1]] + true_area[score_order[0: -1]] - intersection)
        
        # 记录被选中候选框的索引（index）
        picked_box_index.append(score_order[-1])
        # 取出所有IoU/IoM低于阈值的候选框进行下一轮筛选
        score_order = score_order[np.where(o <= threshold)[0]]
    
    # 通过索引取出所有符合要求的候选框，转为list
    result_rectangles = boxes[picked_box_index].tolist()
    
    return result_rectangles # x1, y1, x2, y2, score

'''将矩形人脸候选框转为正方形（边长为较长边边长）'''
def rect2square(rectangles):
    
    '''
    Arguments:
        rectangles: rectangles[i][0: 3] are the coordinates & rectangles[i][4] is the score.
    Returns:
        square bounding box coordinates & the score.
    '''
    w = rectangles[:, 2] - rectangles[:, 0]
    h = rectangles[:, 3] - rectangles[:, 1]
    l = np.maximum(w, h).T
    # 调整x1、y1
    rectangles[:, 0] = rectangles[:, 0] + w * 0.5 - l * 0.5
    rectangles[:, 1] = rectangles[:, 1] + h * 0.5 - l * 0.5
    rectangles[:, 2] = rectangles[:, 0] + l
    rectangles[:, 3] = rectangles[:, 1] + l
    # rectangles[:, 2: 4] = rectangles[:, 0: 2] + np.repeat([l], 2, axis = 0).T
    
    return rectangles

'''从P-Net的输出中选出符合要求的人脸候选框及对应置信度'''
def pnet_detect_face(prob, roi, out_side, scale_reciprocal, width, height, threshold):
    
    in_side = 2 * out_side + 11
    stride = 0
    if out_side != 1:
        stride = float(in_side - 12) / (out_side -1)
    
    # 找出数组prob中所有大于threshold的值的坐标，纵坐标（axis = 0）放入x，横坐标（axis = 1）放入对应位置的y
    # 即：挑选出置信度大于threshold的候选框，用于后续NMS处理
    (x, y) = np.where(prob >= threshold)
    
    # 每对(x, y)点对应输入图像中的一个12x12区域，对应的prob值代表了这个12x12区域中包含人脸的概率
    bbox_position = np.array([x,y]).T 
    
    # 通过该位置坐标还原出该被检测候选框在原输入图像上的坐标
    # (x1, y1)
    bb1 = np.fix((stride * bbox_position + 0 ) * scale_reciprocal) # numpy.fix() Round to nearest integer towards zero.
    # (x2, y2)
    bb2 = np.fix((stride * bbox_position + 11) * scale_reciprocal)
    bounding_box = np.concatenate((bb1, bb2), axis = 1)
    
    # 取出对应的候选框坐标偏移量
    delta_x1 = roi[0][x, y]
    delta_y1 = roi[1][x, y]
    delta_x2 = roi[2][x, y]
    delta_y2 = roi[3][x, y]
    offset = np.array([delta_x1, delta_y1, delta_x2, delta_y2]).T
    
    # 取出对应的“是人脸“概率
    score = np.array([prob[x, y]]).T 
    
    # 通过对应的被检测框坐标和偏移量还原出预测的人脸候选框坐标
    bounding_box = bounding_box + offset * (12.0 * scale_reciprocal)
    
    # 得到候选框并改为正方形
    rectangles = np.concatenate((bounding_box, score), axis = 1)
    rectangles = rect2square(rectangles) # 包含坐标和对应概率：(?, x1, y1, x2, y2, score)
    
    picked_rectangles = []
    # 防止超过原图像边界，去掉超出部分
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        score = rectangles[i][4]
        if x2 > x1 and y2 > y1:
            picked_rectangles.append([x1, y1, x2, y2, score])
    
    result_rectangles = NMS(picked_rectangles, 0.3, 'iou')
    
    return result_rectangles

'''从R-Net的输出中选出符合要求的人脸候选框及对应置信度'''
def rnet_detect_face(prob, roi, rectangles_list, width, height, threshold):
    
    score = prob[:, 1]
    # 选出概率大于阈值的项目的索引
    pick_index = np.where(score >= threshold)
    rectangles_array = np.array(rectangles_list)
    # 挑选出对应项的各个参数
    new_x1 = rectangles_array[pick_index, 0]
    new_y1 = rectangles_array[pick_index, 1]
    new_x2 = rectangles_array[pick_index, 2]
    new_y2 = rectangles_array[pick_index, 3]
    
    score = np.array([score[pick_index]]).T
    
    delta_x1 = roi[pick_index, 0]
    delta_y1 = roi[pick_index, 1]
    delta_x2 = roi[pick_index, 2]
    delta_y2 = roi[pick_index, 3]
    
    w = new_x2 - new_x1
    h = new_y2 - new_y1
    
    x1 = np.array([(new_x1 + delta_x1 * w)[0]]).T
    y1 = np.array([(new_y1 + delta_y1 * h)[0]]).T
    x2 = np.array([(new_x2 + delta_x2 * w)[0]]).T
    y2 = np.array([(new_y2 + delta_y2 * h)[0]]).T
    
    rectangles = np.concatenate((x1, y1, x2, y2, score), axis = 1)
    rectangles = rect2square(rectangles)
    
    picked_rectangles = []
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        score = rectangles[i][4]
        if x2 > x1 and y2 > y1:
            picked_rectangles.append([x1, y1, x2, y2, score])
    
    result_rectangles = NMS(picked_rectangles, 0.3, 'iou')
    
    return result_rectangles

'''从O-Net的输出中选出符合要求的人脸候选框、人脸关键点坐标及对应置信度'''
def onet_detect_face(prob, roi, pts, rectangles, width, height, threshold):
    
    score = prob[:, 1]
    pick_index = np.where(score >= threshold)
    rectangles = np.array(rectangles)
    
    new_x1 = rectangles[pick_index, 0]
    new_y1 = rectangles[pick_index, 1]
    new_x2 = rectangles[pick_index, 2]
    new_y2 = rectangles[pick_index, 3]
    w = new_x2 - new_x1
    h = new_y2 - new_y1
    
    score = np.array([score[pick_index]]).T
    
    delta_x1 = roi[pick_index, 0]
    delta_y1 = roi[pick_index, 1]
    delta_x2 = roi[pick_index, 2]
    delta_y2 = roi[pick_index, 3]
    
    # 根据偏移边框坐标得出人脸关键点坐标
    pts0 = np.array([(pts[pick_index, 0] * w + new_x1)[0]]).T
    pts1 = np.array([(pts[pick_index, 5] * h + new_y1)[0]]).T
    pts2 = np.array([(pts[pick_index, 1] * w + new_x1)[0]]).T
    pts3 = np.array([(pts[pick_index, 6] * h + new_y1)[0]]).T
    pts4 = np.array([(pts[pick_index, 2] * w + new_x1)[0]]).T
    pts5 = np.array([(pts[pick_index, 7] * h + new_y1)[0]]).T
    pts6 = np.array([(pts[pick_index, 3] * w + new_x1)[0]]).T
    pts7 = np.array([(pts[pick_index, 8] * h + new_y1)[0]]).T
    pts8 = np.array([(pts[pick_index, 4] * w + new_x1)[0]]).T
    pts9 = np.array([(pts[pick_index, 9] * h + new_y1)[0]]).T

    # 根据偏移边框坐标得出正确边框坐标
    x1 = np.array([(new_x1 + delta_x1 * w)[0]]).T
    y1 = np.array([(new_y1 + delta_y1 * h)[0]]).T
    x2 = np.array([(new_x2 + delta_x2 * w)[0]]).T
    y2 = np.array([(new_y2 + delta_y2 * h)[0]]).T
    
    rectangles = np.concatenate((x1, y1, x2, y2, score, pts0, pts1, pts2, pts3, pts4, pts5, pts6, pts7, pts8, pts9), axis = 1)
    
    picked_rectangles = []
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        if x2 > x1 and y2 > y1:
            picked_rectangles.append([x1, y1, x2, y2, rectangles[i][4], rectangles[i][5], rectangles[i][6], rectangles[i][7], rectangles[i][8], rectangles[i][9], rectangles[i][10], rectangles[i][11], rectangles[i][12], rectangles[i][13], rectangles[i][14]])
    
    result_rectangles = NMS(picked_rectangles, 0.3, 'iom')
    
    return result_rectangles

'''从修改后的O-Net的输出中选出符合要求的人脸候选框、人脸关键点坐标及对应置信度'''
def new_onet_detect_face(prob, roi, pts, chinpts, rectangles, width, height, threshold):
    
    score = prob[:, 1]
    pick_index = np.where(score >= threshold)
    rectangles = np.array(rectangles)
    
    new_x1 = rectangles[pick_index, 0]
    new_y1 = rectangles[pick_index, 1]
    new_x2 = rectangles[pick_index, 2]
    new_y2 = rectangles[pick_index, 3]
    w = new_x2 - new_x1
    h = new_y2 - new_y1
    
    score = np.array([score[pick_index]]).T
    
    delta_x1 = roi[pick_index, 0]
    delta_y1 = roi[pick_index, 1]
    delta_x2 = roi[pick_index, 2]
    delta_y2 = roi[pick_index, 3]
    
    # 根据偏移边框坐标得出人脸关键点坐标
    pts0 = np.array([(pts[pick_index, 0] * w + new_x1)[0]]).T
    pts1 = np.array([(pts[pick_index, 5] * h + new_y1)[0]]).T
    pts2 = np.array([(pts[pick_index, 1] * w + new_x1)[0]]).T
    pts3 = np.array([(pts[pick_index, 6] * h + new_y1)[0]]).T
    pts4 = np.array([(pts[pick_index, 2] * w + new_x1)[0]]).T
    pts5 = np.array([(pts[pick_index, 7] * h + new_y1)[0]]).T
    pts6 = np.array([(pts[pick_index, 3] * w + new_x1)[0]]).T
    pts7 = np.array([(pts[pick_index, 8] * h + new_y1)[0]]).T
    pts8 = np.array([(pts[pick_index, 4] * w + new_x1)[0]]).T
    pts9 = np.array([(pts[pick_index, 9] * h + new_y1)[0]]).T
    pts10 = np.array([(chinpts[pick_index, 0] * h + new_x1)[0]]).T
    pts11 = np.array([(chinpts[pick_index, 1] * h + new_y1)[0]]).T

    # 根据偏移边框坐标得出正确边框坐标
    x1 = np.array([(new_x1 + delta_x1 * w)[0]]).T
    y1 = np.array([(new_y1 + delta_y1 * h)[0]]).T
    x2 = np.array([(new_x2 + delta_x2 * w)[0]]).T
    y2 = np.array([(new_y2 + delta_y2 * h)[0]]).T
    
    rectangles = np.concatenate((x1, y1, x2, y2, score, pts0, pts1, pts2, pts3, pts4, pts5, pts6, pts7, pts8, pts9, pts10, pts11), axis = 1)
    
    picked_rectangles = []
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        if x2 > x1 and y2 > y1:
            picked_rectangles.append([x1, y1, x2, y2, rectangles[i][4], rectangles[i][5], rectangles[i][6], rectangles[i][7], rectangles[i][8], rectangles[i][9], rectangles[i][10], rectangles[i][11], rectangles[i][12], rectangles[i][13], rectangles[i][14], rectangles[i][15], rectangles[i][16]])
    
    result_rectangles = NMS(picked_rectangles, 0.3, 'iom')
    
    return result_rectangles

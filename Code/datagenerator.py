#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:17:51 2020

@author: yuxuanhe
"""

import os
import cv2
import numpy as np
from skimage.io import imread
from skimage.transform import resize as imresize


def train_generator(train_file, masks_file, batch_size, reshape_size):
    datas = sorted(os.listdir(train_file))[1:]
    masks = sorted(os.listdir(masks_file))[1:]
    num_batch = len(datas)//batch_size
    for i in range(num_batch):
        imgs = []
        masks = []
        train_datas = datas[batch_size*i:batch_size*(i+1)]
        train_masks = masks[batch_size*i:batch_size*(i+1)]
        for img_path in train_datas:
            img = load_img(path1 + img_path, reshape_size) 
            imgs.append(img)
        for mask_path in train_masks:
            mask = load_img(path2 + mask_path, reshape_size)
            masks.append(mask)
        yield imgs, masks
        
def load_img(path, shape):
    image = imread(path)
    image = imresize(image, shape)
    return image
    
train_file = '/Users/yuxuanhe/Desktop/Research IVUS/Carotid-Data/dataset/images_set/'
masks_file = '/Users/yuxuanhe/Desktop/Research IVUS/Carotid-Data/dataset/masks_set/'

def test_generator(test_file, masks_file, batch_size, reshape_size):
    datas = sorted(os.listdir(test_file))[1:]
    masks = sorted(os.listdir(masks_file))[1:]
    num_batch = len(datas)//batch_size
    for i in range(num_batch):
        imgs = []
        masks = []
        train_datas = datas[batch_size*i:batch_size*(i+1)]
        train_masks = masks[batch_size*i:batch_size*(i+1)]
        for img_path in train_datas:
            img = load_img(path1 + img_path, reshape_size) 
            imgs.append(img)
        for mask_path in train_masks:
            mask = load_img(path2 + mask_path, reshape_size)
            masks.append(mask)
        yield imgs, masks
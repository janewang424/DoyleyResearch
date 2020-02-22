# -*- coding: utf-8 -*-
"""
Created on 2019-3-12
@author: LeonShangguan
"""
import os
import cv2

path = '/home/leon/图片/test/'
cnt = 0

for data_file in sorted(os.listdir(path + 'train')):
    cnt = cnt + 1

    img = cv2.imread('train/' + data_file)
    cv2.imwrite('aug/train/' + str(cnt) + 'oL_train.jpg', img)

    hImg = cv2.flip(img,1,dst=None) #水平镜像
    cv2.imwrite('aug/train/' + str(cnt) + 'hL_train.jpg', hImg)
    vImg = cv2.flip(img,0,dst=None) #垂直镜像
    cv2.imwrite('aug/train/' + str(cnt) + 'vL_train.jpg', vImg)
    cImg = cv2.flip(img,-1,dst=None) #对角镜像
    cv2.imwrite('aug/train/' + str(cnt) + 'cL_train.jpg', cImg)
    
    print(data_file)

print('************************************************************************')

cnt = 0

for data_file in sorted(os.listdir(path + 'label')):
    cnt = cnt + 1

    img = cv2.imread('label/' + data_file)
    cv2.imwrite('aug/label/' + str(cnt) + 'oL_label.jpg', img)

    hImg = cv2.flip(img,1,dst=None) #水平镜像
    cv2.imwrite('aug/label/' + str(cnt) + 'hL_label.jpg', hImg)
    vImg = cv2.flip(img,0,dst=None) #垂直镜像
    cv2.imwrite('aug/label/' + str(cnt) + 'vL_label.jpg', vImg)
    cImg = cv2.flip(img,-1,dst=None) #对角镜像
    cv2.imwrite('aug/label/' + str(cnt) + 'cL_label.jpg', cImg)

    print(data_file)

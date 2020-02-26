# -*- coding: utf-8 -*-
"""
Created on 2019-3-12
@author: LeonShangguan
"""
from generater import *
from keras.models import *
import cv2
from segmentation_models import Unet
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

model = load_model('Unet_Pretrained_bce_jaccard_loss_iou_score_tuned.hdf5', compile=False)

img = io.imread("L001.png", as_gray=False)
img = img / 255
# img_copy = io.imread("s.jpg", as_gray=True)/255
flag_multi_class = False
img = trans.resize(img, (512, 512))
print(img.shape)
# img = np.reshape(img, img.shape) if (not flag_multi_class) else img
print(img.shape)
img = np.reshape(img, (1,)+img.shape)
print(img.shape)
result = model.predict(img)
print(result.shape)
saveResult("a.jpg", result[0])

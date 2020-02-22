import os
from keras.models import *
import cv2

from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
import glob
import io
import skimage.io as io
import skimage.transform as trans
import numpy as np
import os


model = load_model('unet_seresnet50_100_line_512.hdf5')

def saveResult(name,save_path, npyfile, flag_multi_class=False, num_class=3):
    for i, item in enumerate(npyfile):
        name = name.split('/')
        
        name = name[-1]
        name = name.split('.')
        name = name[0]
        img = labelVisualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:, :, 0]
        img = cv2.resize(img, (512, 512)) 
        io.imsave(os.path.join(save_path, "%s_predict.jpg" % name), img)
path = r"data_512/scratch_512/"
#save_path = "result/line_result/"
for i in glob.glob(path + '*.jpg'):
    img = io.imread(i, as_gray=False)
    img = img / 255
    #img_copy = io.imread("1.jpg", as_gray=True)/255
    flag_multi_class = False
    img = trans.resize(img, (512, 512))
    #print(img.shape)
    img = np.reshape(img, img.shape) if (not flag_multi_class) else img
    #print(img.shape)
    img = np.reshape(img, (1,)+img.shape)
    #print(img.shape)
    result = model.predict(img)
    #print(result.shape)
    saveResult(i,path, result, flag_multi_class=False)

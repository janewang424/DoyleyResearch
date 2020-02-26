# -*- coding: utf-8 -*-
"""
Created on 2019-3-11
@author: LeonShangguan
"""
from segmentation_models import Unet
# from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from generater import *
import numpy as np
from keras.models import load_model
from segmentation_models import Unet
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from keras.utils import multi_gpu_model


# BACKBONE = 'resnet34'
# preprocess_input = get_preprocessing(BACKBONE)

# define model
# model = Unet(classes=1, encoder_weights='imagenet')
             # decoder_filters=[512, 256, 128, 64, 32])
model = load_model('Unet_Pretrained_bce_jaccard_loss_iou_score_tuned.hdf5', compile=False)

# parallel_model = multi_gpu_model(model, gpus=2)
# parallel_model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])
model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])
model.summary()

data_gen_args = dict(fill_mode='nearest')
train_data = trainGenerator(12, data_gen_args, save_to_dir=None)
val_data = valGenerator(12, data_gen_args, save_to_dir=None)

save_path = 'Unet_Pretrained_bce_jaccard_loss_iou_score_tuned_10epoches' + '.hdf5'

callbacks = [EarlyStopping(monitor='val_loss',
                           patience=8,
                           verbose=1,
                           min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=4,
                               verbose=1,
                               epsilon=1e-4),
             ModelCheckpoint(monitor='val_loss',
                             filepath=save_path,
                             verbose=True,
                             save_best_only=True)]

model.fit_generator(train_data,
                    steps_per_epoch=np.ceil(11122/12),
                    epochs=6,
                    callbacks=callbacks,
                    validation_data=val_data,
                    validation_steps=np.ceil(2766/12),
                    verbose=1)


# parallel_model.fit_generator(train_data,
#                     steps_per_epoch=120,
#                     epochs=50,
#                     verbose=2,
#                     callbacks=callbacks,
#                     # validation_data=val_data,
#                     validation_steps=90,)


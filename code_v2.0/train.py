# -*- coding: utf-8 -*-
"""
Created on 2019-3-11
@author: LeonShangguan
"""

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from segmentation_models import Unet
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

# BACKBONE = 'resnet34'
# preprocess_input = get_preprocessing(BACKBONE)

# define model
model = Unet(classes=1, encoder_weights='imagenet', decoder_filters=[512, 256, 128, 64, 32])
model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])
model.summary()

data_gen_args = dict(fill_mode='nearest')
from dataset import Carotid_DataGenerator
train_data = Carotid_DataGenerator(df_path='dataset/split/train_fold_1_seed_960630.csv',
                                   image_path='../Carotid-Data/Carotid-Data/images/',
                                   mask_path='../Carotid-Data/Carotid-Data/masks/',
                                   batch_size=4,
                                   target_shape=(512, 512),
                                   shuffle=False)
val_data = Carotid_DataGenerator(df_path='dataset/split/val_fold_1_seed_960630.csv',
                                 image_path='../Carotid-Data/Carotid-Data/images/',
                                 mask_path='../Carotid-Data/Carotid-Data/masks/',
                                 batch_size=4,
                                 target_shape=(512, 512),
                                 shuffle=False)

save_path = 'Unet_Pretrained_bce_jaccard_loss_iou_newsplit1' + '.hdf5'

callbacks = [EarlyStopping(monitor='val_loss',
                           patience=8,
                           verbose=1,
                           min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=4,
                               verbose=1,
                               epsilon=1e-4), #min_delta
             ModelCheckpoint(monitor='val_loss',
                             filepath=save_path,
                             verbose=True,
                             save_best_only=True)]

model.fit_generator(train_data,
                    validation_data=val_data,
                    epochs=6,
                    callbacks=callbacks,
                    verbose=1)

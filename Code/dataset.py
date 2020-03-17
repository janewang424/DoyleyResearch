# -*- coding: utf-8 -*-
"""
Created on 2020-3-17
@author: LeonShangguan
"""
import os
import keras
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def get_train_val_split(data_path="../Carotid-Data/Carotid-Data/images/",
                        save_path="dataset/",
                        n_splits=5,
                        seed=960630):
    os.makedirs(save_path + '%s_split_seed_%s' % (n_splits, seed) +'/', exist_ok=True)
    kf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
    df = pd.DataFrame(os.listdir(data_path))

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        train_df, val_df = [], []
        for image_dir in df.iloc[train_idx].values:
            image_ids = os.listdir(data_path + image_dir[0])
            for image_id in image_ids:
                train_df.append(image_dir[0] + '/' + image_id)

        for image_dir in df.iloc[val_idx].values:
            image_ids = os.listdir(data_path + image_dir[0])
            for i in range(len(image_ids)):
                val_df.append(image_dir[0] + '/' + image_ids[i])

        pd.DataFrame(train_df).to_csv(save_path + 'split/' + '/train_fold_%s_seed_%s.csv' % (fold, seed))
        pd.DataFrame(val_df).to_csv(save_path + 'split/' + '/val_fold_%s_seed_%s.csv' % (fold, seed))

    return


class Carotid_DataGenerator(keras.utils.Sequence):
    def __init__(self,
                 df_path,
                 image_path,
                 mask_path,
                 batch_size,
                 target_shape,
                 shuffle=False):

        self.df = pd.read_csv(df_path)['0']
        self.image_path = image_path
        self.mask_path = mask_path
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.shuffle = shuffle
        self.indexes = np.arange(self.df.shape[0])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, len(self.indexes))
        indexes = self.indexes[start:end]

        images = np.empty((len(indexes), *self.target_shape, 3), dtype=np.float32)
        masks = np.empty((len(indexes), *self.target_shape, 3), dtype=np.int32)

        image_ids = self.df.iloc[indexes]
        for i, image_id in enumerate(image_ids):
            image = cv2.resize(cv2.imread(self.image_path + image_id), self.target_shape, cv2.INTER_AREA)
            mask = cv2.resize(cv2.imread(self.mask_path + image_id), self.target_shape, cv2.INTER_AREA)

            images[i, ] = np.asarray(image/255, dtype=np.float32)
            masks[i, ] = np.asarray(mask/255, dtype=np.int32)

        return images, masks

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


if __name__ == "__main__":
    data_loader = Carotid_DataGenerator(df_path='/home/datascience/Leon/DoyleyResearch/Code/dataset/split/train_fold_0_seed_960630.csv',
                                        image_path='/home/datascience/Leon/DoyleyResearch/Carotid-Data/Carotid-Data/images/',
                                        mask_path='/home/datascience/Leon/DoyleyResearch/Carotid-Data/Carotid-Data/masks/',
                                        batch_size=8,
                                        target_shape=(512, 512),
                                        shuffle=True)
    from keras.models import Sequential

    model = Sequential()
    model.compile('Adam')
    model.fit_generator(generator=data_loader)

    print(type(data_loader))
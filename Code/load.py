import numpy as np
from imgaug import augmenters as iaa
import cv2
from sklearn.utils import shuffle
from keras.utils import Sequence
WORKERS = 2
CHANNEL = 3

import warnings
warnings.filterwarnings("ignore")


class data_generator(Sequence):

    def create_train(dataset_info, batch_size, shape, augument=True):
        assert shape[2] == 3
        while True:
            dataset_info = shuffle(dataset_info)
            for start in range(0, len(dataset_info), batch_size):
                end = min(start + batch_size, len(dataset_info))
                batch_images = []
                X_train_batch = dataset_info[start:end]
                batch_labels = np.zeros((len(X_train_batch), NUM_CLASSES))
                for i in range(len(X_train_batch)):
                    image = data_generator.load_image(
                        X_train_batch[i]['path'], shape)
                    if augument:
                        image = data_generator.augment(image)
                    batch_images.append(image / 255.)
                    batch_labels[i][X_train_batch[i]['labels']] = 1

                yield np.array(batch_images, np.float32), batch_labels

    def create_valid(dataset_info, batch_size, shape, augument=False):
        assert shape[2] == 3
        while True:
            # dataset_info = shuffle(dataset_info)
            for start in range(0, len(dataset_info), batch_size):
                end = min(start + batch_size, len(dataset_info))
                batch_images = []
                X_train_batch = dataset_info[start:end]
                batch_labels = np.zeros((len(X_train_batch), NUM_CLASSES))
                for i in range(len(X_train_batch)):
                    image = data_generator.load_image(
                        X_train_batch[i]['path'], shape)
                    if augument:
                        image = data_generator.augment(image)
                    batch_images.append(image / 255.)
                    batch_labels[i][X_train_batch[i]['labels']] = 1
                yield np.array(batch_images, np.float32), batch_labels

    def load_image(path, shape):
        image = cv2.imread(path + '.png')
        image = cv2.resize(image, (SIZE, SIZE))
        return image

    def augment(image):
        augment_img = iaa.Sequential([
            #             sometimes(
            #             iaa.OneOf([
            #                 # iaa.AddToHueAndSaturation((-20, 20)),
            # #                 iaa.Add((-10, 10), per_channel=0.5),
            # #                 iaa.Multiply((0.9, 1.1), per_channel=0.5),
            # #                 # iaa.GaussianBlur((0, 0.5)), # blur images with a sigma between 0 and 3.0
            # #                 iaa.ContrastNormalization((0.8, 1.2), per_channel=0.5), # improve or worsen the contrast
            # #                 iaa.Sharpen(alpha=(0, 0.2), lightness=(0.8, 1.2)), # sharpen images
            # #                 iaa.Emboss(alpha=(0, 0.5), strength=(0, 0.5)), # emboss images
            #                 iaa.Crop(percent=(0, 0.1))
            #                 ])
            #             ),
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                # iaa.Flipud(0.5),
            ])], random_order=True)

        image_aug = augment_img.augment_image(image)
        return image_aug

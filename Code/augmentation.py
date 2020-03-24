# -*- coding: utf-8 -*-
"""
Created on 2020-3-23
@author: LeonShangguan
"""
import cv2
import numpy as np


def do_identity(image, mask):
    return image, mask


def do_horizon_flip(image, mask, p=0.5):
    if np.random.uniform(0, 1) < p:
        image = cv2.flip(image, 1, dst=None)
        mask = cv2.flip(mask, 1, dst=None)
        return image, mask  # 垂直镜像
    else:
        return image, mask


def do_vertical_flip(image, mask, p=0.5):
    if np.random.uniform(0, 1) < p:
        image = cv2.flip(image, 0, dst=None)
        mask = cv2.flip(mask, 0, dst=None)
        return image, mask  # 垂直镜像
    else:
        return image, mask


def do_diagonal_flip(image, mask, p=0.5):
    if np.random.uniform(0, 1) < p:
        image = cv2.flip(image, -1, dst=None)
        mask = cv2.flip(mask, -1, dst=None)
        return image, mask  # 垂直镜像
    else:
        return image, mask


def do_random_rotate(image, mask, magnitude=0.5, p=0.5):
    if np.random.uniform(0, 1) < p:
        angle = np.random.uniform(-1, 1)*180*magnitude

        height, width = image.shape[:2]
        cx, cy = width // 2, height // 2

        transform = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        image = cv2.warpAffine(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpAffine(mask, transform, (width, height), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image, mask


#!/usr/bin/env python3
"""
@Filename:    dataset_generator.py
@Author:      dulanj
@Time:        15/10/2021 20:25
"""
import os
import random
import shutil

import cv2
import numpy as np

NO_OF_IMAGES = 150
IMAGE_SHAPE = (512, 512, 3)
PAD = 150

DATASET_DIR = "dataset"
IMG_DIR = os.path.join(DATASET_DIR, "JPEGImages")
MASK_DIR = os.path.join(DATASET_DIR, "SegmentationClass")
FILE_DIR = os.path.join(DATASET_DIR, "ImageSets/Segmentation")

train_txt_file_voc = os.path.join(DATASET_DIR, "ImageSets/Segmentation/train.txt")
val_txt_file_voc = os.path.join(DATASET_DIR, "ImageSets/Segmentation/val.txt")


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if os.path.exists(DATASET_DIR):
    shutil.rmtree(DATASET_DIR)

for _dir in [DATASET_DIR, IMG_DIR, MASK_DIR, FILE_DIR]:
    create_dir(_dir)


def draw_circle(_image, point, color=(128, 0, 0)):
    return cv2.circle(_image, point, PAD, color=color, thickness=-1)


tf = open(train_txt_file_voc, 'w')
vf = open(val_txt_file_voc, 'w')

for i in range(NO_OF_IMAGES):
    image = np.full(IMAGE_SHAPE, fill_value=255, dtype=np.uint8)
    mask = np.full(IMAGE_SHAPE, fill_value=0, dtype=np.uint8)

    random_point = np.random.random_integers(PAD, IMAGE_SHAPE[0] - PAD, size=(2))

    image = draw_circle(image, random_point, color=(0, 0, 0))
    mask = draw_circle(mask, random_point)

    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
    _basename = f"image_{i}"
    cv2.imwrite(os.path.join(IMG_DIR, f"{_basename}.jpg"), image)
    cv2.imwrite(os.path.join(MASK_DIR, f"{_basename}.png"), mask)

    if random.random() < 0.7:
        tf.write(_basename + '\n')
    else:
        vf.write(_basename + '\n')

    print(i)

tf.close()
vf.close()

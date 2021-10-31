#!/usr/bin/env python3
"""
@Filename:    dataset.py
@Author:      sgx team
@Time:        02/10/2021 00:01
----------------------------------------
Dataloader for CHIP dataset
instance-level_human_parsing
Link - https://keras.io/examples/vision/deeplabv3_plus/
"""

import os
from glob import glob

import tensorflow as tf

from deeplab.params import (
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_TRAIN_IMAGES,
    NUM_VAL_IMAGES,
    DATASET_DIR
)

train_data_dir = os.path.join(DATASET_DIR, "Training")
val_data_dir = os.path.join(DATASET_DIR, "Validation")

train_images = sorted(glob(os.path.join(train_data_dir, "Images/*")))[:NUM_TRAIN_IMAGES]
train_masks = sorted(glob(os.path.join(train_data_dir, "Category_ids/*")))[:NUM_TRAIN_IMAGES]
val_images = sorted(glob(os.path.join(val_data_dir, "Images/*")))[:NUM_VAL_IMAGES]
val_masks = sorted(glob(os.path.join(val_data_dir, "Category_ids/*")))[:NUM_VAL_IMAGES]


def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)
    if mask:
        image = tf.image.decode_png(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE[0], IMAGE_SIZE[1]])
    else:
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE[0], IMAGE_SIZE[1]])
        image = image / 127.5 - 1
    return image


def load_data(image_list, mask_list):
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    return image, mask


def data_generator(image_list, mask_list):
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset


def load_dataset():
    train_dataset = data_generator(train_images, train_masks)
    val_dataset = data_generator(val_images, val_masks)

    print("Train Dataset:", train_dataset)
    print("Val Dataset:", val_dataset)

    return train_dataset, val_dataset


if __name__ == '__main__':
    load_dataset()
#!/usr/bin/env python3
"""
@Filename:    augmentation.py
@Author:      sgx team
@Time:        07/10/2021 20:37
"""
import random

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers.experimental import preprocessing

from deeplab.params import AUG_PROBABILITY


class Augment(tf.keras.layers.Layer):
    """
    Custom augmentations class
    """

    def __init__(self, seed=42):
        super().__init__()
        self.flip = (preprocessing.RandomFlip(mode="horizontal", seed=seed), AUG_PROBABILITY["flip"])
        self.rotate = (preprocessing.RandomRotation(0.2, seed=seed, fill_mode="constant"), AUG_PROBABILITY["rotate"])
        self.trans = (preprocessing.RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1),
                                                      seed=seed, fill_mode='constant'), AUG_PROBABILITY["trans"])
        self.scale = (preprocessing.RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2), seed=seed,
                                               fill_mode='constant'), AUG_PROBABILITY["scale"])

        self.hue = (lambda x: tf.image.random_hue(x, 0.1), AUG_PROBABILITY["hue"])
        self.gau2d = (
            lambda x: tfa.image.gaussian_filter2d(x, filter_shape=(3, 3), sigma=1.0), AUG_PROBABILITY["gaussian"])

    def call(self, inputs, labels):
        augmented = tf.concat((inputs, labels), axis=-1)
        """
        Overall shape augmentation,(applying for both image and mask)
        """
        for _aug, _prob in [self.rotate, self.flip, self.trans, self.scale]:
            if random.random() < _prob:
                augmented = _aug(augmented)
        inputs, labels = augmented[:, :, :3], augmented[:, :, -1]

        """
        Image(inputs) only augmentations (pixel-wise)
        """
        for _aug, _prob in [self.hue, self.gau2d]:
            if random.random() < _prob:
                inputs = _aug(inputs)
        return inputs, labels

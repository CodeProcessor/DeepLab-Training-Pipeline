#!/usr/bin/env python3
"""
@Filename:    augmentation.py
@Author:      dulanj
@Time:        07/10/2021 20:37
"""
import random

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

from deeplab.params import AUG_PROBABILITY


class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        self.flip = (preprocessing.RandomFlip(mode="horizontal", seed=seed), AUG_PROBABILITY["flip"])
        self.rotate = (preprocessing.RandomRotation(0.2, seed=seed, fill_mode="constant"), AUG_PROBABILITY["rotate"])
        self.trans = (preprocessing.RandomTranslation(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2),
                                                      seed=seed, fill_mode='constant'), AUG_PROBABILITY["trans"])
        self.scale = (preprocessing.RandomZoom(height_factor=(-1, 0.5), width_factor=(-1, 0.5), seed=seed,
                                               fill_mode='constant'), AUG_PROBABILITY["scale"])

    def call(self, inputs, labels):
        augmented = tf.concat((inputs, labels), axis=-1)
        for _aug, _prob in [self.rotate, self.flip, self.trans, self.scale]:
            if random.random() < _prob:
                augmented = _aug(augmented)
        # augmented = self.scale(self.trans(self.flip(tf.concat((inputs, labels), axis=-1))))
        return augmented[:, :, :3], augmented[:, :, -1]

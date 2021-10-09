#!/usr/bin/env python3
"""
@Filename:    augmentation.py
@Author:      dulanj
@Time:        07/10/2021 20:37
"""

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing


class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        self.flip = preprocessing.RandomFlip(mode="horizontal", seed=seed)
        self.rotate = preprocessing.RandomRotation(0.2, seed=seed, fill_mode="constant")
        self.trans = preprocessing.RandomTranslation(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2),
                                                     seed=seed, fill_mode='constant')
        self.scale = preprocessing.RandomZoom(height_factor=(-0.2, 0), width_factor=(-0.2, 0), seed=seed,
                                              fill_mode='constant')

    def call(self, inputs, labels):
        augmented = self.scale(self.trans(self.flip(tf.concat((inputs, labels), axis=-1))))
        return augmented[:, :, :3], augmented[:, :, -1]

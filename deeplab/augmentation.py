#!/usr/bin/env python3
"""
@Filename:    augmentation.py
@Author:      dulanj
@Time:        07/10/2021 20:37
"""

import tensorflow as tf


class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        # both use the same seed, so they'll make the same random changes.
        self.flip_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.flip_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
        self.rotate_inputs = tf.keras.layers.RandomRotation(0.2, seed=seed)
        self.rotate_labels = tf.keras.layers.RandomRotation(0.2, seed=seed)
        self.trans_inputs = tf.keras.layers.RandomTranslation(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2),
                                                              seed=seed)
        self.trans_labels = tf.keras.layers.RandomTranslation(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2),
                                                              seed=seed)

    def call(self, inputs, labels):
        inputs = self.trans_inputs(self.flip_inputs(self.rotate_inputs(inputs)))
        labels = self.trans_labels(self.flip_labels(self.rotate_labels(labels)))
        return inputs, labels

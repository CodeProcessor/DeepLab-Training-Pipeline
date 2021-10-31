#!/usr/bin/env python3
"""
@Filename:    preprocess.py
@Author:      dulanj
@Time:        07/10/2021 20:49
"""
import random

import tensorflow as tf
import tensorflow_addons as tfa

from deeplab.params import IMAGE_SIZE, IGNORED_CLASS_ID, AUG_PROBABILITY


class PreProcess(tf.keras.layers.Layer):
    """
    This class will handle most of the preprocess functions
    """
    _MEAN_RGB = [123.15, 115.90, 103.06]

    def __init__(self, output_shape, backbone):
        super().__init__()
        self.target_height = output_shape[0]
        self.target_width = output_shape[1]
        self.backbone = backbone
        self.resize = tf.keras.layers.Resizing(height=self.target_height, width=self.target_width,
                                               interpolation='nearest')
        self.rgb_mean = tf.math.multiply(tf.ones((IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
                                         tf.constant(PreProcess._MEAN_RGB, dtype=tf.float32))

        self.hue = (lambda x: tf.image.random_hue(x, 0.1), AUG_PROBABILITY["hue"])
        self.gau2d = (
            lambda x: tfa.image.gaussian_filter2d(x, filter_shape=(3, 3), sigma=1.0), AUG_PROBABILITY["gaussian"])

    def call(self, image, label):
        """
        Get image and label(mask) do the preprocessing and return
        :param image: Image numpy array
        :param label: Image numpy array
        :return: preprocessed image
        """

        # Set the ignore label
        mask = label != 255
        label = tf.where(mask, x=label, y=IGNORED_CLASS_ID)
        image = tf.cast(self.resize(image), dtype=tf.float32)

        """
        Image(inputs) only augmentations (pixel-wise)
        """
        for _aug, _prob in [self.hue, self.gau2d]:
            if random.random() < _prob:
                image = _aug(image)

        # image = tf.subtract(image, self.rgb_mean)
        """
        Apply Keras recommended pre-process for the relevent model
        Keras Models - https://keras.io/api/applications/
        """
        if self.backbone == "resnet50":
            # Link - https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/preprocess_input
            image = tf.keras.applications.resnet50.preprocess_input(image)
        elif self.backbone == "xception":
            # Link - https://www.tensorflow.org/api_docs/python/tf/keras/applications/xception/preprocess_input
            image = tf.keras.applications.xception.preprocess_input(image)
        elif self.backbone == "mobilenetv2":
            image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        else:
            raise "Unknown backbone"
        # Resize the mask
        label = tf.cast(self.resize(label), dtype=tf.float32)

        return image, label

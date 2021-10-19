#!/usr/bin/env python3
"""
@Filename:    preprocess.py
@Author:      dulanj
@Time:        07/10/2021 20:49
"""
import tensorflow as tf

from deeplab.params import IMAGE_SIZE


class PreProcess(tf.keras.layers.Layer):
    _MEAN_RGB = [123.15, 115.90, 103.06]

    def __init__(self, output_shape, backbone):
        super().__init__()
        self.target_height = output_shape[0]
        self.target_width = output_shape[1]
        self.backbone = backbone
        self.resize = tf.keras.layers.Resizing(height=self.target_height, width=self.target_width)
        self.rgb_mean = tf.math.multiply(tf.ones((IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
                                         tf.constant(PreProcess._MEAN_RGB, dtype=tf.float32))

    def call(self, image, label):
        image = self.resize(image)
        # image = tf.subtract(image, self.rgb_mean)
        if self.backbone == "resnet50":
            image = tf.keras.applications.resnet50.preprocess_input(image)
        elif self.backbone == "xception":
            image = tf.keras.applications.xception.preprocess_input(image)
        elif self.backbone == "mobilenetv2":
            image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        else:
            raise "Unknown backbone"
        label = self.resize(label)

        return image, label

#!/usr/bin/env python3
"""
@Filename:    preprocess.py
@Author:      dulanj
@Time:        07/10/2021 20:49
"""
import tensorflow as tf


class PreProcess(tf.keras.layers.Layer):
    def __init__(self, output_shape):
        super().__init__()
        self.target_height = output_shape[0]
        self.target_width = output_shape[1]
        self.resize = tf.keras.layers.Resizing(height=self.target_height, width=self.target_width)

    def call(self, image, label):
        # rescaled_size = tf.shape(label)[:2]
        # min_index = tf.math.argmin(rescaled_size)
        # pad_value = abs(rescaled_size[0] - rescaled_size[1])
        #
        # vertical_pad_up = 0
        # vertical_pad_down = 0
        # horizontal_pad_right = 0
        # horizontal_pad_left = 0
        #
        # if min_index == 0:
        #     vertical_pad_up = pad_value // 2
        #     vertical_pad_down = pad_value - vertical_pad_up
        # else:
        #     horizontal_pad_left = pad_value // 2
        #     horizontal_pad_right = pad_value - horizontal_pad_left
        #
        # padding = [[vertical_pad_up, vertical_pad_down], [horizontal_pad_left, horizontal_pad_right], [0, 0]]
        # # padding_label = [[vertical_pad_up, vertical_pad_down], [horizontal_pad_left, horizontal_pad_right], [0, 0]]
        # image = tf.pad(image, padding, "CONSTANT", constant_values=0)
        # label = tf.pad(label, padding, "CONSTANT", constant_values=0)

        image = self.resize(image)
        label = self.resize(label)

        return image, label

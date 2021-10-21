#!/usr/bin/env python3
"""
@Filename:    utils.py
@Author:      https://github.com/leimao/DeepLab-V3, teharaf
@Time:        17/10/2021 19:22
"""
import random

import cv2
import numpy as np
import tensorflow as tf

from deeplab.params import IMAGE_SIZE, IGNORED_CLASS_ID, AUG_PROBABILITY, NUM_CLASSES


def flip_image_and_label(image, label):
    image_flipped = np.fliplr(image)
    label_flipped = np.fliplr(label)

    return image_flipped, label_flipped


def resize_image_and_label(image, label, output_size):
    '''
    output_size: [height, width]
    '''

    image_resized = cv2.resize(image, (output_size[1], output_size[0]), interpolation=cv2.INTER_LINEAR)
    label_resized = cv2.resize(label, (output_size[1], output_size[0]), interpolation=cv2.INTER_NEAREST)

    return image_resized, label_resized


def pad_image_and_label(image, label, top, bottom, left, right, pixel_value=0, label_value=IGNORED_CLASS_ID):
    '''
    https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_core/py_basic_ops/py_basic_ops.html#making-borders-for-images-padding
    '''

    image_padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pixel_value)
    label_padded = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT, value=label_value)

    return image_padded, label_padded


def random_crop(image, label, output_size):
    assert image.shape[0] >= output_size[0] and image.shape[1] >= output_size[
        1], 'image size smaller than the desired output size.'

    height_start = np.random.randint(image.shape[0] - output_size[0] + 1)
    width_start = np.random.randint(image.shape[1] - output_size[1] + 1)
    height_end = height_start + output_size[0]
    width_end = width_start + output_size[1]

    image_cropped = image[height_start:height_end, width_start:width_end]
    label_cropped = label[height_start:height_end, width_start:width_end]

    return image_cropped, label_cropped


def hist_plot(image, label):
    import matplotlib.pyplot as plt
    plt.hist(label.ravel(), log=True, bins=range(NUM_CLASSES))
    plt.show()


def plot_wrapper(image, label):
    tf.numpy_function(func=hist_plot, inp=(image, label), Tout=())
    return image, label

def intensity_aug(image):
    # This is not used

    # Blur
    if np.random.random() < AUG_PROBABILITY["gaussian"]:
        image = cv2.blur(src=image, ksize=(3, 3))

    # Brightness
    def brightness(img, val):
        value = random.random() * val
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        # hsv[:, :, 1] = hsv[:, :, 1] + value
        # hsv[:, :, 1][hsv[:, :, 1] > 127] = 127
        hsv[:, :, 2] = hsv[:, :, 2] + value
        hsv[:, :, 2][hsv[:, :, 2] > 127] = 127
        hsv = np.array(hsv, dtype=np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return img

    if np.random.random() < AUG_PROBABILITY["brightness"]:
        image = brightness(image, 30)

    # Hue
    def hue(img, val):
        value = random.random() * val
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        _c = 2
        hsv[:, :, _c] = hsv[:, :, _c] + value
        _cond = hsv[:, :, 1] > 127
        hsv[:, :, _c][_cond] = hsv[:, :, _c][_cond] % 127
        hsv = np.array(hsv, dtype=np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return img

    if np.random.random() < AUG_PROBABILITY["hue"]:
        image = hue(image, 1)

    return image


def augment(image, label, output_size=IMAGE_SIZE, min_scale_factor=0.5, max_scale_factor=2.0):
    original_height = image.shape[0]
    original_width = image.shape[1]
    target_height = output_size[0]
    target_width = output_size[1]

    scale_factor = np.random.uniform(low=min_scale_factor, high=max_scale_factor)

    rescaled_size = [round(original_height * scale_factor), round(original_width * scale_factor)]

    image, label = resize_image_and_label(image=image, label=label, output_size=rescaled_size)

    vertical_pad = round(target_height * 1.5) - rescaled_size[0]
    if vertical_pad < 0:
        vertical_pad = 0
    vertical_pad_up = vertical_pad // 2
    vertical_pad_down = vertical_pad - vertical_pad_up

    horizonal_pad = round(target_width * 1.5) - rescaled_size[1]
    if horizonal_pad < 0:
        horizonal_pad = 0
    horizonal_pad_left = horizonal_pad // 2
    horizonal_pad_right = horizonal_pad - horizonal_pad_left

    image, label = pad_image_and_label(image=image, label=label, top=vertical_pad_up, bottom=vertical_pad_down,
                                       left=horizonal_pad_left, right=horizonal_pad_right, pixel_value=0,
                                       label_value=IGNORED_CLASS_ID)

    image, label = random_crop(image=image, label=label, output_size=output_size)

    # Flip image and label
    if np.random.random() < AUG_PROBABILITY["flip"]:
        image, label = flip_image_and_label(image=image, label=label)

    label = np.expand_dims(label, axis=2)
    return image, label


class AugmentationWrapper(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs, labels):
        image, label = tf.numpy_function(func=augment,
                                         inp=(inputs, labels),
                                         Tout=(tf.float32, tf.float32))
        image.set_shape(tf.TensorShape([IMAGE_SIZE[0], IMAGE_SIZE[1], 3]))
        label.set_shape(tf.TensorShape([IMAGE_SIZE[0], IMAGE_SIZE[1], 1]))
        return image, label


def post_process(image):
    _MEAN_RGB = [123.15, 115.90, 103.06]
    for i in range(len(_MEAN_RGB)):
        image[:, :, i] += _MEAN_RGB[i]
    return image

#!/usr/bin/env python3
"""
@Filename:    overlay.py
@Author:      dulanj
@Time:        02/10/2021 19:22
"""
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from deeplab.dataset_voc import read_image
from deeplab.inference import inference
from deeplab.pascal_voc import VOC_COLORMAP

# Loading the Colormap
colormap = np.array(VOC_COLORMAP).astype(np.uint8)


def decode_segmentation_masks(mask, colormap, n_classes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


def get_overlay(image, colored_mask):
    image = tf.keras.preprocessing.image.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    is_bg = np.tile(np.all(colored_mask == (0, 0, 0), axis=-1), (3, 1, 1)).transpose(1, 2, 0)
    colored_mask = np.where(is_bg, image, colored_mask)
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
    return overlay


def press(event):
    print('press', event.key)
    if event.key == "escape":
        plt.close()
        sys.exit(0)


def plot_samples_matplotlib(display_list, figsize=(5, 3)):
    fig, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    fig.canvas.mpl_connect('key_press_event', press)
    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 3:
            axes[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        else:
            axes[i].imshow(display_list[i])
    plt.show()


def plot_predictions(images_list, model):
    pred_list = []
    for image_file in images_list:
        image_tensor = read_image(image_file)
        prediction_mask = inference(image_tensor=image_tensor, model=model)
        prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, 20)
        overlay = get_overlay(image_tensor, prediction_colormap)
        predict_image_list = [image_tensor, overlay, prediction_colormap]
        plot_samples_matplotlib(
            predict_image_list, figsize=(18, 14)
        )
        pred_list.append(predict_image_list)
    return pred_list


def save_cv_image(save_path, image):
    cv2.imwrite(save_path, cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR))

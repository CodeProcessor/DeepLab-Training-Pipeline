#!/usr/bin/env python3
"""
@Filename:    inference.py
@Author:      dulanj
@Time:        02/10/2021 19:18
"""
import numpy as np

from deeplab.dataset_voc import read_image


def inference(model, image_path="", image_tensor=None):
    """
    Get the model and do the inference and post process it
    :param model: Keras Model
    :param image_path: Image path
    :param image_tensor: Tensor
    :return: prediction
    """
    if image_tensor is None:
        image_tensor = read_image(image_path)
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions

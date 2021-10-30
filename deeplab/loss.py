#!/usr/bin/env python3
"""
@Filename:    loss.py
@Author:      dulanj
@Time:        17/10/2021 22:54
"""
import tensorflow as tf

from deeplab.params import IGNORED_CLASS_ID


def loss_initializer_v2(y_true, y_pred):
    """
    Loss funstion with ignore label considered
    :param y_true: Annotation
    :param y_pred: Prediction
    :return: Cost
    """
    mask = y_true != IGNORED_CLASS_ID
    sce = tf.keras.losses.SparseCategoricalCrossentropy()
    return sce(y_true, y_pred, sample_weight=mask)

#!/usr/bin/env python3
"""
@Filename:    loss.py
@Author:      dulanj
@Time:        17/10/2021 22:54
"""
import tensorflow as tf

from deeplab.params import IGNORED_CLASS_ID


# def loss_initializer(y_true, y_pred):
#     labels_linear = tf.reshape(tensor=NUM_CLASSES, shape=[-1])
#     not_ignore_mask = tf.cast(tf.not_equal(labels_linear, IGNORED_CLASS_ID), dtype=tf.float32)
#     # The locations represented by indices in indices take value on_value, while all other locations take value off_value.
#     # For example, ignore label 255 in VOC2012 dataset will be set to zero vector in onehot encoding (looks like the not ignore mask is not required)
#     onehot_labels = tf.one_hot(indices=labels_linear, depth=NUM_CLASSES, on_value=1.0, off_value=0.0)
#
#     # sce = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
#     #                                        logits=tf.reshape(y_pred, shape=[-1, NUM_CLASSES]),
#     #                                        weights=not_ignore_mask)
#     sce = tf.keras.losses.SparseCategoricalCrossentropy()
#
#     return sce(y_true, y_pred)


def loss_initializer_v2(y_true, y_pred):
    mask = y_true != IGNORED_CLASS_ID
    sce = tf.keras.losses.SparseCategoricalCrossentropy()
    return sce(y_true, y_pred, sample_weight=mask)

# def custom_loss(y_true, y_pred):
#     # Create a loss function that removes loss for a specific class id
#     # y_pred[y_true == IGNORED_CLASS_ID] = IGNORED_CLASS_ID
#     mask = y_true == IGNORED_CLASS_ID
#     # tf.where(mask, x=y_pred, y=[IGNORED_CLASS_ID])
#     return tf.keras.losses.SparseCategoricalCrossentropy(y_true, y_pred)

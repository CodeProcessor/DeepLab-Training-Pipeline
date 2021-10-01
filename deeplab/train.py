#!/usr/bin/env python3
"""
@Filename:    train.py
@Author:      dulanj
@Time:        01/10/2021 23:45
"""
from deeplab.dataset import load_dataset
from deeplab.model import Deeplabv3
import tensorflow as tf
from deeplab.params import EPOCHS, LEARNING_RATE


def train():
    deeplab_model = Deeplabv3(weights=None)

    # Compile the model with loss and optimizer
    # TODO Implement loss function
    loss = None
    optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE)
    metrics = [tf.keras.metrics.MeanSquaredError()]
    deeplab_model.compile(optimizer, loss=loss, metrics=metrics)

    # TODO Load dataset
    data_generator = load_dataset()
    # TODO Callbacks
    callbacks = []

    for epoch_no in range(EPOCHS):
        deeplab_model.fit_generator(
            data_generator,
            epochs=1,
            verbose=1,
            callbacks=callbacks
        )

        if epoch_no % 10 == 0:
            # TODO Run on validation dataset
            ...

            # TODO Save the best model




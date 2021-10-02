#!/usr/bin/env python3
"""
@Filename:    train.py
@Author:      sgx team
@Time:        01/10/2021 23:45
"""
from deeplab.dataset import load_dataset
import tensorflow as tf
from deeplab.params import EPOCHS, LEARNING_RATE


def train(deeplab_model):


    # Loading the data generators
    train_data_gen, val_data_gen = load_dataset()
    # TODO Callbacks
    callbacks = []

    for epoch_no in range(EPOCHS):
        deeplab_model.fit(
            train_data_gen,
            epochs=1,
            verbose=1
        )

        if epoch_no % 10 == 0:
            # evaluate the model
            scores = deeplab_model.evaluate(val_data_gen, verbose=0)
            print("%s: %.2f%%" % (deeplab_model.metrics_names[1], scores[1]))

            # TODO Save the best model




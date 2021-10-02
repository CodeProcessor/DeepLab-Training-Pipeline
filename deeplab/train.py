#!/usr/bin/env python3
"""
@Filename:    train.py
@Author:      sgx team
@Time:        01/10/2021 23:45
"""

import os
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from deeplab.dataset import load_dataset
from deeplab.params import EPOCHS, CKPT_DIR, TENSORBOARD_DIR


def create_callbacks():
    lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-8)
    ckpt_callback = ModelCheckpoint(filepath=os.path.join(CKPT_DIR, 'depplabV3plus_{epoch:02d}-{loss:.2f}.h5'),
                                    monitor='val_loss', mode='min', save_best_only=True)
    tb_callback = TensorBoard(log_dir=TENSORBOARD_DIR),
    return [lr_callback, ckpt_callback, tb_callback]


def train(deeplab_model):
    # Loading the data generators
    train_data_gen, val_data_gen = load_dataset()
    # Creating callbacks
    callbacks = create_callbacks()

    deeplab_model.fit(
        train_data_gen,
        validation_data=val_data_gen,
        epochs=EPOCHS,
        verbose=1,
        validation_freq=10,
        callbacks=callbacks
    )


# def train(deeplab_model):
#     # Loading the data generators
#     train_data_gen, val_data_gen = load_dataset()
#     # TODO Callbacks
#     callbacks = []
#
#     for epoch_no in range(EPOCHS):
#         deeplab_model.fit(
#             train_data_gen,
#             epochs=1,
#             verbose=1,
#             callbacks=callbacks
#         )
#
#         if epoch_no % 10 == 0:
#             # evaluate the model
#             scores = deeplab_model.evaluate(val_data_gen, verbose=0)
#             print("%s: %.2f%%" % (deeplab_model.metrics_names[1], scores[1]))
#
#             # TODO Save the best model

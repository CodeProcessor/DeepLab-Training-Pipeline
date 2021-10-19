#!/usr/bin/env python3
"""
@Filename:    train.py
@Author:      sgx team
@Time:        01/10/2021 23:45
"""

import os
from datetime import datetime

from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard, EarlyStopping

from deeplab.dataset_voc import load_dataset
from deeplab.params import EPOCHS, CKPT_DIR, TENSORBOARD_DIR, VAL_FREQ, SAVE_BEST_ONLY

today = datetime.now()


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# learning rate schedule
def learning_rate_policy(epoch, lr, max_iteration=EPOCHS, power=0.9):
    return lr * (1 - epoch / max_iteration) ** power


def create_callbacks():
    lr_callback = LearningRateScheduler(learning_rate_policy)
    # lr_callback = ReduceLROnPlateau(monitor='loss', factor=0.7, patience=15, min_lr=1e-8)
    _datetime_name = "{}".format(today.strftime("%Y-%b-%d_%Hh-%Mm-%Ss"))
    _ckpt_dir = os.path.join(CKPT_DIR, _datetime_name)
    create_dir(_ckpt_dir)
    ckpt_callback = ModelCheckpoint(
        filepath=os.path.join(_ckpt_dir, 'depplabV3plus_epoch-{epoch:02d}_val-loss-{val_loss:.2f}.h5'),
        monitor='val_loss', mode='min', save_best_only=SAVE_BEST_ONLY
    )
    tb_callback = TensorBoard(log_dir=os.path.join(TENSORBOARD_DIR, _datetime_name))
    es_callback = EarlyStopping(patience=10)
    return [lr_callback, ckpt_callback, tb_callback]


def train(deeplab_model):
    # Loading the data generators
    train_data_gen, val_data_gen = load_dataset()
    # Creating callbacks
    callbacks = create_callbacks()
    # Train the model
    history = deeplab_model.fit(
        train_data_gen,
        validation_data=val_data_gen,
        epochs=EPOCHS,
        verbose=1,
        validation_freq=VAL_FREQ,
        callbacks=callbacks
    )
    # Return the training information
    return history

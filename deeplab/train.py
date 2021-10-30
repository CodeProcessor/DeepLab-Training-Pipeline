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
from deeplab.params import EPOCHS, CKPT_DIR, TENSORBOARD_DIR, VAL_FREQ, SAVE_BEST_ONLY, BACKBONE, IMAGE_SIZE, \
    LEARNING_RATE, AUG_PROBABILITY, UNIQUE_NAME

today = datetime.now()


def create_dir(path):
    """
    Create directory if not available
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def write_model_info(content=None):
    """
    Get the content append with the params and save as a file
    :param content:
    :return:
    """
    _info_dir = os.path.join(CKPT_DIR, UNIQUE_NAME)
    create_dir(_info_dir)
    if content is None:
        content = f"Backbone: {BACKBONE}\nLR: {LEARNING_RATE}\n" \
                  f"Resolution: {IMAGE_SIZE}\nAugmentations: {AUG_PROBABILITY}"

    with open(os.path.join(_info_dir, 'info.txt'), 'a') as fp:
        fp.write(content + '\n')


# learning rate schedule
def learning_rate_policy(epoch, lr, max_iteration=EPOCHS, power=0.9):
    """
    Learning rate scheduler
    :param epoch:
    :param lr:
    :param max_iteration:
    :param power:
    :return:
    """
    return lr * (1 - epoch / max_iteration) ** power


def create_callbacks():
    """
    Different callback functions
    LRschedule - to change the LR on the go
    ModelCheckpoint - To save the best model
    Tensorboard - Save logs for tensorboard analysis purpose
    Early stopping - Stopping early if its not converging further
    :return:
    """
    lr_callback = LearningRateScheduler(learning_rate_policy)
    # lr_callback = ReduceLROnPlateau(monitor='loss', factor=0.7, patience=15, min_lr=1e-8)
    _ckpt_dir = os.path.join(CKPT_DIR, UNIQUE_NAME)
    create_dir(_ckpt_dir)
    write_model_info()
    ckpt_callback = ModelCheckpoint(
        filepath=os.path.join(_ckpt_dir, 'depplabV3plus_epoch-{epoch:02d}_val-loss-{val_loss:.2f}.ckpt'),
        save_weights_only=True,
        monitor='val_loss', mode='min', save_best_only=SAVE_BEST_ONLY
    )
    tb_callback = TensorBoard(log_dir=os.path.join(TENSORBOARD_DIR, UNIQUE_NAME))
    es_callback = EarlyStopping(patience=10)
    return [lr_callback, ckpt_callback, tb_callback]


def train(deeplab_model):
    """
    The main training function
    Load the dataset and then create callbacks
    Call the fit method to train
    :param deeplab_model: TF model
    :return: history object
    """
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

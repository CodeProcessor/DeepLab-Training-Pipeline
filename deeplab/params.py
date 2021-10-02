#!/usr/bin/env python3
"""
@Filename:    params.py
@Author:      sgx team
@Time:        02/10/2021 00:03
"""

LEARNING_RATE = 1e-5
IMAGE_SIZE = (512, 512)

BATCH_SIZE = 2
NUM_CLASSES = 20
NUM_TRAIN_IMAGES = 25
NUM_VAL_IMAGES = 5

WEIGHT_DECAY = 0
EPOCHS = 250
LOAD_MODEL = False
LOAD_MODEL_FILE = 'my_checkpoint.h5'

DATA_DIR = "/home/shared/data/Training"
CKPT_DIR = "./ckpt"
TENSORBOARD_DIR = "./logs"
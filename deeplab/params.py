#!/usr/bin/env python3
"""
@Filename:    params.py
@Author:      sgx team
@Time:        02/10/2021 00:03
"""

PROD_SYS = True

LEARNING_RATE = 1e-5
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 4
NUM_CLASSES = 21  # use 21 for pascal voc else 20

WEIGHT_DECAY = 0
EPOCHS = 25 if PROD_SYS else 3
VAL_FREQ = 1
LOAD_MODEL = False

# Pascal images
NUM_TRAIN_IMAGES = 2200 if PROD_SYS else 100
NUM_VAL_IMAGES = 713 if PROD_SYS else 20

# CIHP images
# NUM_TRAIN_IMAGES = -1 if PROD_SYS else 1000
# NUM_VAL_IMAGES = -1 if PROD_SYS else 50

DATASET_DIR = "../VOC2012"
CKPT_DIR = "./output/ckpt"
TENSORBOARD_DIR = "./output/logs"
PRED_OUTPUT = "./output/pred"
LOAD_MODEL_FILE = 'output/ckpt/depplabV3plus_epoch-10_val-loss-2.91.h5'

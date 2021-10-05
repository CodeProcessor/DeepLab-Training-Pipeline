#!/usr/bin/env python3
"""
@Filename:    params.py
@Author:      sgx team
@Time:        02/10/2021 00:03
"""

PROD_SYS = False

LEARNING_RATE = 1e-5
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 4
NUM_CLASSES = 21

WEIGHT_DECAY = 0
EPOCHS = 10
VAL_FREQ = 1
LOAD_MODEL = False

NUM_TRAIN_IMAGES = -1 if PROD_SYS else 1000
NUM_VAL_IMAGES = -1 if PROD_SYS else 50

DATASET_DIR = "dataset"
CKPT_DIR = "./output/ckpt"
TENSORBOARD_DIR = "./output/logs"
PRED_OUTPUT = "./output/pred"
LOAD_MODEL_FILE = 'output/trained_models/depplabV3plus_epoch-10_val-loss-0.76.h5'

#!/usr/bin/env python3
"""
@Filename:    params.py
@Author:      sgx team
@Time:        02/10/2021 00:03
"""

PROD_SYS = False

LEARNING_RATE = 5e-6
IMAGE_SIZE = (320, 320)
BATCH_SIZE = 16 if PROD_SYS else 8
NUM_CLASSES = 21  # use 21 for pascal voc else 20

WEIGHT_DECAY = 0
EPOCHS = 500
VAL_FREQ = 1
LOAD_MODEL = False

# Pascal images
NUM_TRAIN_IMAGES = 5000 if PROD_SYS else 20
NUM_VAL_IMAGES = 5000 if PROD_SYS else 10
train_txt_file_voc = "dataset_voc/ImageSets/Segmentation/train.txt"
val_txt_file_voc = "dataset_voc/ImageSets/Segmentation/val.txt"

# CIHP images
# NUM_TRAIN_IMAGES = -1 if PROD_SYS else 1000
# NUM_VAL_IMAGES = -1 if PROD_SYS else 50

DATASET_DIR = "dataset_voc"
CKPT_DIR = "./output/ckpt"
TENSORBOARD_DIR = "./output/logs"
PRED_OUTPUT = "./output/pred"
LOAD_MODEL_FILE = 'output/ckpt/depplabV3plus_epoch-10_val-loss-2.91.h5'

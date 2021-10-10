#!/usr/bin/env python3
"""
@Filename:    params.py
@Author:      sgx team
@Time:        02/10/2021 00:03
"""

PROD_SYS = True

LEARNING_RATE = 5e-6
IMAGE_SIZE = (320, 320)
BATCH_SIZE = 16 if PROD_SYS else 8
NUM_CLASSES = 21  # use 21 for pascal voc else 20

USE_TF_RECORDS = True
WEIGHT_DECAY = 0
EPOCHS = 500
VAL_FREQ = 1
SAVE_BEST_ONLY = not PROD_SYS

# Model loading part
LOAD_MODEL = False
MODEL_PATH = 'ckpt/depplabV3plus_epoch-457_val-loss-1.50.h5'

# Augmentation
# set probability negative to disable
AUG_PROBABILITY = {
    "flip": 0.5,
    "rotate": -1,
    "trans": 0.5,
    "scale": 0.5
}

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

TF_RECORDS_DIR = '../tfrecord' if PROD_SYS else 'tfrecord'

train_txt_file_voc = DATASET_DIR + "/ImageSets/Segmentation/train.txt"
val_txt_file_voc = DATASET_DIR + "/ImageSets/Segmentation/val.txt"

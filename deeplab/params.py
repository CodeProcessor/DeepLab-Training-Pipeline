#!/usr/bin/env python3
"""
@Filename:    params.py
@Author:      sgx team
@Time:        02/10/2021 00:03
"""

PROD_SYS = True

LEARNING_RATE = 1e-5  # 0.007
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 8 if PROD_SYS else 4
NUM_CLASSES = 22  # use 21 for pascal voc else 20
IGNORED_CLASS_ID = 21

USE_TF_RECORDS = True
WEIGHT_DECAY = 0
BATCHNORM_DECAY = 0.9997
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
    "trans": 0.7,
    "scale": 0.7,
    "gaussian": 0.5,
    "hue": 0.2
}
# Buffer size to be used when shuffling with tf.data
SHUFFLE_BUFFER_SIZE = BATCH_SIZE * 10

# Pascal images
NUM_TRAIN_IMAGES = -1 if PROD_SYS else 20
NUM_VAL_IMAGES = -1 if PROD_SYS else 10
PREFETCH_LIMIT = 3000

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

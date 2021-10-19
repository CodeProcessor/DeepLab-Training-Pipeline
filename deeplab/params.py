#!/usr/bin/env python3
"""
@Filename:    params.py
@Author:      sgx team
@Time:        02/10/2021 00:03
"""
from datetime import datetime

today = datetime.now()

PROD_SYS = True

LEARNING_RATE = 1e-5
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 8 if PROD_SYS else 4
NUM_CLASSES = 22  # use 21 for pascal voc else 20
IGNORED_CLASS_ID = 21

BACKBONE = "resnet50"  # mobilenetv2, xception, resnet50
FREEZE_BACKBONE = True
INITIAL_WEIGHTS = True
USE_TF_RECORDS = True
WEIGHT_DECAY = 0
BATCHNORM_DECAY = 0.9997
EPOCHS = 20
VAL_FREQ = 1
SAVE_BEST_ONLY = not PROD_SYS
UNIQUE_NAME = "{}".format(today.strftime("%Y-%b-%d_%Hh-%Mm-%Ss"))

# Model loading part
LOAD_WEIGHTS_MODEL = True
MODEL_WEIGHTS_PATH = '/content/gdrive/MyDrive/ckpt/2021-Oct-19_14h-09m-00s/depplabV3plus_epoch-20_val-loss-4.52.ckpt'

# Augmentation
# set probability negative to disable
AUG_PROBABILITY = {
    "flip": 0.5,
    "rotate": -1,
    "trans": 0.7,
    "scale": 0.7,
    "gaussian": 0.5,
    "brightness": 1,
    "hue": 0.5
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
CKPT_DIR = "/content/gdrive/MyDrive/ckpt" if PROD_SYS else "./output/ckptv2"
TENSORBOARD_DIR = "/content/gdrive/MyDrive/tb_logs" if PROD_SYS else "./output/logs"
PRED_OUTPUT = "./output/pred"

TF_RECORDS_DIR = '../tfrecord' if PROD_SYS else 'tfrecord'

train_txt_file_voc = DATASET_DIR + "/ImageSets/Segmentation/train.txt"
val_txt_file_voc = DATASET_DIR + "/ImageSets/Segmentation/val.txt"

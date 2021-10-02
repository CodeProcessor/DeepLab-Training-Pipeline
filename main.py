#!/usr/bin/env python3
"""
@Filename:    main.py
@Author:      sgx team
@Time:        01/10/2021 23:32
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from deeplab.params import IMAGE_SIZE, NUM_CLASSES
from deeplab.train import train
from deeplab.model import DeeplabV3Plus, CompileModel

if __name__ == '__main__':
    deeplab_model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
    deeplab_model = CompileModel(deeplab_model)
    print(deeplab_model.summary())
    train(deeplab_model)

#!/usr/bin/env python3
"""
@Filename:    main.py
@Author:      dulanj
@Time:        01/10/2021 23:32
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from deeplab.model import Deeplabv3

if __name__ == '__main__':
    deeplab_model = Deeplabv3(weights=None)
    print(deeplab_model.summary())

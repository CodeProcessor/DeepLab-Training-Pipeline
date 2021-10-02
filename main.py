#!/usr/bin/env python3
"""
@Filename:    main.py
@Author:      sgx team
@Time:        01/10/2021 23:32
"""

from deeplab.graph_viz import get_graphs
from deeplab.model import DeeplabV3Plus, CompileModel
from deeplab.params import IMAGE_SIZE, NUM_CLASSES
from deeplab.train import train

if __name__ == '__main__':
    deeplab_model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
    deeplab_model = CompileModel(deeplab_model)
    print(deeplab_model.summary())
    history = train(deeplab_model)
    get_graphs(history)

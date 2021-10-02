#!/usr/bin/env python3
"""
@Filename:    main.py
@Author:      sgx team
@Time:        01/10/2021 23:32
"""
import os
from glob import glob

import cv2
from deeplab.graph_viz import get_graphs
from deeplab.inference import load_model
from deeplab.model import DeeplabV3Plus, CompileModel
from deeplab.overlay import plot_predictions
from deeplab.params import IMAGE_SIZE, NUM_CLASSES, LOAD_MODEL_FILE
from deeplab.train import train

if __name__ == '__main__':
    TRAIN = False
    if TRAIN:
        deeplab_model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
        deeplab_model = CompileModel(deeplab_model)
        print(deeplab_model.summary())
        history = train(deeplab_model)
        get_graphs(history)
    else:
        deeplab_model = load_model(LOAD_MODEL_FILE)
        print(deeplab_model.summary())
        image_list = glob("/home/dulanj/Datasets/CIHP/test/*")
        pred_list = plot_predictions(image_list, model=deeplab_model)
        for image_path, pred in zip(image_list, pred_list):
            image_name = os.path.basename(image_path)
            _, _, prediction_colormap = pred
            cv2.imwrite(image_name, prediction_colormap)
            print(f"Saved - {image_name}")

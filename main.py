#!/usr/bin/env python3
"""
@Filename:    main.py
@Author:      sgx team
@Time:        01/10/2021 23:32
"""
import argparse
import os
from glob import glob

import cv2
from deeplab.graph_viz import get_graphs
from deeplab.inference import load_model
from deeplab.model import DeeplabV3Plus, CompileModel
from deeplab.overlay import plot_predictions
from deeplab.params import IMAGE_SIZE, NUM_CLASSES, LOAD_MODEL_FILE, PRED_OUTPUT
from deeplab.train import train
from deeplab.custom_metrics import UpdatedMeanIoU


def main(is_train):
    if is_train:
        deeplab_model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
        deeplab_model = CompileModel(deeplab_model)
        print(deeplab_model.summary())
        history = train(deeplab_model)
        get_graphs(history)
    else:
        deeplab_model = load_model(LOAD_MODEL_FILE, custom_objects={'UpdatedMeanIoU': UpdatedMeanIoU})
        print(deeplab_model.summary())
        image_list = glob("dataset/Testing/Images/*")[:10]
        pred_list = plot_predictions(image_list, model=deeplab_model)
        if not os.path.exists(PRED_OUTPUT):
            os.makedirs(PRED_OUTPUT)
        for image_path, pred in zip(image_list, pred_list):
            output_image_path = os.path.join(PRED_OUTPUT, os.path.basename(image_path))
            _, _, prediction_colormap = pred
            cv2.imwrite(output_image_path, prediction_colormap)
            print(f"Saved - {output_image_path}")


def create_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='Enable if training', action='store_true')

    args = parser.parse_args()
    main(args.train)


if __name__ == '__main__':
    create_argparse()

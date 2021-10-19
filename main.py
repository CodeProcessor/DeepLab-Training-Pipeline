#!/usr/bin/env python3
"""
@Filename:    main.py
@Author:      sgx team
@Time:        01/10/2021 23:32
"""
import argparse
import os
import time
from glob import glob

from deeplab.graph_viz import get_graphs
from deeplab.model import DeeplabV3Plus, CompileModel, load_model
from deeplab.modelv2 import Deeplabv3
from deeplab.overlay import plot_predictions, save_cv_image
from deeplab.params import IMAGE_SIZE, NUM_CLASSES, MODEL_WEIGHTS_PATH, PRED_OUTPUT, LOAD_WEIGHTS_MODEL, BACKBONE, \
    INITIAL_WEIGHTS
from deeplab.train import train, write_model_info
from deeplab.utils import post_process


def main(is_train, content=""):
    write_model_info(content)
    if is_train:
        if not (BACKBONE in {'xception', 'mobilenetv2', 'resnet50'}):
            raise ValueError('The `backbone` argument should be either '
                             '`xception`, `resnet50` or `mobilenetv2` ')

        if BACKBONE == "resnet50":
            # Custom model
            if LOAD_WEIGHTS_MODEL:
                deeplab_model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
                print("Loading weights: {MODEL_WEIGHTS_PATH}")
                deeplab_model.load_weights(MODEL_WEIGHTS_PATH)
                print("Weights loaded!")
                time.sleep(0.5)
            else:
                deeplab_model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
        else:
            if LOAD_WEIGHTS_MODEL:
                deeplab_model = load_model(MODEL_WEIGHTS_PATH)
            else:
                _weights = "pascal_voc" if INITIAL_WEIGHTS else None
                deeplab_model = Deeplabv3(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), classes=NUM_CLASSES,
                                          backbone=BACKBONE, weights=_weights)

        deeplab_model = CompileModel(deeplab_model)

        print(deeplab_model.summary())
        history = train(deeplab_model)
        get_graphs(history)
    else:
        # deeplab_model = load_model(MODEL_PATH, custom_objects={'UpdatedMeanIoU': UpdatedMeanIoU})
        deeplab_model = Deeplabv3(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), classes=NUM_CLASSES)
        print(deeplab_model.summary())
        image_list = glob("dataset/Testing/Images/*")[:10]
        pred_list = plot_predictions(image_list, model=deeplab_model)
        if not os.path.exists(PRED_OUTPUT):
            os.makedirs(PRED_OUTPUT)
        for image_path, pred in zip(image_list, pred_list):
            im, overlay, prediction_colormap = pred
            save_folder = os.path.join(PRED_OUTPUT, os.path.basename(image_path).split('.')[0])
            os.makedirs(save_folder, exist_ok=True)
            save_cv_image(os.path.join(save_folder, 'mask_' + os.path.basename(image_path)), prediction_colormap)
            save_cv_image(os.path.join(save_folder, 'overlay_' + os.path.basename(image_path)), overlay)
            save_cv_image(os.path.join(save_folder, 'image_' + os.path.basename(image_path)), post_process(im))
            # save_cv_image(os.path.join(save_folder, 'image_' + os.path.basename(image_path)), (im + 1) * 127.5)
            print(f"Saved results to - {save_folder}")


def create_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='Enable if training', action='store_true')
    parser.add_argument('--info', help='Information about the model', type=str, default="")

    args = parser.parse_args()
    main(args.train, args.info)


if __name__ == '__main__':
    create_argparse()

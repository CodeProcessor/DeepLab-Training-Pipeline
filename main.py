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
from deeplab.model import DeeplabV3Plus, CompileModel
from deeplab.modelv2 import Deeplabv3
from deeplab.overlay import plot_predictions, save_cv_image
from deeplab.params import IMAGE_SIZE, NUM_CLASSES, MODEL_WEIGHTS_PATH, PRED_OUTPUT, LOAD_WEIGHTS_MODEL, BACKBONE, \
    INITIAL_WEIGHTS, FREEZE_BACKBONE
from deeplab.train import train, write_model_info
from deeplab.utils import post_process


def main(is_train, content=""):
    """
    Write the content to a text file in the model saving directory
    Then we wont loose the track of what we have done as experiments
    """
    write_model_info(content)
    # Select one of the predefined newworks
    if not (BACKBONE in {'xception', 'mobilenetv2', 'resnet50'}):
        raise ValueError('The `backbone` argument should be either '
                         '`xception`, `resnet50` or `mobilenetv2` ')
    print(f"Loading Backbone: {BACKBONE}")
    time.sleep(0.5)
    if BACKBONE == "resnet50":
        """
        Custom model which was taken from Keras team
        Link - https://keras.io/examples/vision/deeplabv3_plus/
        """
        deeplab_model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES,
                                      freeze_backbone=FREEZE_BACKBONE)
    else:
        """
        This implementation was taken from another repository
        Link - https://github.com/bonlime/keras-deeplab-v3-plus
        """
        _weights = "pascal_voc" if INITIAL_WEIGHTS else None
        deeplab_model = Deeplabv3(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), classes=NUM_CLASSES,
                                  backbone=BACKBONE, weights=_weights)
    if LOAD_WEIGHTS_MODEL:
        """
        Loading weights if mentioned in params
        """
        print(f"Loading weights: {MODEL_WEIGHTS_PATH}")
        deeplab_model.load_weights(MODEL_WEIGHTS_PATH)
        print("Weights loaded!")
        time.sleep(0.5)

    if is_train:
        """
        If the model is to train then
        + Compile the model and call training function
        + get the information
        + plot the graphs
        """
        deeplab_model = CompileModel(deeplab_model)
        history = train(deeplab_model)
        get_graphs(history)
    else:
        """
        If the model is for inference 
        Load the images and then run one by one plot the predictions and save
        """
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
    """
    Arg parser for commandline inputs
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='Enable if training', action='store_true')
    parser.add_argument('--info', help='Information about the model', type=str, default="")

    args = parser.parse_args()
    main(args.train, args.info)


if __name__ == '__main__':
    create_argparse()

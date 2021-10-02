# DeepLab-Training-Pipeline

This repo is based on the Keras code available in here - https://keras.io/examples/vision/deeplabv3_plus/

## Building the DeepLabV3+ model

DeepLabv3+ extends DeepLabv3 by adding an encoder-decoder structure. The encoder module processes multiscale contextual
information by applying dilated convolution at multiple scales, while the decoder module refines the segmentation
results along object boundaries.

![image missing](assets/deeplabv3_plus_diagram.png "DeepLabV3 Diagram")

## Download the dataset

This is trained using Crowd Instance-level Human Parsing Dataset

Use this link to download - https://drive.google.com/uc?id=1B9A9UCJYMwTL4oBEo4RZfbMZMaZhKJaz

Set the dataset location in params.py

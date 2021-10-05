#!/usr/bin/env bash

IMAGE_NAME=deeplab:v2.1
DATASET_LOC=/home/dulanj/Datasets/PASCAL_VOC/2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012

docker run --gpus "all" -it --rm -p 80:80 \
-v $DATASET_LOC:/home/dataset_voc:ro \
-v $(pwd)/output:/home/output \
$IMAGE_NAME

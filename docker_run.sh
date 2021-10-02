#!/usr/bin/env bash

IMAGE_NAME=deeplab:v2.0

docker run --gpus "all" -it --rm -p 80:80 \
-v /home/dulanj/Datasets/CIHP/instance-level_human_parsing/instance-level_human_parsing:/home/dataset:ro \
-v $(pwd)/output:/home/output \
$IMAGE_NAME

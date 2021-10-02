#!/usr/bin/env bash

IMAGE_NAME=deeplab:v2.0

docker run --gpus "all" -it --rm -p 80:80 \
-v /home/dulanj/Datasets/CIHP/instance-level_human_parsing/instance-level_human_parsing:/home/shared/data:ro \
-v $(pwd)/docker_output:/home/shared/output \
$IMAGE_NAME

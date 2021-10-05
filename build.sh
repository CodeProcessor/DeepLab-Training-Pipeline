#!/usr/bin/env bash

IMAGE_NAME=deeplab:v2.1

# Download model if not available
RESNET_TOP=models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
if [ -f "$RESNET_TOP" ]; then
    echo "$RESNET_TOP exists."
else
    echo "$RESNET_TOP does not exist. Downloading..."
    (
    cd models || exit
    bash download_models.sh
    )
fi

# Building the docker image
docker build -t $IMAGE_NAME .

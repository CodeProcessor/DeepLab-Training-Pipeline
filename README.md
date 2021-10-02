# DeepLab Training Pipeline

This repo is based on the Keras code available in here - https://keras.io/examples/vision/deeplabv3_plus/

## Building the DeepLabV3+ model

DeepLabv3+ extends DeepLabv3 by adding an encoder-decoder structure. The encoder module processes multiscale contextual
information by applying dilated convolution at multiple scales, while the decoder module refines the segmentation
results along object boundaries.

![image missing](assets/deeplabv3_plus_diagram.png "DeepLabV3 Diagram")

## Installation

### Install docker

Find the installation files under scripts directory

```bash
bash install_docker_ce.sh
```

### Install nvidia-docker runtime

```bash
bash install_nvidia-runtime.sh
```

### Download the dataset

This is trained using Crowd Instance-level Human Parsing Dataset

Link - https://drive.google.com/uc?id=1B9A9UCJYMwTL4oBEo4RZfbMZMaZhKJaz

### Extract the dataset and keepunder dataset-dir

```bash
unzip instance-level-human-parsing.zip
mv instance-level_human_parsing <dataset-dir>
```

### Build the docker

```bash
bash build.sh
```

### Set the path to the dataset location in docker_run.sh script

```bash
DATASET_LOC=<dataset-dir>/instance-level_human_parsing/instance-level_human_parsing
```

### Create container with the shared dataset

```bash
bash docker_run.sh
```

### Run the training

```bash
bash train.sh
```

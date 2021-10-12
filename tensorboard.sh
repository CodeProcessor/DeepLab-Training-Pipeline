#!/bin/bash
if lsof -Pi :6006 -sTCP:LISTEN -t >/dev/null ; then
    echo "Tensorboard already running"
else
    echo "Starting tensorboard..."
    tensorboard --logdir=output --bind_all
fi

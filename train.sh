#!/usr/bin/env python

bash tensorboard.sh &
nohup python3 main.py --train &

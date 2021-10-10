#!/usr/bin/env python

bash tensorboard.sh &
nohup python3 main.py --train &
tail -f nohup.out
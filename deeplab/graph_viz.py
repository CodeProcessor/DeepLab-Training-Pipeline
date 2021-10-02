#!/usr/bin/env python3
"""
@Filename:    graph_viz.py
@Author:      dulanj
@Time:        02/10/2021 17:40
"""
from matplotlib import pyplot as plt


def get_graphs(history):
    for _keys in history.history.keys():
        plt.plot(history.history[_keys])
        plt.title("{} {}".format("Validation" if "val" in _keys else "Training", _keys))
        plt.ylabel(f"{_keys}")
        plt.xlabel("epoch")
        plt.savefig(f'output/{_keys}.png')
        plt.clf()

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np


class softmax_layer(object):
    """docstring for softmax_layer"""

    def __init__(self, layers=0):
        self.delta_mat = np.zeros((layers))
        self.layers = layers
        pass

    def forward(self, input_array):
        self.input_array = input_array
        self.out_put = self.softmax(np.exp(self.input_array))
        return self.out_put

    def softmax(self, x):
        _sum = np.sum(x)
        return x / _sum

    def backward(self, targets):
        self.delta_mat = self.out_put - targets
        return self.delta_mat

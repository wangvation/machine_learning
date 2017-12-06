#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np


class softmax_layer(object):
    """docstring for softmax_layer"""

    def __init__(self, input_array, layers=0):
        self.input_array = input_array
        self.sensitivity_map = np.zeros((input_array.shape))
        self.layers = layers
        self.out_put = None
        pass

    def forward(self):
        self.out_put = self.softmax(np.exp(self.input_array))

    def softmax(self, x):
        _sum = np.sum(x)
        return x / _sum

    def backward(self, targets):
        j = np.argmax(targets)
        for i in range(self.layers[1]):
            self.sensitivity_map[i] = self.out_put[i] - \
                1 if i == j else self.out_put[i]

    def cross_entropy(self, y, targets):
        return -np.sum(np.multiply(targets, np.log(y)))

    def get_sensitivity_map(self):
        return self.sensitivity_map

    def out_put(self):
        return self.out_put

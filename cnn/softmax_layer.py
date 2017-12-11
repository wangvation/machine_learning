#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from cnn_utils import *


class softmax_layer(object):
    """docstring for softmax_layer"""

    def __init__(self, output_nums=0):
        self.output_nums = output_nums

    def forward(self, input_array):
        debug('softmax_layer:\n', input_array)
        self.input_array = input_array
        self.out_put = self.softmax(self.input_array)
        return self.out_put

    def softmax(self, x):
        x = np.exp(self.input_array)
        _sum = np.sum(x)
        return x / _sum

    def backward(self, targets):
        self.delta_map = self.out_put - targets
        debug('softmax--backward:', np.sum(self.delta_map))
        return self.delta_map

    def update(self, alpha, batch_size):
        """
        Don't do anything.
        Args:
          batch_size:batch size
          alpha: learning rate

        Returns:

        """
        pass

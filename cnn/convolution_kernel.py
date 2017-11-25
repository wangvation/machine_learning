#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np


class kernel(object):
    """docstring for kernel"""

    def __init__(self, height, width, depth=None, stride=1):
        self.depth = depth
        self.height = height
        self.width = width
        self.stride = stride

    def shape(self):
        return self.width, self.height, self.depth

    @classmethod
    def obtin_kernels(cls, height, width, depth=None,
                      stride=1, kernel_num=None):
        if kernel_num is None:
            return kernel(height, width, depth, stride)
        return [kernel(height, width, depth, stride) for _ in range(kernel_num)]


class conv_kernel(kernel):
    """docstring for conv_kernel"""

    def __init__(self, height, width, depth=None, stride=1):
        super(conv_kernel, self).__init__(height, width, depth, stride)
        if depth is not None:
            self.weights = np.random.uniform(-1e-4,
                                             1e-4, (height, width, depth))
        else:
            self.weights = np.random.uniform(-1e-4,
                                             1e-4, (height, width))
        self.bias = 0.0
        self.weights_grad = np.zeros(self.weights.shape)
        self.bias_grad = 0

    def update(self, learning_rate):
        self.weights += learning_rate * self.weights_grad
        self.bias += learning_rate * self.bias_grad
        self.weights_grad[:] = 0
        self.bias_grad = 0

    @classmethod
    def obtin_kernels(cls, height, width, depth=None,
                      stride=1, kernel_num=None):
        if kernel_num is None:
            return conv_kernel(height, width, depth, stride)
        return [conv_kernel(height, width, depth, stride)
                for _ in range(kernel_num)]

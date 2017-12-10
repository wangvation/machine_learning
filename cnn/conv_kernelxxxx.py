#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from pooling_kernel import pooling_kernel
from cnn_utils import *


class conv_kernel(pooling_kernel):
    """docstring for conv_kernel"""

    def __init__(self, kernel_shape, weights=None, bias=0.0, stride=1):
        super(conv_kernel, self).__init__(kernel_shape, stride)
        if weights is None:
            depth, height, width = expand_shape(kernel_shape, 1)
            std = 1.0 / np.sqrt(depth * height * width)
            self.weights = np.random.normal(loc=0.0,
                                            scale=std,
                                            size=kernel_shape)
        else:
            self.weights = weights
        self.shape = (self.weights.shape if kernel_shape !=
                      self.weights.shape else kernel_shape)
        self.bias = bias
        self.weights_grad = np.zeros(self.weights.shape)
        self.bias_grad = 0

    def update(self, learning_rate, batch_size):
        """

        Args:
          batch_size:batch size
          learning_rate: learning rate
        Returns:

        """
        self.weights += learning_rate * self.weights_grad / batch_size
        self.bias += learning_rate * self.bias_grad / batch_size
        self.weights_grad[:] = 0
        self.bias_grad = 0

    def turn_round(self):
        """ """
        if self.weights.ndim == 1:
            self.weights = np.flip(self.weights, axis=0)
        if self.weights.ndim == 2:
            tmp = np.flip(self.weights, axis=0)
            self.weights = np.flip(tmp, axis=1)
        if self.weights.ndim == 3:
            tmp = np.flip(self.weights, axis=1)
            self.weights = np.flip(tmp, axis=2)

    def expands_2d(self):
        """Expansion of 3D convolution kernels into 2D convolution kernels"""
        if self.weights.ndim == 3:
            depth, height, width = self.shape
            return [conv_kernel((height, width), self.weights[d, :, :],
                                self.bias, self.stride) for d in range(depth)]
        return [self]

    @classmethod
    def obtin_kernels(cls, kernel_num, kernel_shape,
                      weights=None, bias=0.0, stride=1):
        """

        Args:
          kernel_shape:
          weights:  (Default value = None)
          bias:  (Default value = 0.0)
          stride:  (Default value = 1)
          kernel_num: kernel num (greater than 1 )

        Returns:

        """
        if kernel_num is None or kernel_num <= 1:
            raise ValueError('The kernel_num must be greater than 1.')
        return [conv_kernel(kernel_shape, weights, bias, stride)
                for k in range(kernel_num)]

    @classmethod
    def calc_feature_shape(cls, input_shape, kernel_shape, kernel_num,
                           padding=0, stride=1):
        """

        Args:
          input_shape:
          kernel_shape:
          kernel_num:
          padding:  (Default value = 0)
          stride:  (Default value = 1)

        Returns:

        """
        if kernel_num == 0:
            raise ValueError('The length of the kernels must be more than zero')
        input_depth, input_height, input_width = expand_shape(input_shape)
        kernel_depth, kernel_height, kernel_width = expand_shape(kernel_shape)
        out_width = (input_width - kernel_width + 2 *
                     padding) // stride + 1
        out_height = (input_height - kernel_height + 2 *
                      padding) // stride + 1
        if input_depth != kernel_depth:
            raise ValueError('The depth of the input_shape must be equal to '
                             'the depth of the convolution_shape')
        if out_width <= 0 or out_height <= 0:
            return None
        if kernel_num == 1:
            return out_height, out_width
        return kernel_num, out_height, out_width

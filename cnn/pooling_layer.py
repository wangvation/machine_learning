#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from pooling_kernel import pooling_kernel
from cnn_utils import *


class pooling_layer(object):
    """docstring for pooling_layer"""

    def __init__(self, input_shape, kernel_shape,
                 pooling_type='max_pooling', stride=1):
        '''
        Parameters
        ----------
        input_shape:the shape of input_array
        kernel_shape:the shape of pool_kernel
        pooling_type:{'max_pooling', or 'mean_pooling'}
        '''
        self.input_shape = input_shape
        self.pool_kernel = pooling_kernel(kernel_shape, stride=stride)
        self.delta_map = np.zeros(input_shape, dtype=np.float32)
        self.feature_shape = pooling_kernel.calc_feature_shape(
            input_shape=input_shape, kernel_shape=kernel_shape,
            padding=0, stride=stride)
        self.pooling_type = pooling_type
        if self.pooling_type == 'max_pooling':
            self.pooling = self.max_pooling
        elif self.pooling_type == 'mean_pooling':
            self.pooling = self.mean_pooling

    def forward(self, input_array):
        '''
        Parameters
        ----------
        input_array:input array
        '''
        debug('pooling_layer:\n', input_array)
        self.input_array = input_array
        return self.pooling()

    def max_pooling(self):
        if self.feature_shape is None:
            return
        feature_map = np.zeros(self.feature_shape, dtype=np.float32)
        depth, height, width = expand_shape(self.feature_shape)
        for i in range(height):
            for j in range(width):
                patch = get_patch(i, j, self.input_array, self.pool_kernel)
                if depth is None:
                    feature_map[i, j] = np.max(patch)
                    continue
                for d in range(depth):
                    feature_map[d, i, j] = np.max(patch[d, ...])
        return feature_map

    def mean_pooling(self):
        if self.feature_shape is None:
            return
        feature_map = np.zeros(self.feature_shape, dtype=np.float32)
        depth, height, width = expand_shape(self.feature_shape)
        for i in range(height):
            for j in range(width):
                if depth is None:
                    patch = get_patch(i, j, self.input_array, self.pool_kernel)
                    feature_map[i, j] = np.mean(patch)
                    continue
                for d in range(depth):
                    feature_map[d, i, j] = np.mean(patch[d, ...])
        return feature_map

    def backward(self, delta_map):
        """

        Args:
          delta_map:

        Returns:

        """
        debug('pooling_layer :', np.sum(delta_map), self.input_shape)
        depth, height, width = expand_shape(delta_map.shape)
        if self.pooling_type == 'mean_pooling':
            for i in range(height):
                for j in range(width):
                    delta_patch = get_patch(i, j, self.delta_map,
                                            self.pool_kernel)
                    if depth is None:
                        delta_patch[...] = (delta_map[i, j] /
                                            (self.width * self.height))
                        continue
                    for d in range(depth):
                        delta_patch[d, ...] = (delta_map[d:, i, j] /
                                               (self.width * self.height))
        if self.pooling_type == 'max_pooling':
            for i in range(height):
                for j in range(width):
                    delta_patch = get_patch(i, j, self.delta_map,
                                            self.pool_kernel)
                    input_patch = get_patch(i, j, self.input_array,
                                            self.pool_kernel)
                    if depth is None:
                        argmax = np.argmax(input_patch)
                        max_i, max_j = np.unravel_index(np.argmax(argmax),
                                                        input_patch.shape)
                        delta_patch[max_i, max_j] = delta_map[i, j]
                        continue
                    for d in range(depth):
                        argmax = np.argmax(input_patch[d, ...])
                        max_i, max_j = np.unravel_index(
                            np.argmax(argmax), input_patch[d, ...].shape)
                        delta_patch[d, max_i, max_j] = delta_map[d, i, j]

        return self.delta_map

    def update(self, alpha, batch_size):
        """
        Clear the delta_map.
        Args:
          batch_size:batch size
          alpha: learning rate

        Returns:

        """
        self.delta_map[...] = 0.0
        pass

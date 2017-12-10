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
        self.kernel_shape = kernel_shape
        self.pool_kernel = pooling_kernel(self.kernel_shape, stride=stride)
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
        self.input_array = input_array
        return self.pooling()

    def max_pooling(self):
        if self.feature_shape is None:
            return
        feature_map = np.zeros(self.feature_shape, dtype=np.float32)
        if feature_map.ndim == 3:
            feature_depth, feature_height, feature_width = self.feature_shape
        else:
            feature_depth = None
            feature_height, feature_width = self.feature_shape
        if feature_depth is None:
            for i in range(feature_height):
                for j in range(feature_width):
                    patch = self.get_patch(i, j, self.input_array,
                                           self.pool_kernel.shape,
                                           self.pool_kernel.stride)
                    feature_map[i, j] = np.max(patch)
        else:
            for i in range(feature_height):
                for j in range(feature_width):
                    patch = get_patch(i, j, self.input_array,
                                      self.pool_kernel.shape,
                                      self.pool_kernel.stride)
                    for d in range(feature_depth):
                        feature_map[d, i, j] = np.max(patch[d, :, :])
        return feature_map

    def mean_pooling(self):
        if self.feature_shape is None:
            return
        feature_map = np.zeros(self.feature_shape, dtype=np.float32)
        if feature_map.ndim == 3:
            feature_depth, feature_height, feature_width = self.feature_shape
        else:
            feature_depth = None
            feature_height, feature_width = self.feature_shape
        if feature_depth is None:
            for i in range(feature_height):
                for j in range(feature_width):
                    patch = self.get_patch(i, j, self.input_array,
                                           self.pool_kernel.shape,
                                           self.pool_kernel.stride)
                    feature_map[i, j] = np.mean(patch)
        else:
            for i in range(feature_height):
                for j in range(feature_width):
                    patch = self.get_patch(i, j, self.input_array,
                                           self.pool_kernel.shape,
                                           self.pool_kernel.stride)
                    for d in range(feature_depth):
                        feature_map[d, i, j] = np.mean(patch[d, :, :])
        return feature_map

    def backward(self, delta_map):
        """

        Args:
          delta_map:

        Returns:

        """
        height, width = delta_map.shape
        if self.pooling_type == 'mean_pooling':
            for i in range(height):
                for j in range(width):
                    patch = self.get_patch(self.delta_map, i, j)
                    patch[:] = (delta_map[i, j] /
                                (self.width * self.height))
        if self.pooling_type == 'max_pooling':
            for i in range(height):
                for j in range(width):
                    patch = self.get_patch(self.delta_map, i, j)
                    input_patch = get_patch(self.input_array, i, j)
                    max_i, max_j = np.argmax(input_patch, axis=None, out=None)
                    patch[max_i, max_j] = delta_map[i, j]
        return self.delta_map

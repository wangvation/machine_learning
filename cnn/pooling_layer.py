#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from convolution_kernel import kernel


class pooling_layer(object):
    """docstring for pooling_layer"""

    def __init__(self, input_array, pooling_type='max_pooling', pool_kernel):
        '''
        Parameters
        ----------
        input_array:input_array
        kernel:input kernel
        pooling_type:{'max_pooling', or 'mean_pooling'}
        '''
        self.pooling_type = pooling_type
        self.input_array = input_array
        self.pool_kernel = pool_kernel
        self.shape = input_array.shape
        self.sensitivity_map = np.zeros(self.shape, dtype=np.float32)
        pass

    def pooling(self):
        """ """
        feature_shape = kernel.calc_feature_shape(
            input_shape=self.input_array.shape,
            zero_padding=0, kernels=[self.pool_kernel])
        if feature_shape is None:
            return
        feature_map = np.zeros(feature_shape, dtype=np.float32)
        if feature_map.ndim == 3:
            feature_depth, feature_height, feature_width = feature_shape
        else:
            feature_depth = None
            feature_height, feature_width = feature_shape

        if self.pooling_type == 'max_pooling':
            for i in range(feature_height):
                for j in range(feature_width):
                    patch = self.get_patch(i, j)
                    if feature_depth is None:
                        feature_map[i, j] = np.max(patch)
                    else:
                        for d in range(feature_depth):
                            feature_map[d, i, j] = np.max(patch[d, :, :])
            pass
        if self.pooling_type == 'mean_pooling':
            for i in range(feature_height):
                for j in range(feature_width):
                    patch = self.get_patch(i, j)
                    if feature_depth is None:
                        feature_map[i, j] = np.mean(patch)
                    else:
                        for d in range(feature_depth):
                            feature_map[i, j, d] = np.mean(patch[:, :, d])
            pass

    def backward(self, sensitivity_map):
        """

        Args:
          sensitivity_map:

        Returns:

        """
        height, width = sensitivity_map.shape
        if self.pooling_type == 'mean_pooling':
            for i in range(height):
                for j in range(width):
                    patch = self.get_patch(self.sensitivity_map, i, j)
                    patch[:] = (sensitivity_map[i, j] /
                                (self.width * self.height))
        if self.pooling_type == 'max_pooling':
            for i in range(height):
                for j in range(width):
                    patch = self.get_patch(self.sensitivity_map, i, j)
                    input_patch = self.get_patch(self.input_array, i, j)
                    max_i, max_j = np.argmax(input_patch, axis=None, out=None)
                    patch[max_i, max_j] = sensitivity_map[i, j]

    def get_sensitivity_map(self):
        """return a sensitivity map"""
        return self.sensitivity_map

    def get_patch(self, array, i, j):
        """

        Args:
          array:
          i:
          j:

        Returns:

        """
        start_i = i * self.pool_kernel.stride
        start_j = j * self.pool_kernel.stride
        kernel_depth, kernel_height, kernel_width = self.pool_kernel.shape()
        if kernel_depth is None:
            return array[start_i:start_i + kernel_height,
                         start_j:start_j + kernel_width]
        return array[:, start_i:start_i + kernel_height,
                     start_j:start_j + kernel_width]

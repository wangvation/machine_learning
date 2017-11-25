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
        pass

    def calc_feature_shape(self):
        '''
        Return a shape of feature map
        Returns
        -------
        withd:feature map withd
        height:feature map height
        depth:feature map depth, if feature map is 2D,depth is None
        '''
        if len(self.shape) == 2:
            input_width, input_height = self.shape
            input_depth = None
        elif len(self.shape) == 3:
            input_width, input_height, input_depth = self.shape
        else:
            raise ValueError('the length of the input_shape must be 2 or 3')
        out_width = (input_width - self.pool_kernel[0].width + 2 *
                     self.padding) / self.pool_kernel[0].stride + 1
        out_height = (input_height - self.pool_kernel[0].height + 2 *
                      self.padding) / self.pool_kernel[0].stride + 1
        if (input_depth != self.pool_kernel[0].depth or
                out_width <= 0 or out_height <= 0):
            return None
        return out_width, out_height, input_depth

    def pooling(self):
        feature_shape = kernel.calc_feature_shape(
            input_shape=self.input_array.shape,
            zero_padding=0, kernels=[self.pool_kernel])
        if feature_shape is None:
            return
        feature_map = np.zeros(feature_shape, dtype=np.float32)
        feature_width, feature_height, feature_depth = feature_shape
        if self.pooling_type == 'max_pooling':
            for i in range(feature_width):
                for j in range(feature_height):
                    patch = self.get_patch(i, j)
                    if feature_depth is None:
                        feature_map[i, j] = np.max(patch)
                    else:
                        for d in range(feature_depth):
                            feature_map[i, j, d] = np.max(patch[:, :, d])
            pass
        if self.pooling_type == 'mean_pooling':
            for i in range(feature_width):
                for j in range(feature_height):
                    patch = self.get_patch(i, j)
                    if feature_depth is None:
                        feature_map[i, j] = np.mean(patch)
                    else:
                        for d in range(feature_depth):
                            feature_map[i, j, d] = np.mean(patch[:, :, d])
            pass

    def get_patch(self, i, j):
        start_i = i * self.pool_kernel.stride
        start_j = j * self.pool_kernel.stride
        kernel_width, kernel_height, kernel_depth = self.pool_kernel.shape()
        if kernel_depth is None:
            return self.input_array[start_i:start_i + kernel_width,
                                    start_j:start_j + kernel_height]
        return self.input_array[start_i:start_i + kernel_width,
                                start_j:start_j + kernel_height, :]

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from cnn_utils import *


class pooling_layer(object):
    """docstring for pooling_layer"""

    def __init__(self, input_shape, kernel_shape,
                 pooling_type='max_pooling', stride=1):
        '''
        Parameters
        ----------
        input_shape:the shape of input
        kernel_shape:the shape of pool_kernel
        pooling_type:{'max_pooling', or 'mean_pooling'}
        '''
        self.input_shape = input_shape
        self.kernel_shape = kernel_shape
        self.delta_map = np.zeros(input_shape, dtype=np.float32)
        self.feature_shape = self.calc_feature_shape(
            input_shape=input_shape, kernel_shape=kernel_shape,
            padding=0, stride=stride)
        self.stride = stride
        self.pooling_type = pooling_type
        if self.pooling_type == 'max_pooling':
            self.pooling = self.max_pooling
        elif self.pooling_type == 'mean_pooling':
            self.pooling = self.mean_pooling

    def forward(self, input):
        '''
        Parameters
        ----------
        input:input array
        '''
        debug(True, 'pooling_layer input:', np.max(input),
              np.min(input), np.mean(input))
        # self.delta_map[...] = 0.
        self.input = input
        return self.pooling()

    def max_pooling(self):
        if self.feature_shape is None:
            return
        feature_map = np.zeros(self.feature_shape, dtype=np.float32)
        depth, height, width = expand_shape(self.feature_shape)
        for i in range(height):
            for j in range(width):
                patch = get_patch(i, j, self.input,
                                  self.kernel_shape, self.stride)
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
                    patch = get_patch(i, j, self.input,
                                      self.kernel_shape, self.stride)
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
        debug(True, 'pooling_layer-backward :', self.input_shape,
              np.max(delta_map), np.min(delta_map), np.mean(delta_map))
        depth, height, width = expand_shape(delta_map.shape)
        if self.pooling_type == 'mean_pooling':
            for i in range(height):
                for j in range(width):
                    delta_patch = get_patch(i, j, self.delta_map,
                                            self.kernel_shape, self.stride)
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
                                            self.kernel_shape, self.stride)
                    input_patch = get_patch(i, j, self.input,
                                            self.kernel_shape, self.stride)
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
        do nothing.
        Args:
          batch_size:batch size
          alpha: learning rate

        Returns:

        """
        pass

    def calc_feature_shape(self, input_shape, kernel_shape,
                           padding=0, stride=1):
        """

        Args:
          input_shape:
          kernel_shape:
          padding:  (Default value = 0)
          stride:  (Default value = 1)

        Returns:


        """
        if len(input_shape) == 2:
            input_height, input_width = input_shape
            kernel_height, kernel_width = kernel_shape
            kernel_depth = None
            input_depth = None
        elif len(input_shape) == 3:
            input_depth, input_height, input_width = input_shape
            kernel_depth, kernel_height, kernel_width = kernel_shape
        else:
            raise ValueError('the length of the input_shape must be 2 or 3')
        out_width = (input_width - kernel_width + 2 *
                     padding) // stride + 1
        out_height = (input_height - kernel_height + 2 *
                      padding) // stride + 1
        if (input_depth != kernel_depth):
            raise ValueError('The depth of the input_shape must be equal to '
                             'the depth of the convolution_shape')
        if(out_width <= 0 or out_height <= 0):
            return None
        if input_depth is None:
            return out_height, out_width
        return input_depth, out_height, out_width

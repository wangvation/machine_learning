#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from convolution_kernel import conv_kernel
from cnn_utils import *


class convolution_layer(object):
    '''docstring for convolution_layer'''

    def __init__(self, active, active_derive, input_array,
                 zero_padding=0, kernels=[]):
        self.active = active
        self.active_derive = active_derive
        self.kernels = kernels
        self.padding = zero_padding
        self.input_array = input_array
        self.shape = self.input_array.shape
        self.feature_map = None
        pass

    def forward(self):
        conv_map = self.convolution(
            self.input_array, self.padding, *self.kernels)
        self.feature_map = self.active(conv_map)

    def convolution(self, array, padding, *kernels):
        array = around_with_zero(input_array=array,
                                 width_padding=padding,
                                 height_padding=padding)
        feature_shape = conv_kernel.calc_feature_shape(
            input_shape=array.shape, kernel_shape=kernels[0].shape,
            kernel_num=len(kernels), padding=padding,
            stride=kernels[0].stride)
        if feature_shape is None:
            return
        conv_map = np.zeros(feature_shape, dtype=np.float32)
        if conv_map.ndim == 3:
            out_depth, out_height, out_width = feature_shape
        else:
            out_depth = None
            out_height, out_width = feature_shape
        for i in range(out_height):
            for j in range(out_width):
                k = kernels[0]
                patch = get_patch(i, j, array, k)
                if out_depth is None:
                    conv_map[i, j] = np.sum(patch * k.weights) + k.bias
                else:
                    for d in range(out_depth):
                        k = kernels[d]
                        conv_map[d, i, j] = np.sum(patch * k.weights) + k.bias
        return conv_map

    def get_output(self):
        return self.feature_map

    def extend_to_one_stride(self, kernel_shape, old_stride, sensitivity_map):
        if sensitivity_map.ndim == 3:
            old_depth, old_height, old_width = np.shape(sensitivity_map)
        else:
            old_height, old_width = np.shape(sensitivity_map)
            old_depth = None
        new_map_shape = conv_kernel.calc_feature_shape(
            input_shape=self.shape, kernel_shape=kernel_shape,
            kernel_num=old_depth, padding=self.padding, stride=1)
        new_st_map = np.zeros(new_map_shape)
        for i in range(old_height):
            for j in range(old_width):
                if old_depth is None:
                    new_st_map[i * old_stride, j *
                               old_stride] = sensitivity_map[i, j]
                    continue
                for d in range(old_depth):
                    new_st_map[d, i * old_stride,
                               j * old_stride] = sensitivity_map[d, i, j]
        return new_st_map

    def backward(self, sensitivity_map):
        '''误差反向传递'''
        new_kernels = [k.deepcopy() for k in self.kernels]
        for new_kernel in new_kernels:
            new_kernel.turn_round()
        old_depth, old_height, old_width = expand_shape(
            np.shape(sensitivity_map))
        i_depth, i_height, i_width = expand_shape(self.shape)
        k_depth, k_height, k_width = expand_shape(new_kernels[0].shape)
        new_st_map = self.extend_to_one_stride(new_kernels[0].shape,
                                               new_kernels[0].stride,
                                               sensitivity_map)
        new_width = (k_width + (i_width - 1) * 1 - 2 * self.padding)
        new_height = (k_height + (i_height - 1) * 1 - 2 * self.padding)

        width_padding = new_width - old_width
        height_padding = new_height - old_height
        new_st_map = around_with_zero(input_array=sensitivity_map,
                                      width_padding=width_padding,
                                      height_padding=height_padding)
        conv_map = np.zeros(self.shape)
        for i, new_kernel in enumerate(new_kernels, 0):
            if k_depth is None:
                conv_map[:, :] += self.convolution(
                    new_st_map[i, :, :], 0, new_kernel)
                continue
            kernels_2d = new_kernel.expands_2d()
            for d, kernel_2d in enumerate(kernels_2d, 0):
                conv_map[d, :, :] += self.convolution(new_st_map[i, :, :],
                                                      0, kernel_2d)
        self.sensitivity_map = conv_map * self.active_derive(self.input_array)

    def get_sensitivity_map(self):
        # 计算残差网络
        return self.sensitivity_map

    def update(self, sensitivity_map, learning_rate):
        if self.input_array.ndim == 3:
            i_depth, i_height, i_width = np.shape(sensitivity_map)
        else:
            i_height, i_width = np.shape(sensitivity_map)
            i_depth = None
        new_map = self.extend_to_one_stride(kernel_shape=self.kernel[0].shape,
                                            old_stride=self.kernels[0].stride,
                                            sensitivity_map=sensitivity_map)
        depth, height, width = self.expand_shape(new_map.shape)
        for k, _kernel in enumerate(self.kernels):
            _conv_kernel = conv_kernel(kernel_shape=(height, width),
                                       weights=new_map[k, :, :],
                                       bias=0.0, stride=1)
            if i_depth is None:
                _kernel.weights_grad += self.convolution(
                    self.input_array, self.padding, _conv_kernel)
                continue
            for d in range(i_depth):
                _kernel.weights_grad[d, :, :] += self.convolution(
                    self.input_array[d, :, :], self.padding, _conv_kernel)
            _kernel.bias_grad += np.sum(sensitivity_map[k, :, :])
            _kernel.update(learning_rate)

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from conv_kernel import conv_kernel
from cnn_utils import *


class conv_layer(object):
    '''docstring for conv_layer'''

    def __init__(self, **kwargs):
        """

        Args:
          action: action function
          action_derive: the derivative of the function
          zero_padding:zero padding
          kernel_shape:the shape of kernel
          kernel_stride:kernel stride
          kernel_num:number of kernels
          alpha: learning rate

        Returns:

        """
        if kwargs is None:
            raise ValueError('Parameters is None!')
        self.action = kwargs['action']
        self.action_derive = kwargs['action_derive']
        self.padding = kwargs['zero_padding']
        self.shape = kwargs['input_shape']
        self.kernel_shape = kwargs['kernel_shape']
        self.kernel_stride = kwargs['kernel_stride']
        self.kernel_num = kwargs['kernel_num']
        self.kernels = conv_kernel.obtin_kernels(kernel_shape=self.kernel_shape,
                                                 stride=self.kernel_stride,
                                                 kernel_num=self.kernel_num)
        self.feature_map = None
        self.feature_shape = conv_kernel.calc_feature_shape(
            input_shape=self.shape, kernel_shape=self.kernel_shape,
            kernel_num=self.kernel_num, padding=self.padding,
            stride=self.kernel_stride)

    def forward(self, input_array):
        self.input_array = input_array
        array = around_with_zero(input_array=self.input_array,
                                 width_padding=self.padding,
                                 height_padding=self.padding)
        conv_map = self.convolution(array, *self.kernels)
        self.feature_map = self.action(conv_map)
        return self.feature_map

    def convolution(self, array, *kernels):
        if self.feature_shape is None:
            return
        conv_map = np.zeros(self.feature_shape)
        out_depth, out_height, out_width = expand_shape(self.feature_shape)
        if out_depth is None:
            for i in range(out_height):
                for j in range(out_width):
                    k = kernels[0]
                    patch = get_patch(i, j, array, k)
                    conv_map[i, j] = np.sum(patch * k.weights) + k.bias
        else:
            for i in range(out_height):
                for j in range(out_width):
                    k = kernels[0]
                    patch = get_patch(i, j, array, k.shape, k.stride)
                    for d in range(out_depth):
                        k = kernels[d]
                        conv_map[d, i, j] = np.sum(patch * k.weights) + k.bias
        return conv_map

    def extend_to_one_stride(self, kernel_shape, old_stride, delta_map):
        old_depth, old_height, old_width = expand_shape(np.shape(delta_map))
        new_map_shape = conv_kernel.calc_feature_shape(
            input_shape=self.shape, kernel_shape=kernel_shape,
            kernel_num=old_depth, padding=self.padding, stride=1)
        new_st_map = np.zeros(new_map_shape)
        if old_depth is None:
            for i in range(old_height):
                for j in range(old_width):
                    new_st_map[i * old_stride, j *
                               old_stride] = delta_map[i, j]
        else:
            for d in range(old_depth):
                for i in range(old_height):
                    for j in range(old_width):
                        new_st_map[d, i * old_stride,
                                   j * old_stride] = delta_map[d, i, j]
        return new_st_map

    def backward(self, delta_map):
        '''误差反向传递'''
        new_kernels = [k.deepcopy() for k in self.kernels]
        for new_kernel in new_kernels:
            new_kernel.turn_round()
        old_depth, old_height, old_width = expand_shape(np.shape(delta_map))
        i_depth, i_height, i_width = expand_shape(self.shape)
        k_depth, k_height, k_width = expand_shape(self.kernel_shape)
        new_st_map = self.extend_to_one_stride(self.kernel_shape,
                                               self.kernel_stride,
                                               delta_map)
        new_width = (k_width + (i_width - 1) * 1 - 2 * self.padding)
        new_height = (k_height + (i_height - 1) * 1 - 2 * self.padding)

        width_padding = new_width - old_width
        height_padding = new_height - old_height
        new_st_map = around_with_zero(input_array=new_st_map,
                                      width_padding=width_padding,
                                      height_padding=height_padding)
        conv_map = np.zeros(self.shape)
        if k_depth is None:
            for i, new_kernel in enumerate(new_kernels, 0):
                conv_map[:, :] += self.convolution(new_st_map[i, :, :],
                                                   new_kernel)
        else:
            for i, new_kernel in enumerate(new_kernels, 0):
                kernels_2d = new_kernel.expands_2d()
                for d, kernel_2d in enumerate(kernels_2d, 0):
                    conv_map[d, :, :] += self.convolution(new_st_map[i, :, :],
                                                          kernel_2d)
        self.delta_map = conv_map * self.action_derive(self.input_array)
        self.calc_gradient(delta_map)
        return self.delta_map

    def calc_gradient(self, delta_map):
        i_depth, i_height, i_width = expand_shape(np.shape(delta_map))
        new_map = self.extend_to_one_stride(kernel_shape=self.kernel_shape,
                                            old_stride=self.kernel_stride,
                                            delta_map=delta_map)
        depth, height, width = self.expand_shape(new_map.shape)
        array = around_with_zero(input_array=self.input_array,
                                 width_padding=self.width_padding,
                                 height_padding=self.height_padding)
        if i_depth is None:
            for k, _kernel in enumerate(self.kernels):
                _conv_kernel = conv_kernel(kernel_shape=(height, width),
                                           weights=new_map[k, :, :],
                                           bias=0.0, stride=1)
                _kernel.weights_grad += self.convolution(
                    array, _conv_kernel)
        else:
            for k, _kernel in enumerate(self.kernels):
                _conv_kernel = conv_kernel(kernel_shape=(height, width),
                                           weights=new_map[k, :, :],
                                           bias=0.0, stride=1)
                for d in range(i_depth):
                    _kernel.weights_grad[d, :, :] += self.convolution(
                        array[d, :, :], _conv_kernel)
                _kernel.bias_grad += np.sum(delta_map[k, :, :])

    def update(self, alpha, batch_size):
        """

        Args:
          batch_size:batch size
          alpha: learning rate

        Returns:

        """
        for _kernel in self.kernels:
            _kernel.update(alpha, batch_size)

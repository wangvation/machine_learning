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
        self.input_shape = kwargs['input_shape']
        self.kernel_shape = kwargs['kernel_shape']
        self.kernel_stride = kwargs['kernel_stride']
        self.kernel_num = kwargs['kernel_num']
        self.kernels = conv_kernel.obtin_kernels(kernel_num=self.kernel_num,
                                                 kernel_shape=self.kernel_shape,
                                                 stride=self.kernel_stride)
        self.feature_map = None
        self.feature_shape = conv_kernel.calc_feature_shape(
            input_shape=self.input_shape, kernel_shape=self.kernel_shape,
            kernel_num=self.kernel_num, padding=self.padding,
            stride=self.kernel_stride)

    def forward(self, input):
        self.input = input.reshape(self.input_shape)
        debug(True, 'conv_layer input:', np.max(self.input),
              np.min(self.input), np.mean(self.input))
        array = around_with_zero(input_array=self.input,
                                 width_padding=self.padding,
                                 height_padding=self.padding)
        conv_map = self.convolution(array, self.feature_shape, *self.kernels)
        self.feature_map = self.action(conv_map)
        return self.feature_map

    def convolution(self, array, out_shape, *kernels):
        if out_shape is None:
            return
        conv_map = np.zeros(out_shape)
        out_depth, out_height, out_width = expand_shape(out_shape)
        for i in range(out_height):
            for j in range(out_width):
                k = kernels[0]
                patch = get_patch(i, j, array, k)
                if out_depth is None:
                    conv_map[i, j] += (np.sum(
                        np.multiply(patch, k.weights)) + k.bias)
                    continue
                for d in range(out_depth):
                    k = kernels[d]
                    conv_map[d, i, ] += (np.sum(
                        np.multiply(patch, k.weights)) + k.bias)
        return conv_map

    def extend_to_one_stride(self, kernel_shape, old_stride, delta_map):
        if old_stride == 1:
            return delta_map
        old_depth, old_height, old_width = expand_shape(delta_map.shape)
        new_map_shape = conv_kernel.calc_feature_shape(
            input_shape=self.input_shape, kernel_shape=kernel_shape,
            kernel_num=self.kernel_num, padding=self.padding, stride=1)
        new_delta_map = np.zeros(new_map_shape)
        for i in range(old_height):
            for j in range(old_width):
                if old_depth is None:
                    new_delta_map[i * old_stride, j *
                                  old_stride] += delta_map[i, j]
                else:
                    new_delta_map[:, i * old_stride,
                                  j * old_stride] += delta_map[:, i, j]
        return new_delta_map

    def backward(self, delta_map):
        '''误差反向传递'''
        debug(True, 'conv layer-backward:', self.input_shape,
              np.max(delta_map), np.min(delta_map), np.mean(delta_map))
        delta_map = delta_map.reshape(self.feature_shape)
        one_stride_map = self.extend_to_one_stride(self.kernel_shape,
                                                   self.kernel_stride,
                                                   delta_map)

        i_depth, i_height, i_width = expand_shape(self.input_shape)
        k_depth, k_height, k_width = expand_shape(self.kernel_shape)
        os_depth, os_height, os_width = expand_shape(one_stride_map.shape)

        new_width = k_width + i_width - 1
        new_height = k_height + i_height - 1

        width_padding = (new_width - os_width) // 2
        height_padding = (new_height - os_height) // 2

        padding_map = around_with_zero(input_array=one_stride_map,
                                       width_padding=width_padding,
                                       height_padding=height_padding)
        self.delta_map = np.zeros(self.input_shape)

        new_kernels = [k.deepcopy() for k in self.kernels]
        for new_kernel in new_kernels:
            new_kernel.turn_round()
            new_kernel.bias = 0
            new_kernel.stride = 1
        for k, new_kernel in enumerate(new_kernels, 0):
            if k_depth is None:
                conv_map = self.convolution(padding_map[k, ...],
                                            (i_height, i_width),
                                            new_kernel)
                self.delta_map += np.multiply(
                    conv_map, self.action_derive(self.input))
                continue
            kernels_2d = new_kernel.expands_2d()
            for d, kernel_2d in enumerate(kernels_2d, 0):
                conv_map = self.convolution(padding_map[k, ...],
                                            (i_height, i_width),
                                            kernel_2d)
                self.delta_map[d, ...] += np.multiply(
                    conv_map, self.action_derive(self.input[d, ...]))
        self.calc_gradient(delta_map)
        return self.delta_map

    def calc_gradient(self, delta_map):
        k_depth, k_height, k_width = expand_shape(self.kernel_shape)
        new_map = self.extend_to_one_stride(kernel_shape=self.kernel_shape,
                                            old_stride=self.kernel_stride,
                                            delta_map=delta_map)
        depth, height, width = expand_shape(new_map.shape)
        array = around_with_zero(input_array=self.input,
                                 width_padding=self.padding,
                                 height_padding=self.padding)
        for k, _kernel in enumerate(self.kernels):
            _conv_kernel = conv_kernel(kernel_shape=(height, width),
                                       weights=new_map[k, ...],
                                       bias=0.0, stride=1)
            _kernel.bias_grad += np.sum(delta_map[k, ...])
            if k_depth is None:
                _kernel.weights_grad += self.convolution(
                    array, (k_height, k_width), _conv_kernel)
                continue
            for d in range(k_depth):
                _kernel.weights_grad[d, ...] += self.convolution(
                    array[d, ...], (k_height, k_width), _conv_kernel)

    def update(self, alpha, batch_size):
        """

        Args:
          batch_size:batch size
          alpha: learning rate

        Returns:

        """
        for _kernel in self.kernels:
            _kernel.update(alpha, batch_size)

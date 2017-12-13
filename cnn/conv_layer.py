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
        self.weights = []
        self.bias = []
        for k in range(self.kernel_num):
            weight = np.random.normal(loc=0.0,
                                      scale=0.03,
                                      size=self.kernel_shape)
            self.weights.append(weight)
            self.bias.append(0.0)
        self.weights_grad = [np.zeros(w.shape) for w in self.weights]
        self.bias_grad = [0.0 for b in self.bias]
        self.feature_map = None
        self.feature_shape = conv_kernel.calc_feature_shape(
            input_shape=self.input_shape, kernel_shape=self.kernel_shape,
            kernel_num=self.kernel_num, padding=self.padding,
            stride=self.kernel_stride)

    def forward(self, input):
        self.input = input.reshape(self.input_shape)
        debug(True, 'conv_layer input:', np.max(self.input),
              np.min(self.input), np.mean(self.input))
        padding_array = around_with_zero(input_array=self.input,
                                         width_padding=self.padding,
                                         height_padding=self.padding)
        conv_map = self.convolution(array=padding_array,
                                    out_shape=self.feature_shape,
                                    kernel_shape=self.kernel_shape,
                                    weights=self.weights, bias=self.bias,
                                    stride=self.kernel_stride)
        self.feature_map = self.action(conv_map)
        return self.feature_map

    def convolution(self, array, out_shape,
                    kernel_shape, weights, bias, stride):
        if out_shape is None:
            return
        conv_map = np.zeros(out_shape)
        out_depth, out_height, out_width = expand_shape(out_shape)
        for i in range(out_height):
            for j in range(out_width):
                patch = get_patch(i, j, array, kernel_shape, stride)
                if out_depth is None:
                    conv_map[i, j] += (np.sum(
                        np.multiply(patch, weights)) + bias)
                    continue
                for k in range(len(weights)):
                    conv_map[k, i, j] += (np.sum(
                        np.multiply(patch, weights[k])) + bias[k])
        return conv_map

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

        flip_weights = [self.turn_round(w, True) for w in self.weights]
        for k, flip_weight in enumerate(flip_weights, 0):
            if k_depth is None:
                conv_map = self.convolution(array=padding_map[k, ...],
                                            out_shape=(i_height, i_width),
                                            kernel_shape=self.kernel_shape,
                                            weights=flip_weight,
                                            bias=0.0, stride=1)

                self.delta_map += np.multiply(
                    conv_map, self.action_derive(self.input))
                continue
            for d in range(k_depth):
                conv_map = self.convolution(array=padding_map[k, ...],
                                            out_shape=(i_height, i_width),
                                            kernel_shape=(k_width, k_height),
                                            weights=flip_weight[d, ...],
                                            bias=0.0, stride=1)
                self.delta_map[d, ...] += np.multiply(
                    conv_map, self.action_derive(self.input[d, ...]))
        self.calc_gradient(delta_map)
        return self.delta_map

    def calc_gradient(self, delta_map):
        k_depth, k_height, k_width = expand_shape(self.kernel_shape)
        one_stride_map = self.extend_to_one_stride(
            kernel_shape=self.kernel_shape,
            old_stride=self.kernel_stride,
            delta_map=delta_map)
        os_depth, os_height, os_width = expand_shape(one_stride_map.shape)
        padding_array = around_with_zero(input_array=self.input,
                                         width_padding=self.padding,
                                         height_padding=self.padding)
        for k, weight_grad in enumerate(self.weights_grad, 0):
            self.bias_grad[k] += np.sum(delta_map[k, ...])
            if k_depth is None:
                weight_grad += self.convolution(
                    array=padding_array, out_shape=(k_height, k_width),
                    kernel_shape=(os_height, os_width),
                    weights=one_stride_map[k, ...], bias=0.0, stride=1)
                continue
            for d in range(k_depth):
                weight_grad[d, ...] += self.convolution(
                    array=padding_array[d, ...], out_shape=(k_height, k_width),
                    kernel_shape=(os_height, os_width),
                    weights=one_stride_map[k, ...], bias=0.0, stride=1)

    def update(self, alpha, batch_size):
        """

        Args:
          batch_size:batch size
          alpha: learning rate

        Returns:

        """
        for k in range(self.kernel_num):
            self.weights[k] -= alpha * self.weights_grad[k] / batch_size
            self.bias[k] -= alpha * self.bias_grad[k] / batch_size
            self.weights_grad[k][...] = 0.0
            self.bias_grad[k] = 0.0

    def extend_to_one_stride(self, kernel_shape, old_stride, delta_map):
        if old_stride == 1:
            return delta_map
        old_depth, old_height, old_width = expand_shape(delta_map.shape)
        new_map_shape = self.calc_feature_shape(
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

    def turn_round(self, array, copy=False):
        """turn round the array"""
        if copy:
            flip_array = np.copy(array)
        else:
            flip_array = array
        if flip_array.ndim == 2:
            np.flip(flip_array, axis=0)
            np.flip(flip_array, axis=1)
        if flip_array.ndim == 3:
            np.flip(flip_array, axis=1)
            np.flip(flip_array, axis=2)
        return flip_array

    def calc_feature_shape(self, input_shape, kernel_shape, kernel_num,
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

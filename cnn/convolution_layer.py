#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
# from convolution_kernel import kernel


class convolution_layer(object):
    '''docstring for convolution_layer'''

    def __init__(self, input_array, zero_padding=0, kernels=[]):
        self.kernels = kernels
        self.padding = zero_padding
        self.input_array = self.around_with_zero(input_array)
        self.shape = self.input_array.shape
        self.feature_map = None
        pass

    def conv(self):
        feature_shape = self.calc_feature_shape(self.kernels)
        if feature_shape is None:
            return
        out_width, out_height, out_depth = feature_shape
        if out_depth is None:
            self.feature_map = np.zeros((out_width, out_height),
                                        dtype=np.float32)
        else:
            self.feature_map = np.zeros(feature_shape, dtype=np.float32)
        for i in range(out_width):
            for j in range(out_height):
                k = self.kernels[0]
                patch = self.get_patch(i, j, self.input_array, k)
                if out_depth is None:
                    net_ij = np.sum(patch * k.weights) + k.bias
                    self.feature_map[i, j] = self.relu(net_ij)
                else:
                    for d in range(out_depth):
                        k = self.kernels[d]
                        net_dij = np.sum(patch * k.weights) + k.bias
                        self.feature_map[i, j, d] = self.relu(net_dij)
        return self.feature_map

    def around_with_zero(self, input_array):
        if self.zero_padding == 0:
            return input_array
        if input_array.ndim == 2:
            width, height = input_array.shape
            ret = np.zeros((width + 2 * self.padding,
                            height + 2 * self.padding), dtype=np.float32)
            ret[self.padding:-self.padding,
                self.padding:-self.padding] = input_array
            return ret
        else:
            width, height, depth = input_array.shape
            ret = np.zeros((width + 2 * self.padding,
                            height + 2 * self.padding,
                            depth), dtype=np.float32)
            ret[self.padding:-self.padding, self.padding:-
                self.padding, :] = input_array
            return ret

    def get_patch(self, i, j, p_kernel):
        start_i = i * p_kernel.stride
        start_j = j * p_kernel.stride

        if self.input_array.ndim == 2:
            return self.input_array[start_i:start_i + p_kernel.width,
                                    start_j:start_j + p_kernel.height]
        if self.input_array.ndim == 3:
            return self.input_array[start_i:start_i + p_kernel.width,
                                    start_j:start_j + p_kernel.height, :]

    def calc_feature_shape(self):
        '''
        Return a shape of feature map
        Returns
        -------
        withd:feature map withd
        height:feature map height
        depth:feature map depth, if feature map is 2D,depth is None
        '''
        kernel_num = len(self.kernels)
        if kernel_num == 0:
            raise ValueError('the length of the kernels must be more than zero')
        for i in range(kernel_num - 1):
            if self.kernels[i].shape() != self.kernels[i + 1].shape():
                raise ValueError('the kernels must be equal in shape')
        if len(self.shape) == 2:
            input_width, input_height = self.shape
            input_depth = None
        elif len(self.shape) == 3:
            input_width, input_height, input_depth = self.shape
        else:
            raise ValueError('the length of the input_shape must be 2 or 3')
        out_width = (input_width - self.kernels[0].width + 2 *
                     self.padding) / self.kernels[0].stride + 1
        out_height = (input_height - self.kernels[0].height + 2 *
                      self.padding) / self.kernels[0].stride + 1
        if input_depth != self.kernels[0].depth or\
                out_width <= 0 or out_height <= 0:
            return None
        if kernel_num == 1:
            return out_width, out_height, None
        return out_width, out_height, kernel_num

    def get_output(self):
        return self.feature_map

    def get_sensitivity_map(self):
        # 计算残差网络
        pass

    def relu(self, x):
        return max(0, x)

    def relu_prime(self, y):
        return 0 if y <= 0 else 1

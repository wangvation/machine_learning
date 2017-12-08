#!/usr/bin/python3
# -*- coding: utf-8 -*-
import copy


class pooling_kernel(object):
    """docstring for kernel"""

    def __init__(self, kernel_shape, stride=1):
        self.stride = stride
        self.shape = kernel_shape

    def deepcopy(self):
        """ """
        return copy.deepcopy(self, memo=None, _nil=[])

    @classmethod
    def obtin_kernels(cls, kernel_num, kernel_shape, stride=1):
        """

        Args:
          kernel_shape:
          stride:  (Default value = 1)
          kernel_num:  (Default value = None)

        Returns:

        """
        if kernel_num is None or kernel_num <= 1:
            raise ValueError('The kernel_num must be greater than 1.')
        return [pooling_kernel(kernel_shape, stride) for _ in range(kernel_num)]

    def calc_feature_shape(cls, input_shape, kernel_shape, padding=0, stride=1):
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
                     padding) / stride + 1
        out_height = (input_height - kernel_height + 2 *
                      padding) / stride + 1
        if (input_depth != kernel_depth):
            raise ValueError('The depth of the input_shape must be equal to '
                             'the depth of the convolution_shape')
        if(out_width <= 0 or out_height <= 0):
            return None
        if input_depth is None:
            return out_height, out_width
        return input_depth, out_height, out_width

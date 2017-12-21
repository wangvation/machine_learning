#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from conv_layer import conv_layer
# from fully_connect_layer import fc_layer
# from softmax_layer import softmax_layer

# #########################################


def relu(array):
    negative_mask = array < 0.0
    array[negative_mask] = 0.0
    return array


def relu_derive(y):
    derive = np.zeros(y.shape)
    derive[y > 0.0] = 1.0
    return derive


def test_conv_layer():
    input_arr = np.array([[[0, 1, 1, 0, 2],
                           [2, 2, 2, 2, 1],
                           [1, 0, 0, 2, 0],
                           [0, 1, 1, 0, 0],
                           [1, 2, 0, 0, 2]],
                          [[1, 0, 2, 2, 0],
                           [0, 0, 0, 2, 0],
                           [1, 2, 1, 2, 1],
                           [1, 0, 0, 0, 0],
                           [1, 2, 1, 1, 1]],
                          [[2, 1, 2, 0, 0],
                           [1, 0, 0, 1, 0],
                           [0, 2, 1, 0, 1],
                           [0, 1, 2, 2, 2],
                           [2, 1, 0, 0, 1]]])
    weights_1 = np.array([[[-1, 1, 0],
                           [0, 1, 0],
                           [0, 1, 1]],
                          [[-1, -1, 0],
                           [0, 0, 0],
                           [0, -1, 0]],
                          [[0, 0, -1],
                           [0, 1, 0],
                           [1, -1, -1]]])
    weights_2 = np.array([[[1, 1, -1],
                           [-1, -1, 1],
                           [0, -1, 1]],
                          [[0, 1, 0],
                           [-1, 0, -1],
                           [-1, 1, 0]],
                          [[-1, 0, 0],
                           [-1, 0, 1],
                           [-1, 0, 0]]])
    bias_1 = 1
    bias_2 = 0
    cl = conv_layer(action=relu, zero_padding=1,
                    action_derive=relu_derive,
                    input_shape=(3, 5, 5), kernel_stride=2,
                    kernel_shape=(3, 3, 3), kernel_num=2)
    cl.weights[0] = weights_1
    cl.weights[1] = weights_2
    cl.bias[0] = bias_1
    cl.bias[1] = bias_2
    feature_map = cl.forward(input_arr)
    print(feature_map)
    pass


if __name__ == '__main__':
    test_conv_layer()

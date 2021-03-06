#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np


def around_with_zero(input_array, width_padding, height_padding):
    """

    Args:
        input_array:
        width_padding:
        height_padding:

    Returns:

    """
    if height_padding == 0 and width_padding == 0:
        return input_array
    if input_array.ndim == 2:
        height, width = input_array.shape
        new_array = np.zeros((height + 2 * height_padding,
                              width + 2 * width_padding), dtype=np.float32)
        new_array[height_padding:-height_padding,
                  width_padding:-width_padding] += input_array
        return new_array
    else:
        depth, height, width = input_array.shape
        new_array = np.zeros((depth, height + 2 * height_padding,
                              width + 2 * width_padding), dtype=np.float32)
        new_array[:, height_padding:-height_padding,
                  width_padding:-width_padding] += input_array
        return new_array


def get_patch(i, j, array, kernel_shape, stride):
    """

    Args:
      i: row index
      j: col index
      array:
      p_kernel:

    Returns:

    """
    depth, width, height = expand_shape(kernel_shape)
    start_i = i * stride
    start_j = j * stride

    if depth is None:
        return array[start_i:start_i + height,
                     start_j:start_j + width]
    return array[:, start_i:start_i + height,
                 start_j:start_j + width]


def expand_shape(shape, default_depth=None):
    """
    expand the shape
    Args:
      shape:

    Returns:
    depth, height, width
    if the length of shape is 2, depth is default_depth,
    """
    if len(shape) == 3:
        depth, height, width = shape
    else:
        depth = default_depth
        height, width = shape
    return depth, height, width


def debug(allow_log=False, *msg):
    if allow_log is False:
        print(msg)
    pass


def square_sum(x):
    return np.sum(np.multiply(x, x))

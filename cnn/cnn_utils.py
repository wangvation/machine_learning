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
    if input_array.ndim == 2:
        height, width = input_array.shape
        new_array = np.zeros((height + 2 * height_padding,
                              width + 2 * width_padding), dtype=np.float32)
        new_array[height_padding:-height_padding,
                  width_padding:-width_padding] = input_array
        return new_array
    else:
        depth, height, width = input_array.shape
        new_array = np.zeros((depth, height + 2 * height_padding,
                              width + 2 * width_padding), dtype=np.float32)
        new_array[:, height_padding:-height_padding,
                  width_padding:-width_padding] = input_array
        return new_array


def get_patch(i, j, array, p_kernel):
    """

    Args:
      i: row index
      j: col index
      array:
      p_kernel:

    Returns:

    """
    start_i = i * p_kernel.stride
    start_j = j * p_kernel.stride

    if array.ndim == 2:
        return array[start_i:start_i + p_kernel.height,
                     start_j:start_j + p_kernel.width]
    if array.ndim == 3:
        return array[:, start_i:start_i + p_kernel.height,
                     start_j:start_j + p_kernel.width]


def expand_shape(shape):
    """
    expand the shape
    Args:
      shape:

    Returns:
    depth, height, width
    if the length of shape is 2, depth is None,
    """
    if len(shape) == 3:
        depth, height, width = shape
    else:
        depth = None
        height, width = shape
    return depth, height, width

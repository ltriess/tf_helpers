# -*- coding: utf-8 -*-

import math

import tensorflow as tf


def get_padding_sizes(
        spacial_dimensions: tuple, kernel_size: tuple, strides: tuple, scope: str = None
):

    """
    Computes the padding sizes for a rank 4 tensor according to the `kernel_size` and `strides`
    of a 2D convolution with `same` padding. The computation is equivalent to TF implementation.

    Parameters:
        spacial_dimensions (tuple): A tuple of two integers corresponding to the spacial
            dimensions H and W of Tensor(shape=[N, H, W, C]).
        kernel_size (tuple): A tuple of two integers defining the kernel size of a
            convolution for which to compute the corresponding padding of the input tensor.
        strides (tuple): A tuple of two integers defining the strides of a
            convolution for which to compute the corresponding padding of the input tensor.
        scope (str): The name scope of the function.

    Returns:
        (int, int, int, int): Four integers corresponding to the padding at the top, bottom,
            left, and right, respectively.

    Raises:
        TypeError: When `spacial_dimensions`, `kernel_size`, or `strides` are not tuples of integers.
        ValueError: If `strides` is smaller than 1 or greater than the spacial dimension.
        ValueError: If `kernel_size` is smaller than 1 or greater than the spacial dimension.
    """

    if not (isinstance(spacial_dimensions, tuple) and all([isinstance(d, int) for d in spacial_dimensions])):
        raise TypeError('`spacial_dimensions` must be a tuple of two integers.')
    if not (isinstance(kernel_size, tuple) and all([isinstance(k, int) for k in kernel_size])):
        raise TypeError('`kernel_size` must be a tuple of two integers.')
    if not (isinstance(strides, tuple) and all([isinstance(s, int) for s in strides])):
        raise TypeError('`strides` must be a tuple of two integers.')
    if not all([1 <= s <= d for s, d in zip(strides, spacial_dimensions)]):
        raise ValueError('Strides cannot be smaller than 1 or greater than the corresponding '
                         'spacial dimension. I got {}.'.format(strides))
    if not all([1 <= k <= d for k, d in zip(kernel_size, spacial_dimensions)]):
        raise ValueError('Kernel size cannot be smaller than 1 or greater than the corresponding '
                         'spacial dimension. I got {} vs. {}.'.format(kernel_size, spacial_dimensions))

    with tf.name_scope(name=scope, default_name='calculate_padding_sizes'):
        # Compute the output height and width
        out_h = int(math.ceil(float(spacial_dimensions[0]) / float(strides[0])))
        out_w = int(math.ceil(float(spacial_dimensions[1]) / float(strides[1])))

        # Calculate the amount of padding per spacial dimension
        pad_h = max((out_h - 1) * strides[0] + kernel_size[0] - spacial_dimensions[0], 0)
        pad_w = max((out_w - 1) * strides[1] + kernel_size[1] - spacial_dimensions[1], 0)

        pad_top = pad_h // 2  # amount of padding on the top
        pad_bottom = pad_h - pad_top  # amount of padding on the bottom
        pad_left = pad_w // 2  # amount of padding on the left
        pad_right = pad_w - pad_left  # amount of padding on the right

    return pad_top, pad_bottom, pad_left, pad_right

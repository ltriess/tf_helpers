# -*- coding: utf-8 -*-

import tensorflow as tf


def expand_image_height(
        tf_image: tf.Tensor, factor: int, scope: str = None
):

    """
    Enlarges the input image-like tensor [H, W, C] of rank 3 in the first dimension (H)

    Parameters:
        tf_image (tf.Tensor): A rank 3 `Tensor` of shape [H, W, C].
        factor (int): The factor specifies how many multiples of each element in H are produced.
        scope (str): The name scope of the function.

    Returns:
        Tensor(shape=[H * factor, W, C], dtype=tf.image.dtype)

    Raises:
        NotImplementedError: When `tf_image` has rank 4. Batch dimension is not yet supported.
        ValueError: When `tf_image` has a different rank than 3 and 4.
        ValueError: If `factor` is an integer greater or equal to 1.

    """

    shape = tf_image.shape

    if len(shape) == 4:
        raise NotImplementedError('Expand image height function does not support batched tensors.')
    if not len(shape) == 3:
        raise ValueError('The input tensor must be of rank 3.')
    if not isinstance(factor, int) or factor < 1:
        raise ValueError('Expected integer greater or equal to 1. I got {}.'.format(factor))

    with tf.name_scope(name=scope, default_name='expand_height'):
        shape = [shape[0] * factor, shape[1], shape[2]]
        tf_image = tf.expand_dims(tf_image, axis=1)
        tf_image = tf.tile(tf_image, multiples=[1, factor, 1, 1])
        tf_image = tf.reshape(tf_image, shape)
    return tf_image

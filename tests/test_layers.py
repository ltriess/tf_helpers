# -*- coding: utf-8 -*-

__author__ = """Larissa Triess"""
__email__ = 'mail@triess.eu'

import tensorflow as tf

from tf_helpers.training.layers import get_padding_sizes, pad

tf.enable_eager_execution()


class TestGetPaddingSizes(tf.test.TestCase):

    def test_exceptions(self):
        self.assertRaises(TypeError, get_padding_sizes, 10, (3, 3), (1, 1))
        self.assertRaises(TypeError, get_padding_sizes, (10, 10), 3, (1, 1))
        self.assertRaises(TypeError, get_padding_sizes, (10, 10), (3, 3), (1.0, 1.0))
        self.assertRaises(ValueError, get_padding_sizes, (10, 10), (3, 3), (0, 2))
        self.assertRaises(ValueError, get_padding_sizes, (10, 10), (3, 3), (1, 12))
        self.assertRaises(ValueError, get_padding_sizes, (10, 10), (12, 3), (2, 2))

    def test_output(self):
        sd = (10, 10)
        k = (3, 3)
        s = (1, 1)
        output = get_padding_sizes(sd, k, s)

        self.assertEqual(4, len(output))
        for o in output:
            self.assertIsInstance(o, int)
            self.assertLessEqual(0, o)


class TestPad(tf.test.TestCase):

    def test_shapes(self):
        t_shape = [4, 8]
        t = tf.random.uniform(t_shape, 0, 100, dtype=tf.int64)
        p = [[0, 1], [2, 1]]

        out_shape = [j + sum(k) for j, k in zip(t_shape, p)]

        modes = ['CONSTANT', 'REFLECT', 'SYMMETRIC', 'CYCLIC']
        for m in modes:
            o = pad(t, tf.convert_to_tensor(p), mode=m)
            self.assertEqual(out_shape, o.shape)


if __name__ == '__main__':
    tf.test.main()

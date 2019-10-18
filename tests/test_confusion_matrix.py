# -*- coding: utf-8 -*-

import unittest
import random
import string

import tensorflow as tf
import numpy as np

from tf_helpers.visu.confusion_matrix import confusion_matrix_image


class TestConfusionMatrix(unittest.TestCase):

    def setUp(self):
        self.num_classes = 10
        self.tf_conf_mat = tf.random.uniform([self.num_classes] * 2, minval=0, maxval=100, dtype=tf.float32)
        self.class_names = [''.join(random.choice(string.ascii_letters)
                                    for _ in range(random.randint(4, 12)))
                            for _ in range(self.num_classes)]

    def test_exceptions(self):
        self.assertRaises(TypeError, confusion_matrix_image, self.tf_conf_mat)
        self.assertRaises(AttributeError, confusion_matrix_image, self.class_names, self.tf_conf_mat)
        self.assertRaises(TypeError, confusion_matrix_image, self.tf_conf_mat, None)
        self.assertRaises(ValueError, confusion_matrix_image, self.tf_conf_mat, list())
        self.assertRaises(ValueError, confusion_matrix_image, self.tf_conf_mat, self.class_names[:-3])
        self.assertRaises(ValueError, confusion_matrix_image, self.tf_conf_mat[:, :-1], self.class_names)

    def test_output_shape_and_type(self):
        fig_dpi = 320
        fig_size = 3

        tf_conf_mat_img = confusion_matrix_image(self.tf_conf_mat, self.class_names, fig_dpi, fig_size)

        self.assertEqual((1, fig_dpi * fig_size, fig_dpi * fig_size, 3), tf_conf_mat_img.shape)
        self.assertEqual(tf.uint8, tf_conf_mat_img.dtype)

    def test_color_scaling(self):
        tf_conf_mat = confusion_matrix_image(self.tf_conf_mat, self.class_names)
        with tf.Session() as sess:
            conf_mat_img = sess.run(tf_conf_mat)

        self.assertAlmostEqual(0, np.amin(conf_mat_img))
        self.assertAlmostEqual(255, np.amax(conf_mat_img))


if __name__ == '__main__':
    unittest.main()

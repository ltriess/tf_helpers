# -*- coding: utf-8 -*-

import random
import string

import tensorflow as tf
import numpy as np

from tf_helpers.visu.confusion_matrix import confusion_matrix_image
from tf_helpers.visu.colorful_tensors import expand_image_height, create_image

tf.enable_eager_execution()


class TestConfusionMatrix(tf.test.TestCase):

    def setUp(self):
        self.num_classes = 10
        self.conf_mat = tf.random.uniform([self.num_classes] * 2, minval=0, maxval=100, dtype=tf.float32)
        self.class_names = [''.join(random.choice(string.ascii_letters)
                                    for _ in range(random.randint(4, 12)))
                            for _ in range(self.num_classes)]

    def test_exceptions(self):
        self.assertRaises(TypeError, confusion_matrix_image, self.conf_mat)
        self.assertRaises(AttributeError, confusion_matrix_image, self.class_names, self.conf_mat)
        self.assertRaises(TypeError, confusion_matrix_image, self.conf_mat, None)
        self.assertRaises(ValueError, confusion_matrix_image, self.conf_mat, list())
        self.assertRaises(ValueError, confusion_matrix_image, self.conf_mat, self.class_names[:-3])
        self.assertRaises(ValueError, confusion_matrix_image, self.conf_mat[:, :-1], self.class_names)

    def test_output_shape_and_type(self):
        fig_dpi = 320
        fig_size = 3

        conf_mat_img = confusion_matrix_image(self.conf_mat, self.class_names, fig_dpi, fig_size)

        self.assertEqual((1, fig_dpi * fig_size, fig_dpi * fig_size, 3), conf_mat_img.shape)
        self.assertEqual(tf.uint8, conf_mat_img.dtype)

    def test_color_scaling(self):
        conf_mat_img = confusion_matrix_image(self.conf_mat, self.class_names)

        self.assertAlmostEqual(0, np.amin(conf_mat_img))
        self.assertAlmostEqual(255, np.amax(conf_mat_img))


class TestExpandImageHeight(tf.test.TestCase):

    def test_exceptions(self):
        self.assertRaises(NotImplementedError, expand_image_height, tf.random.normal([1]*4), 3)
        self.assertRaises(ValueError, expand_image_height, tf.random.normal([1]*2), 1)
        self.assertRaises(ValueError, expand_image_height, tf.random.normal([1]*5), 8)
        self.assertRaises(ValueError, expand_image_height, tf.random.normal([1]*3), 0)
        self.assertRaises(ValueError, expand_image_height, tf.random.normal([1]*3), -1)
        self.assertRaises(ValueError, expand_image_height, tf.random.normal([1]*3), 3.2)

    def test_shape(self):
        f = random.randint(1, 10)
        h, w, c = random.randint(1, 40), random.randint(1, 100), random.randint(1, 10)
        self.assertEqual((h * f, w, c),
                         expand_image_height(tf.random.normal([h, w, c]), f).shape)


class TestCreateImage(tf.test.TestCase):

    def setUp(self):
        self.shape = (20, 80)
        self.tensor = tf.random.normal(self.shape, mean=10, stddev=20, dtype=tf.float32)
        self.mask = tf.convert_to_tensor(np.random.choice([0, 1], self.shape), dtype=tf.bool)

    def test_exceptions(self):
        self.assertRaises(TypeError, create_image, self.tensor, mask=tf.cast(self.mask, dtype=tf.int64))
        self.assertRaises(ValueError, create_image, self.tensor[:, 10:], mask=self.mask)
        self.assertRaises(ValueError, create_image, self.tensor, minval=20, maxval=10)
        self.assertRaises(ValueError, create_image, self.tensor, expand_height=0)
        self.assertRaises(ValueError, create_image, self.tensor, cycle_color_map=0)
        self.assertRaises(ValueError, create_image, self.tensor, cmap='norealcmap')

    def test_shape_and_type(self):
        # No 1
        image = create_image(self.tensor)
        self.assertEqual((1, *self.shape, 3), image.shape)
        self.assertEqual(tf.uint8, image.dtype)

        # No 2
        image = create_image(self.tensor, mask=self.mask)
        self.assertEqual((1, *self.shape, 3), image.shape)
        self.assertEqual(tf.uint8, image.dtype)

        # No 3
        image = create_image(self.tensor, expand_height=8)
        self.assertEqual((1, 8 * self.shape[0], self.shape[1], 3), image.shape)

    def test_color_scaling(self):
        for mi in [random.uniform(-100, 100) for _ in range(10)]:
            for ma in [random.uniform(0, 200) for _ in range(10)]:
                if ma <= mi:
                    self.assertRaises(ValueError, create_image, self.tensor, minval=mi, maxval=ma)
                else:
                    image = create_image(self.tensor, minval=mi, maxval=ma)
                    self.assertLessEqual(0, np.amin(image))
                    self.assertGreaterEqual(255, np.amax(image))
                    self.assertLess(1, np.mean(image))


if __name__ == '__main__':
    tf.test.main()

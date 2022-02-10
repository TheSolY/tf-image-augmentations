import tensorflow as tf
from tf_image_augmentations import utils


class TestUtils(tf.test.TestCase):
    def setUp(self):
        pass

    def test_image_shape_to_hw(self):
        img_shape_2d = tf.random.uniform([2], maxval=500, dtype=tf.int32)
        self.assertAllEqual(utils.image_shape_to_hw(img_shape_2d), img_shape_2d)
        img_shape_3d = tf.random.uniform([3], maxval=500, dtype=tf.int32)
        self.assertAllEqual(utils.image_shape_to_hw(img_shape_3d), img_shape_3d[:2])
        img_shape_4d = tf.random.uniform([4], maxval=500, dtype=tf.int32)
        self.assertAllEqual(utils.image_shape_to_hw(img_shape_4d), img_shape_4d[1:3])

    def test_hwc_to_bhwc(self):
        img_shape_3d = tf.random.uniform([3], maxval=500, dtype=tf.int32)
        dummy_img_3d = tf.random.uniform(img_shape_3d)
        self.assertAllEqual(tf.expand_dims(dummy_img_3d, 0), utils.hwc_to_bhwc(dummy_img_3d))

    def test_bhwc_to_hwc(self):
        img_shape_4d = [1, 256, 256, 3]
        dummy_img_4d = tf.random.uniform(img_shape_4d)
        self.assertAllEqual(tf.squeeze(dummy_img_4d, 0), utils.bhwc_to_hwc(dummy_img_4d))

    def test_image_dims_to_coordinates(self):
        img_shape_3d = tf.random.uniform([3], maxval=10, dtype=tf.int32)
        coords = utils.image_dims_to_coordinates(img_shape_3d)
        self.assertAllEqual(tf.shape(coords), [3, img_shape_3d[0] * img_shape_3d[1] * img_shape_3d[2]])
        self.assertAllEqual(tf.reduce_max(coords, axis=1), img_shape_3d - 1)

    def test_image_to_coordinates(self):
        img_shape_3d = tf.random.uniform([3], maxval=10, dtype=tf.int32)
        dummy_img_3d = tf.random.uniform(img_shape_3d)
        coords = utils.image_to_coordinates(dummy_img_3d)
        self.assertAllEqual(tf.shape(coords), [3, img_shape_3d[0] * img_shape_3d[1] * img_shape_3d[2]])
        self.assertAllEqual(tf.reduce_max(coords, axis=1), img_shape_3d - 1)



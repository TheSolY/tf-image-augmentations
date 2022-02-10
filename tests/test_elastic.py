import tensorflow as tf
from tf_image_augmentations import elastic


class TestElastic(tf.test.TestCase):
    def setUp(self):
        self.sigma = 3
        self.alpha = 10.1

    def test_generate_random_elastic_flow(self):
        image_shape = tf.random.uniform([2], maxval=500, dtype=tf.int32)
        flow = elastic.generate_random_elastic_flow(image_shape, self.sigma, self.alpha)

        # The flow shape should be HxWx2
        self.assertEqual(flow.shape, tf.concat([image_shape, [2]], axis=0))
        # the flow absolut values should be less or equal to the deformation intensity
        self.assertLessEqual(tf.reduce_max(flow), self.alpha)
        self.assertGreaterEqual(tf.reduce_min(flow), -self.alpha)

    def test_warp_image_by_flow(self):
        image_shape = tf.random.uniform([3], maxval=500, dtype=tf.int32)
        org_image = tf.random.uniform(image_shape, maxval=1.0, dtype=tf.float32)
        flow = elastic.generate_random_elastic_flow(image_shape, self.sigma, self.alpha)
        trans_image = elastic.warp_image_by_flow(org_image, flow)

        # The image should be different after the transformation
        self.assertNotAllEqual(trans_image, org_image)

        # Nothing fishy happens
        self.assertTrue(tf.reduce_all(tf.math.is_finite(trans_image)))




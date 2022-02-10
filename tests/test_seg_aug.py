import tensorflow as tf
from tf_image_augmentations import seg_aug


class TestSegAug(tf.test.TestCase):
    def setUp(self):
        img_dims = [256, 256, 3]
        lbl_dims = [256, 256, 1]
        self.dummy_image = tf.random.uniform(img_dims, maxval=255, dtype=tf.int32)
        self.dummy_label = tf.random.uniform(lbl_dims, maxval=1, dtype=tf.float32)
        pass

    def test_random_affine_transform(self):
        # The image and label should stay the same with the default values
        trans_image, trans_label = seg_aug.random_affine_transform(self.dummy_image, self.dummy_label)
        self.assertAllEqual(trans_image, self.dummy_image)
        self.assertAllEqual(trans_label, self.dummy_label)

        # The image and label should change after the augmentation
        trans_image, trans_label = seg_aug.random_affine_transform(self.dummy_image, self.dummy_label,
                                                                   shear_min=-1.0, shear_max=1.0,
                                                                   rotation_min=-1.0, rotation_max=1.0,
                                                                   zoom_min=0.1, zoom_max=10.0,
                                                                   rate_flip_lr=0.5, rate_flip_ud=0.5)
        self.assertNotAllEqual(trans_image, self.dummy_image)
        self.assertNotAllEqual(trans_image, self.dummy_label)

        # The image and label should remain compatible after the transformation
        trans_label1, trans_label2 = seg_aug.random_affine_transform(self.dummy_label, self.dummy_label,
                                                                     shear_min=-1.0, shear_max=1.0,
                                                                     rotation_min=-1.0, rotation_max=1.0,
                                                                     zoom_min=0.1, zoom_max=10.0,
                                                                     rate_flip_lr=0.5, rate_flip_ud=0.5)
        self.assertAllEqual(trans_label1, trans_label2)
        # The transformation should change the image
        self.assertNotAllEqual(trans_label1, self.dummy_label)

    def test_elastic_deformation(self):
        # The image and label should remain compatible after the transformation
        SIGMA = 3
        ALPHA = 0.1
        trans_label1, trans_label2 = seg_aug.elastic_deformation(self.dummy_label, self.dummy_label, SIGMA, ALPHA)
        self.assertAllEqual(trans_label1, trans_label2)

        # The transformation should change the image
        self.assertNotAllEqual(trans_label1, self.dummy_label)



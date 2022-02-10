import tensorflow as tf
from tf_image_augmentations import affine, utils


class TestAffine(tf.test.TestCase):
    def setUp(self) -> None:
        self.image_dims = (256, 256, 4)
        self.dummy_image = tf.random.uniform(self.image_dims, minval=0.0, maxval=1.0)
        self.unit_matrix = tf.eye(3)

    def test_generate_shear_matrix(self):
        # assert zero shear is the unit transformation
        self.assertAllEqual(affine.generate_shear_matrix(0.0), self.unit_matrix)

    def test_generate_rotation_matrix(self):
        # assert zero rotation is the unit matrix
        self.assertAllEqual(affine.generate_rotation_matrix(0.0), self.unit_matrix)

    def test_generate_zoom_matrix(self):
        # assert x1 zoom is the unit matrix
        self.assertAllEqual(affine.generate_zoom_matrix(1.0, 1.0), self.unit_matrix)

    def test_linear_transform_coords(self):
        # The coords should stay the same after unit transformation
        old_coords, new_coords = affine.linear_transform_coords(self.image_dims, self.unit_matrix)
        self.assertAllEqual(old_coords, new_coords)

        # after non-unit transform the coords should have the same shape but different values
        old_coords, new_coords = affine.linear_transform_coords(self.image_dims, affine.generate_shear_matrix(0.1))
        self.assertShapeEqual(old_coords.numpy(), new_coords)
        self.assertNotAllEqual(old_coords, new_coords)

    def test_linear_transform_from_coords(self):
        # The image should stay the same if the coordinates are the same
        coords = utils.image_dims_to_coordinates(self.image_dims)
        trans_image = affine.linear_transform_from_coords(self.dummy_image, coords, coords)
        self.assertAllEqual(trans_image, self.dummy_image)

        # The image should change if the coordinates are different (but maintain shape)
        old_coords, new_coords = affine.linear_transform_coords(self.image_dims, affine.generate_shear_matrix(0.1))
        trans_image = affine.linear_transform_from_coords(self.dummy_image, old_coords, new_coords)
        self.assertShapeEqual(trans_image.numpy(), self.dummy_image)
        self.assertNotAllEqual(trans_image, self.dummy_image)

    def test_linear_transform_image(self):
        # The image should stay the same with the default args
        trans_image = affine.linear_transform_image(self.dummy_image)
        self.assertAllEqual(trans_image, self.dummy_image)

        # The image should change if the transform is non-unit (but maintain shape)
        trans_image = affine.linear_transform_image(self.dummy_image, shear_factor=0.1)
        self.assertShapeEqual(trans_image.numpy(), self.dummy_image)
        self.assertNotAllEqual(trans_image, self.dummy_image)

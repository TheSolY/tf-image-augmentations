import tensorflow as tf
import tensorflow_addons as tfa
from tf_image_augmentations import utils


def generate_random_elastic_flow(img_size, elasticity_coefficient, deformation_intensity):
    """
    Generates a random flow field for elastic deformation
    img_size: height, width, elasticity_coefficient = sigma, deformation_intensity = alpha
    """
    img_size = utils.image_shape_to_hw(img_size)
    y__x__d = tf.random.uniform(tf.concat([img_size, [2]], axis=0), minval=-1, maxval=1)
    y__x__g = tfa.image.gaussian_filter2d(y__x__d, sigma=elasticity_coefficient, filter_shape=elasticity_coefficient)
    elastic_flow = deformation_intensity * y__x__g
    return elastic_flow


def warp_image_by_flow(img, flow):
    """
    Compatible with TF data pipeline when an explicit image size is given
    img: HWC, flow: HW2
    """
    img_shape = tf.shape(img)
    tf.assert_equal(len(img_shape), 3, 'Expected image format HWC (3D)')

    height, width, channels = tf.unstack(img_shape)
    grid_x, grid_y = tf.meshgrid(tf.range(width), tf.range(height))
    stacked_grid = tf.cast(tf.stack([grid_y, grid_x], axis=2), flow.dtype)
    batched_grid = tf.expand_dims(stacked_grid, axis=0)
    query_points_on_grid = batched_grid - flow
    query_points_flattened = tf.reshape(
        query_points_on_grid, [1, height * width, 2]
    )
    interpolated = tfa.image.interpolate_bilinear(tf.cast(utils.hwc_to_bhwc(img), tf.float32), query_points_flattened)
    interpolated = tf.cast(interpolated, img.dtype)
    warped = tf.reshape(interpolated, [height, width, channels])
    return warped


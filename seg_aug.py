import tensorflow as tf
from tf_image_augmentations import utils, affine, elastic

""" The functions in this model are performed on both the image and the mask.
Functions to preform random segmentation augmentations maintaining compatability between the image and the mask"""


def random_affine_transform(inputs, labels,
                            rotation_min=0.0, rotation_max=0.0,
                            shear_min=0.0, shear_max=0.0,
                            zoom_min=1.0, zoom_max=1.0,
                            rate_flip_lr=0.0, rate_flip_ud=0.0):
    """
    Apply random affine transformations for data augmentation
    rotation in radians
    shear and zoom relative to the image size
    rate flip_lr, flip_ud, which fraction of the images to flip: 0.0 never, 1.0 always
   """

    tf.assert_rank(inputs, 3, 'expected image format HWC (3D)')
    tf.assert_rank(labels, 3, 'expected label format HWC (3D)')

    flag_flip_lr = tf.random.uniform(shape=[], minval=0.0, maxval=1.0) < rate_flip_lr
    flag_flip_ud = tf.random.uniform(shape=[], minval=0.0, maxval=1.0) < rate_flip_ud

    rot_angle = tf.random.uniform(shape=[], minval=rotation_min, maxval=rotation_max)
    shear_factor = tf.random.uniform(shape=[], minval=shear_min, maxval=shear_max)
    zoom_h, zoom_w = tf.unstack(tf.random.uniform(shape=(2,), minval=zoom_min, maxval=zoom_max))

    if flag_flip_lr:
        inputs = tf.image.flip_left_right(inputs)
        labels = tf.image.flip_left_right(labels)

    if flag_flip_ud:
        inputs = tf.image.flip_up_down(inputs)
        labels = tf.image.flip_up_down(labels)

    rot_mat = affine.generate_rotation_matrix(rot_angle)
    shear_mat = affine.generate_shear_matrix(shear_factor)
    zoom_mat = affine.generate_zoom_matrix(zoom_h, zoom_w)
    trans_mat = rot_mat @ shear_mat @ zoom_mat

    org_coords_inputs, new_coords_inputs = affine.linear_transform_coords(tf.shape(inputs), trans_mat)
    org_coords_labels, new_coords_labels = affine.linear_transform_coords(tf.shape(labels), trans_mat)

    inputs = affine.linear_transform_from_coords(inputs, org_coords_inputs, new_coords_inputs)
    labels = affine.linear_transform_from_coords(labels, org_coords_labels, new_coords_labels)
    return inputs, labels


def random_affine_transform_fcn(rotation_min, rotation_max,
                                shear_min, shear_max,
                                zoom_min, zoom_max,
                                rate_flip_lr, rate_flip_ud):
    """Function closure for TF dataset map"""
    def transform_fcn(input_image, binary_mask):
        return random_affine_transform(input_image, binary_mask,
                                       rotation_min, rotation_max,
                                       shear_min, shear_max,
                                       zoom_min, zoom_max,
                                       rate_flip_lr, rate_flip_ud
                                       )
    return transform_fcn


def elastic_deformation(images, labels, elasticity_coefficient, deformation_intensity):
    """
    Deforms a batch of images and pixel labels by a random elastic transformation
    Image format: HWC
    """
    img_hw = utils.image_shape_to_hw(tf.shape(images))

    elastic_flow = elastic.generate_random_elastic_flow(img_hw, elasticity_coefficient, deformation_intensity)

    deformed_image = elastic.warp_image_by_flow(images, elastic_flow)
    deformed_label = elastic.warp_image_by_flow(labels, elastic_flow)

    return deformed_image, deformed_label


def elastic_augmentation_fcn(sigma, alpha):
    """Closure for TF dataset map """
    def elastic_augmentation(input_image, binary_mask):
        aug_img, aug_lbl = elastic_deformation(input_image, binary_mask, sigma, alpha)
        return aug_img, aug_lbl
    return elastic_augmentation

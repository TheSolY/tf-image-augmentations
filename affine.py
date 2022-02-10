import tensorflow as tf
from tf_image_augmentations import utils
"""Affine transformations"""


def generate_shear_matrix(shear_level):
    """Generates the shear x matrix to multiply the coordinates by
    shear_level in radians"""
    shear_matrix = tf.convert_to_tensor([[1.0, shear_level, 0.0],
                                         [shear_level, 1.0, 0.0],
                                         [0.0, 0.0, 1.0]], dtype=tf.float32)
    return shear_matrix


def generate_rotation_matrix(rotation_angle):
    """Generates the rotation matrix to multiply the coordinates by
    angle in radians"""
    rot_mat = tf.convert_to_tensor([[tf.cos(rotation_angle), -tf.sin(rotation_angle), 0.],
                                   [tf.sin(rotation_angle), tf.cos(rotation_angle), 0.],
                                   [0., 0., 1.]], dtype=tf.float32)
    return rot_mat


def generate_zoom_matrix(zoom_h, zoom_w):
    """Generates the stretch matrix to multiply the coordinates by
    Stretch relative to the original size
    for shrink (zoom out) use 0<zoom<1, for stretch (zoom in) use 1<zoom"""

    stretch_mat = tf.convert_to_tensor([[zoom_h, 0.0, 0.0],
                                         [0.0, zoom_w, 0.0],
                                         [0.0, 0.0, 1.0]], dtype=tf.float32)
    return stretch_mat


def linear_transform_coords(img_dims, trans_mat):
    """Calculate the old and new pixel coordinates, image format HWC"""
    hwc__img_center = tf.convert_to_tensor([img_dims[0] / 2, img_dims[1] / 2, 0.0], dtype=tf.float32)
    hwc__sample__new_coords = utils.image_dims_to_coordinates(img_dims)
    hwc__sample__new_coords_cent = tf.cast(hwc__sample__new_coords, tf.float32) - tf.expand_dims(hwc__img_center, 1)
    inv_rot_mat = tf.linalg.inv(trans_mat)
    hwc__sample__org_coords_cent = inv_rot_mat @ hwc__sample__new_coords_cent
    hwc__sample__org_coords = hwc__sample__org_coords_cent + tf.expand_dims(hwc__img_center, 1)

    hwc__sample__trim_mask = (hwc__sample__org_coords < tf.cast(tf.expand_dims(img_dims, -1), tf.float32)) &\
                             (hwc__sample__org_coords >= [[0.0], [0.0], [0.0]])
    sample__trim_mask = hwc__sample__trim_mask[0, :] & hwc__sample__trim_mask[1, :] & hwc__sample__trim_mask[2, :]

    hwc__sample__org_coords_trimmed = tf.boolean_mask(hwc__sample__org_coords, sample__trim_mask, axis=1)
    hwc__sample__org_coords_trimmed = tf.cast(hwc__sample__org_coords_trimmed, tf.int32)
    hwc__sample__new_coords_trimmed = tf.boolean_mask(hwc__sample__new_coords, sample__trim_mask, axis=1)
    return hwc__sample__org_coords_trimmed, hwc__sample__new_coords_trimmed


def linear_transform_from_coords(org_img, org_pixel_coords, new_pixel_coords, fill_value=0.0):
    img_dims = tf.shape(org_img)
    new_img = tf.fill(img_dims, fill_value)
    new_img = tf.tensor_scatter_nd_update(new_img,
                                          tf.transpose(new_pixel_coords),
                                          tf.gather_nd(tf.cast(org_img, tf.float32), tf.transpose(org_pixel_coords)))
    new_img = tf.cast(new_img, org_img.dtype)
    return new_img


def linear_transform_image(org_img, rotation_angle=0.0, shear_factor=0.0, zoom_h=1.0, zoom_w=1.0, fill_value=0.0):
    """Applies liner transformation to an image"""
    rot_mat = generate_rotation_matrix(rotation_angle)
    shear_mat = generate_shear_matrix(shear_factor)
    stretch_mat = generate_zoom_matrix(zoom_h, zoom_w)
    trans_mat = rot_mat @ shear_mat @ stretch_mat
    img_dims = tf.shape(org_img)
    org_pixel_coords, new_pixel_coords = linear_transform_coords(img_dims, trans_mat)
    new_img = linear_transform_from_coords(org_img, org_pixel_coords, new_pixel_coords, fill_value)
    return new_img


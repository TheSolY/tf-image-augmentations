import tensorflow as tf

"""Utility functions for the augmentations, mostly related dimensions and coordinates"""


def image_shape_to_hw(image_shape):
    """Extract the height and width of the image from HW, HWC or BHWC"""
    img_size = (-1, -1)
    if len(image_shape) == 2:
        img_size = image_shape
    elif len(image_shape) == 3:
        img_size = image_shape[:2]
    elif len(image_shape) == 4:
        img_size = image_shape[1:3]
    else:
        ValueError('image dimensions must be 2, 3, or 4')
    return img_size


def hwc_to_bhwc(img):
    """Transforms an image of HWC to BHWC for tf.image compatability"""
    tf.assert_rank(img, 3, 'expected image format HWC (3D)')
    return tf.expand_dims(img, 0)


def bhwc_to_hwc(img):
    """Transform an image of 1HWC to HWC"""
    tf.assert_rank(img, 4, 'expected image format BHWC (4D)')
    return tf.squeeze(img, 0)


def image_dims_to_coordinates(img_dims):
    """Generates a matrix of coordinates (h, w, c) x pixel index from the image dimensions"""
    h, w, c = tf.unstack(img_dims)
    H, W, C = tf.meshgrid(tf.range(h), tf.range(w), tf.range(c))
    hwc__sample__coords = tf.concat([tf.reshape(H, (1, -1)), tf.reshape(W, (1, -1)), tf.reshape(C, (1, -1))], 0)
    return hwc__sample__coords


def image_to_coordinates(image):
    """Transforms an image to a grid of its coordinates"""
    tf.assert_rank(image, 3, 'expected image format HWC (3D)')
    return image_dims_to_coordinates(tf.shape(image))


import tensorflow as tf

"""Functions to apply to the binary mask"""

def tight_box_coordinates(binary_mask):
    """Calculates the minimal encapsulating rectangle from a binary mask"""
    binary_mask = tf.squeeze(binary_mask)
    tf.assert_rank(binary_mask, 2, 'expected a 2D binary mask')
    binary_mask_shape = tf.shape(binary_mask, out_type=tf.int64)
    x_idxs = tf.where(tf.reduce_any(binary_mask > 0, axis=0))
    y_idxs = tf.where(tf.reduce_any(binary_mask > 0, axis=1))
    min_x = tf.reduce_min(x_idxs) / (binary_mask_shape[1] - 1)
    max_x = tf.reduce_max(x_idxs) / (binary_mask_shape[1] - 1)
    min_y = tf.reduce_min(y_idxs) / (binary_mask_shape[0] - 1)
    max_y = tf.reduce_max(y_idxs) / (binary_mask_shape[0] - 1)
    boxes = tf.convert_to_tensor([min_y, min_x, max_y, max_x], dtype=tf.float32)
    return boxes


def loose_box_coordinates(binary_mask, margin):
    """Margin to be added from each size- number in [0, 1] relative to the image height/width"""
    min_y, min_x, max_y, max_x = tf.unstack(tight_box_coordinates(binary_mask))
    new_min_y = tf.maximum(min_y - margin, 0)
    new_min_x = tf.maximum(min_x - margin, 0)
    new_max_y = tf.minimum(max_y + margin, 1)
    new_max_x = tf.minimum(max_x + margin, 1)
    boxes = tf.convert_to_tensor([new_min_y, new_min_x, new_max_y, new_max_x], dtype=tf.float32)
    return boxes


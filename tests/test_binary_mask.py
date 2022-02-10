import tensorflow as tf
from tf_image_augmentations import binary_mask


class TestBinaryMask(tf.test.TestCase):
    def setUp(self):
        self.label_dims = (256, 256, 1)
        self.ymin = tf.random.uniform([], minval=0, maxval=127, dtype=tf.int32)
        self.xmin = tf.random.uniform([], minval=0, maxval=127, dtype=tf.int32)
        self.ymax = tf.random.uniform([], minval=128, maxval=255, dtype=tf.int32)
        self.xmax = tf.random.uniform([], minval=128, maxval=255, dtype=tf.int32)
        self.dummy_label = self.generate_dummy_label()
        self.ymin_rel = tf.cast(self.ymin / (self.label_dims[0] - 1), tf.float32)
        self.xmin_rel = tf.cast(self.xmin / (self.label_dims[1] - 1), tf.float32)
        self.ymax_rel = tf.cast(self.ymax / (self.label_dims[0] - 1), tf.float32)
        self.xmax_rel = tf.cast(self.xmax / (self.label_dims[1] - 1), tf.float32)

    def generate_dummy_label(self):
        box_mask_rows = tf.where(
            (tf.range(self.label_dims[0]) < self.ymax) & (tf.range(self.label_dims[0]) > self.ymin),
            1.0, 0.0)

        box_mask_cols = tf.where(
            (tf.range(self.label_dims[1]) < self.xmax) & (tf.range(self.label_dims[1]) > self.xmin),
            1.0, 0.0)

        box_mask = tf.tensordot(box_mask_rows, box_mask_cols, axes=0)
        full_label = tf.random.uniform(self.label_dims, maxval=1)

        dummy_label = full_label * tf.expand_dims(box_mask, -1)
        return dummy_label

    def test_tight_box_coordinates(self):
        box_coords = binary_mask.tight_box_coordinates(self.dummy_label)
        # Tight box coordinates should be at most the box (could be less due to randomness)
        y_min, x_min, y_max, x_max = tf.unstack(box_coords)
        self.assertGreaterEqual(y_min, self.ymin_rel)
        self.assertGreaterEqual(x_min, self.xmin_rel)
        self.assertLessEqual(y_max, self.ymax_rel)
        self.assertLessEqual(x_max, self.xmax_rel)

    def test_loose_box_coordinates(self):
        MARGIN = 0.01
        box_coords = binary_mask.loose_box_coordinates(self.dummy_label, MARGIN)
        # Loose box coordinates should be at most the box + margin (could be less due to randomness)
        y_min, x_min, y_max, x_max = tf.unstack(box_coords)
        self.assertGreaterEqual(y_min, self.ymin_rel - MARGIN)
        self.assertGreaterEqual(x_min, self.xmin_rel - MARGIN)
        self.assertLessEqual(y_max, self.ymax_rel + MARGIN)
        self.assertLessEqual(x_max, self.xmax_rel + MARGIN)




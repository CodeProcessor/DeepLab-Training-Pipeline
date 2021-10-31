import tensorflow as tf

from deeplab.params import IGNORED_CLASS_ID


class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
    """
    Updated MIOU calculation with the addition of ignored classes.
    """

    def __init__(self,
                 y_true=None,
                 y_pred=None,
                 num_classes=None,
                 name=None,
                 dtype=None):
        super(UpdatedMeanIoU, self).__init__(num_classes=num_classes, name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis=-1)
        mask = y_true == IGNORED_CLASS_ID
        sample_weight = tf.where(mask, 0, 1)
        return super().update_state(y_true, y_pred, sample_weight=sample_weight)

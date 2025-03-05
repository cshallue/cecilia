import tensorflow as tf
from tensorflow import keras


def weighted_mean_squared_error(y_true, y_pred, class_weights=None):
  y_pred = tf.convert_to_tensor(y_pred)
  y_true = tf.convert_to_tensor(y_true, dtype=y_pred.dtype)
  losses = tf.square(y_true - y_pred)
  if class_weights is not None:
    class_weights = tf.convert_to_tensor(class_weights, dtype=y_pred.dtype)
    losses *= class_weights

  # Note that we take a mean, not a sum, after applying class weights.
  # This means that class_weights = 1 is equivalent to taking the mean
  # over classes (as opposed to class_weights = 1 / num_classes).
  return tf.reduce_mean(losses, axis=-1)


class WeightedMeanSquaredError(keras.losses.Loss):

  def __init__(self,
               class_weights=None,
               reduction="sum_over_batch_size",
               name="weighted_mse"):
    super().__init__(reduction=reduction, name=name)
    self.class_weights = class_weights

  def call(self, y_true, y_pred):
    return weighted_mean_squared_error(y_true,
                                       y_pred,
                                       class_weights=self.class_weights)

import tensorflow as tf
from tensorflow import keras


def _convert_to_tensors(y_true, y_pred):
  y_pred = tf.convert_to_tensor(y_pred)
  y_true = tf.convert_to_tensor(y_true, dtype=y_pred.dtype)
  return y_true, y_pred


# ==== Unweighted loss functions ====


def squared_error(y_true, y_pred):
  y_true, y_pred = _convert_to_tensors(y_true, y_pred)
  return tf.square(y_true - y_pred)


def squared_log_error(y_true, y_pred):
  y_true, y_pred = _convert_to_tensors(y_true, y_pred)
  return tf.square(tf.math.log(y_true) - tf.math.log(y_pred))


def _mean_error_fn(error_fn):

  def calc_mean_error(y_true, y_pred):
    # Mean over the classes dimension (not the batch dimension).
    return tf.reduce_mean(error_fn(y_true, y_pred), axis=-1)

  return calc_mean_error


mean_squared_error = _mean_error_fn(squared_error)
mean_squared_log_error = _mean_error_fn(squared_log_error)

MeanSquaredError = keras.losses.MeanSquaredError  # Alias


class MeanSquaredLogError(keras.losses.Loss):

  def call(self, y_true, y_pred):
    return mean_squared_log_error(y_true, y_pred)


# ==== Weighted loss functions ====


def _weighted_mean_error_fn(error_fn):

  def calc_weighted_mean_error(y_true, y_pred, class_weights=None):
    losses = error_fn(y_true, y_pred)
    if class_weights is not None:
      class_weights = tf.convert_to_tensor(class_weights, dtype=y_pred.dtype)
      losses *= class_weights

    # Note that we take a mean, not a sum, after applying class weights.
    # This means that class_weights = 1 is equivalent to taking the mean
    # over classes (as opposed to class_weights = 1 / num_classes).
    return tf.reduce_mean(losses, axis=-1)

  return calc_weighted_mean_error


weighted_mean_squared_error = _weighted_mean_error_fn(squared_error)
weighted_mean_squared_log_error = _weighted_mean_error_fn(squared_log_error)


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


class WeightedMeanSquaredLogError(keras.losses.Loss):

  def __init__(self,
               class_weights=None,
               reduction="sum_over_batch_size",
               name="weighted_mse"):
    super().__init__(reduction=reduction, name=name)
    self.class_weights = class_weights

  def call(self, y_true, y_pred):
    return weighted_mean_squared_log_error(y_true,
                                           y_pred,
                                           class_weights=self.class_weights)

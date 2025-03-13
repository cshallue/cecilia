import tensorflow as tf
from tensorflow import keras

from cecilia import distributions


def _convert_to_tensors(y_true, y_pred):
  y_pred = tf.convert_to_tensor(y_pred)
  y_true = tf.convert_to_tensor(y_true, dtype=y_pred.dtype)
  return y_true, y_pred


# Transforms per-class losses to rescale * losses + shift before taking the mean
# over the class dimension to get the total loss.
class ScaledShiftedLossFn(keras.losses.Loss):

  def __init__(self, loss_fn, rescale=None, shift=None, **kwargs):
    super().__init__(**kwargs)
    self.loss_fn = loss_fn
    self.rescale = rescale
    self.shift = shift

  def call(self, y_true, y_pred):
    losses = self.loss_fn(y_true, y_pred)
    if self.rescale is not None:
      rescale = tf.convert_to_tensor(self.rescale, dtype=y_pred.dtype)
      losses *= rescale
    if self.shift is not None:
      shift = tf.convert_to_tensor(self.shift, dtype=y_pred.dtype)
      losses += shift

    return tf.reduce_mean(losses, axis=-1)


# ==== Log likelihood loss function ====


def log_likelihood(y_true, y_pred):
  dist = distributions.to_tensorflow_distribution(y_pred)
  return -dist.log_prob(y_true)


class LogLikelihood(ScaledShiftedLossFn):

  def __init__(self, **kwargs):
    super().__init__(log_likelihood, **kwargs)


# ==== Error-based loss functions ====


def squared_error(y_true, y_pred):
  y_true, y_pred = _convert_to_tensors(y_true, y_pred)
  return tf.square(y_true - y_pred)


def squared_log_error(y_true, y_pred):
  y_true, y_pred = _convert_to_tensors(y_true, y_pred)
  return tf.square(tf.math.log(y_true) - tf.math.log(y_pred))


def absolute_error(y_true, y_pred):
  y_true, y_pred = _convert_to_tensors(y_true, y_pred)
  return tf.abs(y_true - y_pred)


def relative_error(y_true, y_pred):
  y_true, y_pred = _convert_to_tensors(y_true, y_pred)
  return tf.abs(y_pred / y_true - 1)


class MeanSquaredError(ScaledShiftedLossFn):

  def __init__(self, **kwargs):
    super().__init__(squared_error, **kwargs)


class MeanSquaredLogError(ScaledShiftedLossFn):

  def __init__(self, **kwargs):
    super().__init__(squared_log_error, **kwargs)


class MeanAbsoluteError(keras.losses.Loss):

  def __init__(self, **kwargs):
    super().__init__(absolute_error, **kwargs)


class MeanRelativeError(keras.losses.Loss):

  def __init__(self, **kwargs):
    super().__init__(relative_error, **kwargs)

""" Similar to scipy preprocessing pipeline, but implemented in TensorFlow."""

import tensorflow as tf


class Transformer:

  def __call__(self, data):
    return self.transform(data)

  def fit(self, data):
    pass

  def transform(self, data):
    raise NotImplementedError

  def inverse_transform(self, data):
    raise NotImplementedError

  def fit_transform(self, data):
    self.fit(data)
    return self.transform(data)


class IdentityTransformer(Transformer):

  def transform(self, data):
    return data

  def inverse_transform(self, data):
    return data


class LogTransformer(Transformer):

  def transform(self, data):
    return tf.math.log(data)

  def inverse_transform(self, data):
    return tf.math.exp(data)


class Normalizer(Transformer):

  def __init__(self):
    super().__init__()
    self.mean = None
    self.std = None

  def fit(self, data):
    self.mean = tf.reduce_mean(data, axis=0, keepdims=True)
    self.var = tf.reduce_mean((data - self.mean)**2, axis=0, keepdims=True)

  def transform(self, data):
    if self.mean is None:
      raise ValueError("fit() must be called before transform()")

    return (data - self.mean) / tf.sqrt(self.var)

  def inverse_transform(self, data):
    if self.mean is None:
      raise ValueError("fit() must be called before inverse_transform()")

    return data * tf.sqrt(self.var) + self.mean


class TransformerPipeline(Transformer):

  def __init__(self, layers):
    super().__init__()
    self._layers = layers

  def fit(self, data):
    for layer in self._layers:
      data = layer.fit_transform(data)

  def transform(self, data):
    for layer in self._layers:
      data = layer.transform(data)
    return data

  def inverse_transform(self, data):
    for layer in self._layers[::-1]:
      data = layer.inverse_transform(data)
    return data


def create_scalers(log_transform_y, normalize_y):
  x_scaler = Normalizer()

  y_scalers = []
  if log_transform_y:
    y_scalers.append(LogTransformer())
  if normalize_y:
    y_scalers.append(Normalizer())

  if len(y_scalers) == 0:
    y_scaler = IdentityTransformer()
  elif len(y_scalers) == 1:
    y_scaler = y_scalers[0]
  else:
    y_scaler = TransformerPipeline(y_scalers)

  return x_scaler, y_scaler

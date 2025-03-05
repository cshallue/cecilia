""" Similar to scipy preprocessing pipeline, but implemented in TensorFlow."""

import tensorflow as tf
from tensorflow import keras


# The invert parameter only changes the behavior of call().
# It does not affect transform() or inverse_transform().
class Transformer(keras.Layer):

  def __init__(self, invert=False):
    super().__init__()
    self.invert = invert

  def fit(self, data):
    pass

  @property
  def is_fit(self):
    return True  # Default fit() is no-op.

  def transform(self, data):
    raise NotImplementedError

  def inverse_transform(self, data):
    raise NotImplementedError

  def call(self, data):
    if self.invert:
      return self.inverse_transform(data)

    return self.transform(data)

  def fit_transform(self, data):
    self.fit(data)
    return self.transform(data)


class LogTransformer(Transformer):

  def transform(self, data):
    return tf.math.log(data)

  def inverse_transform(self, data):
    return tf.math.exp(data)


class Normalizer(Transformer):

  def __init__(self, invert=False):
    super().__init__(invert)
    # The Keras Normalization class also has an invert option, but we will set
    # it explicitly each time we call the layer, so we do not set it here.
    self._norm_layer = keras.layers.Normalization()
    self._is_fit = False

  @property
  def mean(self):
    return self._norm_layer.mean

  @property
  def variance(self):
    return self._norm_layer.variance

  def build(self, input_shape):
    self._norm_layer.build(input_shape)

  def fit(self, data):
    self._norm_layer.adapt(data)
    self._is_fit = True

  @property
  def is_fit(self):
    return self._is_fit

  def transform(self, data):
    self._norm_layer.invert = False
    return self._norm_layer(data)

  def inverse_transform(self, data):
    self._norm_layer.invert = True
    return self._norm_layer(data)


class TransformerPipeline(Transformer):

  def __init__(self, layers, invert=False):
    super().__init__(invert)
    self.layers = layers

  def build(self, input_shape):
    for layer in self.layers:
      # Transformers do not change the input shape.
      layer.build(input_shape)

  def fit(self, data):
    for layer in self.layers:
      data = layer.fit_transform(data)

  @property
  def is_fit(self):
    for layer in self.layers:
      if not layer.is_fit:
        return False
    return True

  def transform(self, data):
    for layer in self.layers:
      data = layer.transform(data)
    return data

  def inverse_transform(self, data):
    for layer in self.layers[::-1]:
      data = layer.inverse_transform(data)
    return data


def create_pipeline(log_transform=False, normalize=True, invert=False):
  transformers = []
  if log_transform:
    transformers.append(LogTransformer())
  if normalize:
    transformers.append(Normalizer())

  return TransformerPipeline(transformers, invert=invert)

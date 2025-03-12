"""Preprocessing pipelines."""

import tensorflow as tf
from tensorflow import keras

from cecilia import distributions


def transform_dist(dist, old_name, new_name):
  distributions.validate(dist, old_name)
  return {
      f"{new_name}_{param}": dist[f"{old_name}_{param}"]
      for param in ["loc", "scale"]
  }


# The invert parameter only changes the behavior of call().
# It does not affect transform() or inverse_transform().
class Transformer(keras.layers.Layer):

  def __init__(self, invert=False, **kwargs):
    if kwargs.get("trainable"):
      raise ValueError("Transformer layers are not trainable!")
    kwargs["trainable"] = False
    super().__init__(**kwargs)
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

  def get_config(self):
    config = super().get_config()
    config.update({
        "invert": self.invert,
    })
    return config


@keras.utils.register_keras_serializable(package="transformers")
class LogTransformer(Transformer):

  def transform(self, data):
    if distributions.is_distribution(data):
      return transform_dist(data, "LogNormal", "Normal")  # LogNormal -> Normal

    return tf.math.log(data)

  def inverse_transform(self, data):
    if distributions.is_distribution(data):
      return transform_dist(data, "Normal", "LogNormal")  # Normal -> LogNormal

    return tf.math.exp(data)


@keras.utils.register_keras_serializable(package="transformers")
class Normalizer(Transformer):

  def __init__(self, invert=False, **kwargs):
    super().__init__(invert, **kwargs)
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
    if distributions.is_distribution(input_shape):
      input_shape = input_shape["loc"]
    self._norm_layer.build(input_shape)

  def fit(self, data):
    self._norm_layer.adapt(data)
    self._is_fit = True

  @property
  def is_fit(self):
    return self._is_fit

  def transform(self, data):
    self._norm_layer.invert = False
    if distributions.is_distribution(data):
      data = transform_dist(data, "Normal", "Normal")
      data["Normal_loc"] = self._norm_layer(data["Normal_loc"])
      data["Normal_scale"] = data["Normal_scale"] / tf.sqrt(self.variance)
      return data

    return self._norm_layer(data)

  def inverse_transform(self, data):
    self._norm_layer.invert = True
    if distributions.is_distribution(data):
      data = transform_dist(data, "Normal", "Normal")
      data["Normal_loc"] = self._norm_layer(data["Normal_loc"])
      data["Normal_scale"] = data["Normal_scale"] * tf.sqrt(self.variance)
      return data

    return self._norm_layer(data)


@keras.utils.register_keras_serializable(package="transformers")
class TransformerPipeline(Transformer):

  def __init__(self, layers, invert=False, **kwargs):
    super().__init__(invert, **kwargs)
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

  def get_config(self):
    config = super().get_config()
    config.update({
        "layers":
        [keras.utils.serialize_keras_object(layer) for layer in self.layers],
    })
    return config

  @classmethod
  def from_config(cls, config):
    layer_configs = config.pop("layers")
    layers = [
        keras.utils.deserialize_keras_object(layer_config)
        for layer_config in layer_configs
    ]
    return cls(layers, **config)


def create_pipeline(log_transform=False, normalize=True, invert=False):
  transformers = []
  if log_transform:
    transformers.append(LogTransformer())
  if normalize:
    transformers.append(Normalizer())

  return TransformerPipeline(transformers, invert=invert)

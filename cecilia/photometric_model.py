import json
import os

import ml_collections
import tensorflow as tf
from tensorflow import keras

from cecilia import distributions, losses
from cecilia.preprocessing import transformers


def _broadcast_scale(scale, loc):
  if tf.is_symbolic_tensor(loc):
    # Placeholder used when building the model. Return a placeholder with the
    # same shape and dtype.
    return keras.KerasTensor(shape=loc.shape, dtype=loc.dtype)

  return tf.broadcast_to(scale, loc.shape)


class ConstantScale(keras.layers.Layer):

  def __init__(self, scale, **kwargs):
    super().__init__(**kwargs)
    self.scale = tf.convert_to_tensor(scale, dtype=tf.float32)

  def call(self, loc):
    scale = _broadcast_scale(self.scale, loc)
    return {"Normal_loc": loc, "Normal_scale": scale}


class LearnedScale(keras.layers.Layer):

  def build(self, input_shape):
    input_dim = input_shape[-1]
    # scale is softplus(_scale_params).
    self._scale_params = self.add_weight(
        name="scale_params",
        shape=(input_dim, ),
        initializer=keras.initializers.Zeros(),
    )

  def call(self, loc):
    scale = _broadcast_scale(tf.math.softplus(self._scale_params), loc)
    return {"Normal_loc": loc, "Normal_scale": scale}


class DenseLocScale(keras.layers.Layer):

  def __init__(self, output_dim, **kwargs):
    super().__init__(**kwargs)
    self._dense = keras.layers.Dense(2 * output_dim)

  def build(self, input_shape):
    self._dense.build(input_shape)

  def call(self, input):
    loc_scale = self._dense.call(input)
    loc, scale = tf.split(loc_scale, 2, axis=-1)
    scale = tf.math.softplus(scale)
    return {"Normal_loc": loc, "Normal_scale": scale}


class PhotometricModel(keras.Model):

  def __init__(self, config, **kwargs):
    super().__init__(**kwargs)
    # Input transformer.
    x_transformer = transformers.create_pipeline(normalize=config.normalize_x)

    # Fully connected layers.
    fc_layers = []
    for _ in range(config.n_hidden):
      fc_layers.append(
          keras.layers.Dense(config.dim_hidden, config.hidden_activation))

    if config.predict_std_method == "dense":
      fc_layers.append(DenseLocScale(config.dim_output))
    else:
      fc_layers.append(keras.layers.Dense(config.dim_output))
      if config.predict_std_method == "constant":
        fc_layers.append(ConstantScale(config.constant_std))
      elif config.predict_std_method == "per_class":
        fc_layers.append(LearnedScale())
      elif config.predict_std_method != "none":
        raise ValueError(config.predict_std_method)

    # Output transformer.
    y_transformer = transformers.create_pipeline(
        log_transform=config.log_transform_y,
        normalize=config.normalize_y,
        invert=True)

    self.x_transformer = x_transformer
    self.fc_layers = fc_layers
    self.y_transformer = y_transformer

    # We know the input dimension so we might as well build now.
    input_layer = keras.Input(shape=(config.dim_input, ))
    self.call(input_layer)

  def call(self, inputs):
    x = self.x_transformer(inputs)
    for layer in self.fc_layers:
      x = layer(x)
    return self.y_transformer(x)


def _get_loss_fn(model, config):

  predicts_distribution = (config.predict_std_method != "none")
  if predicts_distribution != (config.loss == "log_likelihood"):
    raise ValueError(
        "Must have loss='log_likelihood' iff predicting a distribution")

  # Log likelihood loss function.
  if config.loss == "log_likelihood":
    if config.log_transform_y:
      return losses.LogNormalLogLikelihood()
    return losses.NormalLogLikelihood()

  # Unweighted loss functions.

  if config.loss == "mean_squared_error":
    return losses.MeanSquaredError()

  if config.loss == "mean_squared_log_error":
    return losses.MeanSquaredLogError()

  if config.loss == "mean_absolute_error":
    return losses.MeanAbsoluteError()

  if config.loss == "mean_relative_error":
    return losses.MeanRelativeError()

  # Weighted loss functions.

  if not config.normalize_y:
    raise ValueError(f"loss='{config.loss}' requires normalize_y=True")

  if not model.y_transformer.is_fit:
    raise ValueError(
        "Must call model.y_transformer.fit() before creating weighted loss")

  y_normalizer = model.y_transformer.layers[-1]
  assert isinstance(y_normalizer, transformers.Normalizer)
  class_weights = tf.divide(1.0, y_normalizer.variance)

  if config.loss == "weighted_mean_squared_error":
    return losses.WeightedMeanSquaredError(class_weights)

  if config.loss == "weighted_mean_squared_log_error":
    return losses.WeightedMeanSquaredLogError(class_weights)

  raise ValueError(f"Unrecognized loss function: {config.loss}")


def compile(model, config):
  lr = config.learning_rate
  momentum = config.momentum
  optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
  loss_fn = _get_loss_fn(model, config)
  model.compile(optimizer=optimizer, loss=loss_fn)


def load(model_dir):
  with open(os.path.join(model_dir, "config.json"), "r") as f:
    config = ml_collections.ConfigDict(json.load(f))

  model = PhotometricModel(config)

  model_filename = os.path.join(model_dir, "model.keras")
  model.load_weights(model_filename)
  print(f"Loaded weights from {model_filename}")

  return config, model

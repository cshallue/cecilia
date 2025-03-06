import json
import os

import ml_collections
import tensorflow as tf
from tensorflow import keras

from cecilia import losses
from cecilia.preprocessing import transformers


class PhotometricModel(keras.Sequential):

  def __init__(self, config):
    # Input and output transformers.
    x_transformer = transformers.create_pipeline(normalize=config.normalize_x)
    y_transformer = transformers.create_pipeline(
        log_transform=config.log_transform_y,
        normalize=config.normalize_y,
        invert=True)

    # Fully connected layers.
    fc_layers = []
    for _ in range(config.n_hidden):
      fc_layers.append(
          keras.layers.Dense(config.dim_hidden, config.hidden_activation))
    fc_layers.append(keras.layers.Dense(config.dim_output))

    self.x_transformer = x_transformer
    self.y_transformer = y_transformer
    self.fc_layers = fc_layers

    input_layer = keras.Input(shape=(config.dim_input, ))
    all_layers = [input_layer, x_transformer] + fc_layers + [y_transformer]
    super().__init__(all_layers)


def _get_loss_fn(model, config):
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

import json
import os

import ml_collections
import tensorflow as tf
from tensorflow import keras

from cecilia import losses
from cecilia.preprocessing import tf_transformers as transformers


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
  if config.normalize_y:
    loss_cls = (losses.WeightedMeanSquaredLogError
                if config.log_transform_y else losses.WeightedMeanSquaredError)
    if not model.y_transformer.is_fit:
      raise ValueError(
          "Must call model.y_transformer.fit() before creating loss function")
    class_weights = tf.divide(1.0, model.y_transformer.layers[-1].variance)
    return loss_cls(class_weights)

  loss_cls = (losses.MeanSquaredLogError
              if config.log_transform_y else losses.MeanSquaredError)
  return loss_cls()


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

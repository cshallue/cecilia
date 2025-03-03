import json
import os

import ml_collections
from tensorflow import keras


def build(config):
  layers = [keras.Input(shape=(config.dim_input, ))]
  for i in range(config.n_hidden):
    layers.append(
        keras.layers.Dense(config.dim_hidden, config.hidden_activation))
  layers.append(keras.layers.Dense(config.dim_output))
  return keras.Sequential(layers)


def compile(model, config):
  lr = config.learning_rate
  momentum = config.momentum
  optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
  model.compile(optimizer=optimizer, loss='mean_squared_error')


def build_and_compile(config):
  model = build(config)
  compile(model, config)
  return model


def load(model_dir):
  with open(os.path.join(model_dir, "config.json"), "r") as f:
    config = ml_collections.ConfigDict(json.load(f))

  model = build(config)

  model_filename = os.path.join(model_dir, "model.keras")
  model.load_weights(model_filename)
  print(f"Loaded weights from {model_filename}")

  return config, model

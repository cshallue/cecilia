import json
import os

import ml_collections
import tensorflow as tf
from tensorflow import keras

from cecilia import losses
from cecilia.preprocessing import transformers


class PhotometricModel(keras.Model):

  def __init__(self, config, **kwargs):
    super().__init__(**kwargs)
    self.config = config

    # Construct the layers.
    self.x_transformer = transformers.create_pipeline(
        normalize=config.normalize_x)
    self.hidden_layers = self.construct_hidden_layers()
    self.output_layers = self.construct_output_layers()
    self.y_transformer = transformers.create_pipeline(
        log_transform=config.log_transform_y,
        normalize=config.normalize_y,
        invert=True)

    # We know the input dimension so we might as well build now.
    self.build(input_shape=(None, config.dim_input))
    self.loss_fn = None

  def construct_hidden_layers(self):
    layers = []
    for _ in range(self.config.n_hidden):
      layers.append(
          keras.layers.Dense(self.config.dim_hidden,
                             self.config.hidden_activation))
    return layers

  def construct_output_layers(self):
    return [keras.layers.Dense(self.config.dim_output)]

  def build(self, input_shape):
    input_layer = keras.Input(shape=(input_shape[-1], ))
    self.call(input_layer)  # Builds all the layers.
    self.built = True

  def call(self, inputs):
    x = self.x_transformer(inputs)
    layers = self.hidden_layers + self.output_layers
    for layer in layers:
      x = layer(x)
    return self.y_transformer(x)

  def compute_loss(
      self,
      x=None,
      y=None,
      y_pred=None,
      sample_weight=None,
      training=True,
  ):
    # x and training arguments are unused.
    del x
    del training

    if self.loss_fn is None:
      raise ValueError("Must call compile() before compute_loss()")

    losses = [self.loss_fn(y, y_pred, sample_weight)]
    for loss in self.losses:
      losses.append(self._aggregate_additional_loss(loss))

    return losses[0] if len(losses) == 1 else tf.sum(losses)

  def _get_y_normalizer(self):
    y_tr_layers = self.y_transformer.layers
    if y_tr_layers and isinstance(y_tr_layers[-1], transformers.Normalizer):
      return y_tr_layers[-1]

  def _build_loss_fn(self):
    loss_name = self.config.loss

    # Loss rescaling.
    shift = self.config.loss_shift or None
    rescale = None
    y_normalizer = self._get_y_normalizer()
    if self.config.loss_rescaling_method == 'constant':
      rescale = self.config.loss_rescaling_value
    elif self.config.loss_rescaling_method == 'per_class_variance':
      if y_normalizer is None:
        raise ValueError("loss_rescaling_method='per_class_variance' requires "
                         "normalize_y=True")
      rescale = tf.divide(1.0, y_normalizer.variance)
    elif self.config.loss_rescaling_method != "none":
      raise ValueError(self.config.loss_rescaling_method)

    # Create loss function.

    if loss_name == "mean_squared_error":
      return losses.MeanSquaredError(rescale=rescale, shift=shift)

    if loss_name == "mean_squared_log_error":
      return losses.MeanSquaredLogError(rescale=rescale, shift=shift)

    if loss_name == "mean_absolute_error":
      return losses.MeanAbsoluteError(rescale=rescale, shift=shift)

    if loss_name == "mean_relative_error":
      return losses.MeanRelativeError(rescale=rescale, shift=shift)

    raise ValueError(f"Unrecognized loss function: {loss_name}")

  def compile(self, **kwargs):
    # Build the loss function
    self.loss_fn = self._build_loss_fn()

    # Build the optimizer.
    lr = self.config.learning_rate
    momentum = self.config.momentum
    optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
    super().compile(optimizer=optimizer, **kwargs)


def load(model_dir):
  with open(os.path.join(model_dir, "config.json"), "r") as f:
    config = ml_collections.ConfigDict(json.load(f))

  model = PhotometricModel(config)

  weights_filename = os.path.join(model_dir, "model.weights.h5")
  model.load_weights(weights_filename)
  print(f"Loaded weights from {weights_filename}")

  return config, model

import json
import math
import os

import ml_collections
import tensorflow as tf
from tensorflow import keras

from cecilia import losses
from cecilia.preprocessing import transformers


class ConstantScale(keras.layers.Layer):

  def __init__(self, scale, **kwargs):
    super().__init__(**kwargs)
    self.scale = tf.convert_to_tensor(scale, dtype=tf.float32)

  def call(self, loc):
    scale = tf.broadcast_to(self.scale, tf.shape(loc))
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
    scale = tf.math.softplus(self._scale_params)
    scale = tf.broadcast_to(scale, tf.shape(loc))
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
    self.config = config

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
        fc_layers.append(ConstantScale(config.predict_constant_std_value))
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
    self.build(input_shape=(None, config.dim_input))
    self.loss_fn = None

  def build(self, input_shape):
    input_layer = keras.Input(shape=(input_shape[-1], ))
    self.call(input_layer)  # Builds all the layers.
    self.built = True

  def call(self, inputs):
    x = self.x_transformer(inputs)
    for layer in self.fc_layers:
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
    predicts_distribution = (self.config.predict_std_method != "none")
    if predicts_distribution != (loss_name == "log_likelihood"):
      raise ValueError(
          "Must have loss='log_likelihood' iff predicting a distribution")

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
    elif self.config.loss_rescaling_method == 'log_likelihood_rescaling':
      # There is a factor of 1/2 in the log likelihood that is not included in
      # the squared error. So rescale by a factor of 2.
      rescale = 2.0
      # Shift away constant factors.
      if shift:
        raise ValueError("loss_shift cannot be used in conjunction with "
                         "log_likelihood_rescaling")
      shift = -tf.math.log(2 * math.pi)
      if y_normalizer is not None:
        shift -= tf.reduce_mean(tf.math.log(y_normalizer.variance))
    elif self.config.loss_rescaling_method != "none":
      raise ValueError(self.config.loss_rescaling_method)

    # Create loss function.

    if loss_name == "log_likelihood":
      return losses.LogLikelihood(rescale=rescale, shift=shift)

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

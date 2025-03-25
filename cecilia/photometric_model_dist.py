import math

import tensorflow as tf
from tensorflow import keras

from cecilia import losses, photometric_model


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


class PhotometricModelDist(photometric_model.PhotometricModel):

  def __init__(self, config, **kwargs):
    super().__init__(**kwargs)

  def construct_output_layers(self):
    layers = []
    if self.config.predict_std_method == "dense":
      layers.append(DenseLocScale(self.config.dim_output))
    else:
      layers.append(keras.layers.Dense(self.config.dim_output))
      if self.config.predict_std_method == "constant":
        layers.append(ConstantScale(self.config.predict_constant_std_value))
      elif self.config.predict_std_method == "per_class":
        layers.append(LearnedScale())
      else:
        raise ValueError(self.config.predict_std_method)
    return layers

  def _build_loss_fn(self):
    if self.config.loss != "log_likelihood":
      raise ValueError(self.config.loss)

    # Loss rescaling.
    shift = self.config.loss_shift or None
    rescale = None
    y_normalizer = self._get_y_normalizer()
    if self.config.loss_rescaling_method == 'constant':
      rescale = self.config.loss_rescaling_value
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

    return losses.LogLikelihood(rescale=rescale, shift=shift)

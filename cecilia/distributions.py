"""Utilities for dealing with probability distributions."""

import tensorflow as tf
from tensorflow import keras

# Enums that identify different distributions.
NORMAL = tf.constant(0, dtype=tf.int32)
LOG_NORMAL = tf.constant(1, dtype=tf.int32)


def is_distribution(dist):
  return isinstance(dist, dict) and "distribution" in dist


# Identify dummy distributions used when building model layers.
def is_dummy(dist):
  return isinstance(dist["distribution"], keras.KerasTensor)


def get_name(dist):
  if is_dummy(dist):
    return "Dummy"

  key = dist["distribution"]

  if key == NORMAL:
    return "Normal"

  if key == LOG_NORMAL:
    return "LogNormal"

  raise ValueError(f"Unrecognized distribution key: {key}")


def validate(dist):
  if set(dist.keys()) != {"distribution", "loc", "scale"}:
    raise ValueError(
        "Distribution keys should be {'distribution', 'loc', 'scale'}")

  _ = get_name(dist)  # Validate key.

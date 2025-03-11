"""Utilities for dealing with probability distributions."""

import tensorflow as tf

# Enums that identify different distributions.
NORMAL = tf.constant(0, dtype=tf.int32)
LOG_NORMAL = tf.constant(1, dtype=tf.int32)


def is_distribution(dist):
  return isinstance(dist, dict) and "distribution" in dist


# Identify symbolic distributions used when building model layers.
def is_symbolic(dist):
  return tf.is_symbolic_tensor(dist["distribution"])


def get_name(dist):
  if is_symbolic(dist):
    return "Symbolic"

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

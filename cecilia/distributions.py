"""Utilities for dealing with input and output distributions."""

# Enums that identify different distributions.
NORMAL = 0
LOG_NORMAL = 1


def get_name(dist_key):
  if dist_key == NORMAL:
    return "Normal"
  if dist_key == LOG_NORMAL:
    return "LogNormal"

  raise ValueError(f"Unrecognized distribution key: {dist_key}")


def is_distribution(dist):
  return isinstance(dist, dict) and "distribution" in dist


def validate(dist, copy=False):
  if set(dist.keys()) != {"distribution", "loc", "scale"}:
    raise ValueError("Keys should be 'distribution', 'loc', 'scale'")

  _ = get_name(dist["distribution"])  # Validate key.

  if copy:
    dist = dist.copy()  # Shallow copy

  return dist

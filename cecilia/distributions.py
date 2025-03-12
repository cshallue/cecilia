"""Utilities for dealing with probability distributions."""

from tensorflow_probability import distributions as tfd

NAMES = ["Normal", "LogNormal"]


def is_distribution(dist):
  if not isinstance(dist, dict):
    return False

  for key in dist:
    if key.endswith("_loc"):
      return True

  return False


def validate(dist, expected_name=None):
  dist_names = set()
  param_names = set()
  for key in dist:
    split_key = key.split("_")
    if len(split_key) != 2:
      raise ValueError("Distribution keys should have format NAME_PARAM")
    dist_names.add(split_key[0])
    param_names.add(split_key[1])

  if len(dist_names) != 1 or param_names != {"loc", "scale"}:
    raise ValueError("Distribution keys should be {NAME_loc, NAME_scale}")

  dist_name = dist_names.pop()
  if dist_name not in NAMES:
    raise ValueError(f"Unrecognized distribution name: {dist_name}")

  if expected_name and dist_name != expected_name:
    raise ValueError(
        f"Expected '{expected_name}' distribution, got '{dist_name}'")

  return dist_name


def to_tensorflow_distribution(dist):
  dist_name = validate(dist)

  if dist_name == "Normal":
    return tfd.Normal(loc=dist["Normal_loc"], scale=dist["Normal_scale"])

  if dist_name == "LogNormal":
    return tfd.LogNormal(loc=dist["LogNormal_loc"],
                         scale=dist["LogNormal_scale"])

  raise ValueError(dist_name)

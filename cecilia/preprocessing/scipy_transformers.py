""" Preprocessing pipeline using scipy. '"""

import numpy as np
import sklearn


def _identity(x):
  return x


def IdentityTransformer():
  return sklearn.preprocessing.FunctionTransformer(func=_identity,
                                                   inverse_func=_identity)


def LogTransformer():
  return sklearn.preprocessing.FunctionTransformer(
      func=np.log10, inverse_func=lambda x: np.power(10, x))


def create_scalers(log_transform_y, normalize_y):
  x_scaler = sklearn.preprocessing.StandardScaler()

  y_scalers = []
  if log_transform_y:
    y_scalers.append(("log", LogTransformer()))
  if normalize_y:
    y_scalers.append(("norm", sklearn.preprocessing.StandardScaler()))

  if len(y_scalers) == 0:
    y_scaler = IdentityTransformer()
  elif len(y_scalers) == 1:
    y_scaler = y_scalers[0][1]
  else:
    y_scaler = sklearn.pipeline.Pipeline(y_scalers)

  return x_scaler, y_scaler

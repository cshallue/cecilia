import numpy as np
import pandas as pd


def calc_mean_squared_error(Y, Y_pred, axis=0):
  return np.mean((Y - Y_pred)**2, axis=axis)


def calc_RMSE(Y, Y_pred, axis=0):
  return np.sqrt(calc_mean_squared_error(Y, Y_pred, axis=0))


def calc_mean_abs_error(Y, Y_pred, axis=0):
  return np.mean(np.abs(Y - Y_pred), axis=axis)


def calc_mean_rel_error(Y, Y_pred, axis=0):
  return np.mean(np.abs(Y_pred / Y - 1), axis=axis)


DEFAULT_METRIC_FNS = {
    "mean_squared_error": calc_mean_squared_error,
    "mean_abs_error": calc_mean_abs_error,
    "mean_rel_error": calc_mean_rel_error,
}


def calc_metrics(datasets, metric_fns=DEFAULT_METRIC_FNS, y_scaler=None):
  eval_metrics = {}
  for dataset, (Y, Y_pred) in datasets.items():
    # Go back to the unscaled values.
    if y_scaler is not None:
      Y = y_scaler.inverse_transform(Y)
      Y_pred = y_scaler.inverse_transform(Y_pred)

    print("Evaluating", dataset)
    values = {}
    for name, fn in metric_fns.items():
      values[name] = fn(Y, Y_pred)
      # Average over all output features so we have a single number.
      print(f"Average {name}: {np.mean(values[name]):.4g}")
    eval_metrics[dataset] = values

  return eval_metrics


def calc_metrics_df(datasets, metric_fns=DEFAULT_METRIC_FNS, y_scaler=None):
  eval_metrics = calc_metrics(datasets, metric_fns, y_scaler)
  return {
      name: pd.DataFrame.from_dict(values)
      for name, values in eval_metrics.items()
  }

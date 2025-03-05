import numpy as np
import pandas as pd
import tensorflow as tf

from cecilia import losses

# We don't call these "mean" error functions because we're not averaging over
# the class dimension (we will estimate the per-class errors by averaging over
# the batch dimension).
DEFAULT_METRIC_FNS = {
    "squared_error": losses.squared_error,
    "abs_error": losses.absolute_error,
    "rel_error": losses.relative_error,
}


def calc_metrics(datasets, metric_fns=DEFAULT_METRIC_FNS, y_scaler=None):
  eval_metrics = {}
  for dataset, (y_true, y_pred) in datasets.items():
    # Go back to the unscaled values.
    if y_scaler is not None:
      y_true = y_scaler.inverse_transform(y_true)
      y_pred = y_scaler.inverse_transform(y_pred)

    print("Evaluating", dataset)
    values = {}
    for name, fn in metric_fns.items():
      # Estimate the per-class errors by averaging over the batch dimension.
      values[name] = tf.reduce_mean(fn(y_true, y_pred), axis=0).numpy()
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

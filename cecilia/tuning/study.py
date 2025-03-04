import copy
import os
import shutil

import numpy as np
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from cecilia import training

TB_METRICS = {
    'loss': 'train_loss',
    'val_loss': 'test_loss',
    'train_mean_squared_error': 'train_MSE',
    'test_mean_squared_error': 'test_MSE',
    'train_mean_abs_error': 'train_MAE',
    'test_mean_abs_error': 'test_MAE',
    'train_mean_rel_error': 'train_MRE',
    'test_mean_rel_error': 'test_MRE'
}


def _log_metrics_df(metrics_df):
  for epoch, row in metrics_df.iterrows():
    for name in metrics_df.columns:
      tf.summary.scalar(name, row[name], step=epoch)


def run_tuning_study(study_dir,
                     base_config,
                     search_space,
                     train_df,
                     test_df,
                     X_cols,
                     Y_cols,
                     n_trials=None,
                     overwrite=False):
  if os.path.exists(study_dir):
    if overwrite:
      print(f"Removing existing output directory: {study_dir}")
      shutil.rmtree(study_dir)
    else:
      raise ValueError(f"Output directory exists: {study_dir}")

  # Log information for TensorBoard in the top level directory.
  with tf.summary.create_file_writer(study_dir).as_default():
    hp.hparams_config(hparams=search_space.get_tensorboard_specs(),
                      metrics=[
                          hp.Metric(name, display_name=dname)
                          for name, dname in TB_METRICS.items()
                      ])

  # Run trials.
  for n, trial_params in enumerate(search_space.search()):
    trial_id = str(n)
    trial_dir = os.path.join(study_dir, trial_id)
    if os.path.exists(trial_dir):
      continue  # Already done.

    print("Trial", n)

    # Sample new point in the search space.
    config = copy.deepcopy(base_config)
    config_updates = {}
    for name, value in trial_params.items():
      if name.startswith("one_minus_"):
        name = name[len("one_minus_"):]
        value = 1.0 - value
      config_updates[name] = value
    config.update(config_updates)
    print("Params:", config_updates)

    # Run the trial.
    history, eval_metrics = training.train_model(config,
                                                 train_df,
                                                 test_df,
                                                 X_cols,
                                                 Y_cols,
                                                 trial_dir,
                                                 save_model=False)

    # Log to Tensorboard.
    with tf.summary.create_file_writer(trial_dir).as_default():
      hp.hparams(trial_params,
                 trial_id=trial_id)  # record the values used in this trial
      _log_metrics_df(history)
      for dataset, metrics_df in eval_metrics.items():
        for metric in metrics_df.columns:
          epoch = config.num_epochs - 1
          avg_value = np.mean(metrics_df[metric])
          tf.summary.scalar(f"{dataset}_{metric}", avg_value, step=epoch)

    print()
    if n_trials and n >= n_trials - 1:
      break

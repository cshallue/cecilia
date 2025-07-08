import copy
import os
import shutil

import numpy as np
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from cecilia import configuration, training
from cecilia.data import photometry
from cecilia.tuning import search_space

METRIC_LABELS = {
    'loss': 'train_loss',
    'val_loss': 'test_loss',
    'train_squared_error': 'train_MSE',
    'test_squared_error': 'test_MSE',
    'train_abs_error': 'train_MAE',
    'test_abs_error': 'test_MAE',
    'train_rel_error': 'train_MRE',
    'test_rel_error': 'test_MRE'
}


def _log_metrics_df(metrics_df):
  for epoch, row in metrics_df.iterrows():
    for name in metrics_df.columns:
      tf.summary.scalar(name, row[name], step=epoch)


def run_tuning_study(study_config,
                     data_dir,
                     study_dir,
                     n_trials=None,
                     overwrite=False):
  if os.path.exists(study_dir):
    if overwrite:
      print(f"Removing existing output directory: {study_dir}")
      shutil.rmtree(study_dir)
    else:
      raise ValueError(f"Output directory exists: {study_dir}")

  # Save the study config.
  configuration.save(study_config, study_dir, basename="study_config")

  # Load the data.
  photometry.set_data_dir(data_dir)
  train_df, X_cols, Y_cols = photometry.read_and_process_dataset(
      study_config.train_filename)
  test_df, _, _ = photometry.read_and_process_dataset(
      study_config.test_filename)

  # Create the search space.
  ss = search_space.from_config(study_config.search_space)

  # Log information for TensorBoard in the top level directory.
  with tf.summary.create_file_writer(study_dir).as_default():
    hp.hparams_config(hparams=ss.get_tensorboard_specs(),
                      metrics=[
                          hp.Metric(name, display_name=dname)
                          for name, dname in METRIC_LABELS.items()
                      ])

  # Run trials.
  for n, search_params in enumerate(ss.search()):
    trial_id = str(n)
    trial_dir = os.path.join(study_dir, trial_id)
    if os.path.exists(trial_dir):
      continue  # Already done.

    print("Trial", n)

    # Override the base config with trial parameters.
    trial_params = copy.deepcopy(study_config.base_param_overrides)
    for name, value in search_params.items():
      if name.startswith("one_minus_"):
        name = name[len("one_minus_"):]
        value = 1.0 - value
      if "_X_" in name:
        name, param2 = name.split("_X_")
        # In order to specify param1_X_param2, param2 must appear in
        # base_param_overrides or in the search space before param1.
        value = value / trial_params[param2]
      trial_params[name] = value
    print("Params:", trial_params)

    config = configuration.default()
    config.update(trial_params)

    # Run the trial.
    history, eval_results = training.train_model(config,
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
      for dataset, metrics_df in eval_results.items():
        for metric in metrics_df.columns:
          epoch = config.num_epochs - 1
          avg_value = np.mean(metrics_df[metric])
          tf.summary.scalar(f"{dataset}_{metric}", avg_value, step=epoch)

    print()
    if n_trials and n >= n_trials - 1:
      break

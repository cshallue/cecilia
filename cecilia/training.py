import os
import shutil

import pandas as pd

from . import evaluation, model_builder
from .data import tf_dataset
from .preprocessing import utils


def train_model(config,
                train_df,
                test_df,
                X_cols,
                Y_cols,
                output_dir,
                save_model=True,
                overwrite=False,
                callbacks=None):
  if os.path.exists(output_dir):
    if overwrite:
      print(f"Removing existing output directory: {output_dir}")
      shutil.rmtree(output_dir)
    else:
      raise ValueError(f"Output directory exists: {output_dir}")

  # Write config.
  os.makedirs(output_dir)
  with open(os.path.join(output_dir, "config.json"), "w") as f:
    f.write(config.to_json(indent=2))

  # Extract features and targets.
  x_scaler, y_scaler = utils.create_scalers(config.log_transform_y,
                                            config.normalize_y)
  X_train, Y_train, X_test, Y_test = utils.extract_features_targets(
      train_df, test_df, X_cols, Y_cols, x_scaler=x_scaler, y_scaler=y_scaler)

  # Create batched dataset iterators.
  train_dataset = tf_dataset.build(X_train,
                                   Y_train,
                                   config.batch_size,
                                   shuffle=True)
  test_dataset = tf_dataset.build(X_test,
                                  Y_test,
                                  config.batch_size,
                                  shuffle=False)

  # Build and train model.
  model = model_builder.build_and_compile(config)
  print("Training model...")
  history = model.fit(train_dataset,
                      epochs=config.num_epochs,
                      validation_data=test_dataset,
                      callbacks=callbacks)
  if save_model:
    model.save(os.path.join(output_dir, "model.keras"))
  print("Done training model")

  # Save the train curve.
  history_df = pd.DataFrame(history.history)
  history_df.to_csv(os.path.join(output_dir, "train_curve.csv"))

  # Evaluate the model.
  print("Evaluating model...")
  Y_pred_train = model.predict(X_train, batch_size=config.batch_size)
  Y_pred_test = model.predict(X_test, batch_size=config.batch_size)
  datasets = {"train": (Y_train, Y_pred_train), "test": (Y_test, Y_pred_test)}
  eval_metrics = evaluation.calc_metrics_df(datasets=datasets,
                                            y_scaler=y_scaler)
  for name, df in eval_metrics.items():
    df.to_csv(os.path.join(output_dir, f"metrics_{name}.csv"))
  print("Done evaluating model")

  return history_df, eval_metrics

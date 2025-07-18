import os
import shutil

import numpy as np
import pandas as pd

from cecilia import configuration, evaluation, photometric_model, preprocessing
from cecilia.data import tf_dataset


def train_model(config,
                train_df,
                test_df,
                X_cols,
                Y_cols,
                output_dir,
                run_eval=True,
                save_model=True,
                save_eval_tables=False,
                overwrite=False,
                callbacks=None):
  if os.path.exists(output_dir):
    if overwrite:
      print(f"Removing existing output directory: {output_dir}")
      shutil.rmtree(output_dir)
    else:
      raise ValueError(f"Output directory exists: {output_dir}")

  # Write config.
  configuration.save(config, output_dir)

  # Extract features and targets.
  X_train, Y_train, X_test, Y_test = preprocessing.extract_features_targets(
      train_df, test_df, X_cols, Y_cols)
  steps_per_epoch = int(np.ceil(len(X_train) / config.batch_size))
  print(f"Steps per epoch: {steps_per_epoch}")

  # Build and compile model.
  model = photometric_model.PhotometricModel(config)
  model.x_transformer.fit(X_train)
  model.y_transformer.fit(Y_train)
  model.compile(steps_per_epoch=steps_per_epoch)

  # Create batched dataset iterators.
  train_dataset = tf_dataset.build(X_train,
                                   Y_train,
                                   config.batch_size,
                                   shuffle=True)
  test_dataset = tf_dataset.build(X_test,
                                  Y_test,
                                  config.batch_size,
                                  shuffle=False)

  # Train model.
  print("Training model...")
  history = model.fit(train_dataset,
                      epochs=config.num_epochs,
                      validation_data=test_dataset,
                      callbacks=callbacks)
  if save_model:
    model.save_weights(os.path.join(output_dir, "model.weights.h5"))
  print("Done training model")

  # Save the train curve.
  history_df = pd.DataFrame(history.history)
  history_df.to_csv(os.path.join(output_dir, "train_curve.csv"))
  train_loss, val_loss = history_df.iloc[-1][["loss", "val_loss"]]
  print(f"Final train loss = {train_loss:.6g}")
  print(f"Final validation loss = {val_loss:.6g}")

  if not run_eval:
    return history_df

  # Evaluate the model.
  print("Evaluating model...")
  Y_pred_train = model.predict(X_train, batch_size=config.batch_size)
  Y_pred_test = model.predict(X_test, batch_size=config.batch_size)
  datasets = {"train": (Y_train, Y_pred_train), "test": (Y_test, Y_pred_test)}
  if save_eval_tables:
    for name, data in [
        ("Y_train", Y_train),
        ("Y_pred_train", Y_pred_train),
        ("Y_test", Y_test),
        ("Y_pred_test", Y_pred_test),
    ]:
      filename = os.path.join(output_dir, f"{name}.csv")
      pd.DataFrame(data, columns=Y_cols).to_csv(filename, index=False)
  eval_results = evaluation.calc_metrics_df(datasets=datasets)
  for name, df in eval_results.items():
    df.to_csv(os.path.join(output_dir, f"metrics_{name}.csv"))
  print("Done evaluating model")

  return history_df, eval_results

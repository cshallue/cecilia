import ml_collections


def default():
  return ml_collections.ConfigDict({
      # Data.
      "normalize_x": True,
      "log_transform_y": False,
      "normalize_y": True,
      # Model.
      "dim_input": 17,
      "n_hidden": 2,
      "dim_hidden": 1024,
      "hidden_activation": "relu",
      "dim_output": 16,
      "predict_std_method": "none",
      "predict_constant_std_value": 1.0,
      # Training.
      "loss": "mean_squared_error",
      "loss_rescaling_method": "none",
      "loss_rescaling_value": [],
      "loss_shift": [],
      "learning_rate": 1e-3,
      "momentum": 0.9,
      "num_epochs": 20,
      "batch_size": 512,
  })

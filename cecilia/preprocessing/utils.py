def apply_scaler(train_arr, test_arr, scaler):
  train_arr = scaler.fit_transform(train_arr)
  test_arr = scaler.transform(test_arr)
  return train_arr, test_arr


def extract_features_targets(train_df,
                             test_df,
                             x_cols,
                             y_cols,
                             x_scaler=None,
                             y_scaler=None):
  X_train, Y_train = train_df[x_cols].values, train_df[y_cols].values
  X_test, Y_test = test_df[x_cols].values, test_df[y_cols].values

  if x_scaler is not None:
    X_train, X_test = apply_scaler(X_train, X_test, x_scaler)
    print("Re-scaled x's")

  if y_scaler is not None:
    Y_train, Y_test = apply_scaler(Y_train, Y_test, y_scaler)
    print("Re-scaled y's")

  return X_train, Y_train, X_test, Y_test

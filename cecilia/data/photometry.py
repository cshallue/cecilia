import os
import re

import pandas as pd

_DATA_DIR = None


def set_data_dir(data_dir):
  global _DATA_DIR
  _DATA_DIR = data_dir


def read_raw_dataset(filename):
  if _DATA_DIR is None:
    raise ValueError("Must call set_data_dir() before reading data file.")

  return pd.read_pickle(os.path.join(_DATA_DIR, filename))


def read_and_process_dataset(filename, teff_choice='all', data_type='fP'):
  df = read_raw_dataset(filename)
  return process_dataset(df, teff_choice, data_type)


def process_dataset(phot_and_labels_df, teff_choice='all', data_type='fP'):
  print(f"Number of Synthetic WDs in the sample: {len(phot_and_labels_df)}")

  # Filter based on teff.
  if teff_choice == 'all':
    teff_choice = ['low', 'medium', 'high']
  if isinstance(teff_choice, str):
    teff_choice = [teff_choice]

  if teff_choice:
    phot_and_labels_df = phot_and_labels_df[
        phot_and_labels_df['teff_type'].isin(teff_choice)]

  print(f"Number matching teff = {teff_choice}: {len(phot_and_labels_df)}")

  #=========================================================================================
  #::: Remove unnecessary "string" columns for training ('mdl_name', 'teff_type', and any 'bessel_*' passband columns)
  #=========================================================================================

  columns_to_drop = ['mdl_name', 'teff_type'] + [
      col for col in phot_and_labels_df.columns if col.startswith('bessel_')
  ]
  data = phot_and_labels_df.drop(columns=columns_to_drop)

  #=========================================================================================
  #::: Get the Names and Indices of the Stellar Properties (X) and Photometric Properties (Y)
  #=========================================================================================

  # Define regex patterns based on data_type
  if data_type == 'fPoverfNP':
    photometric_pattern = re.compile(
        r'(sdss|ps|gaiadr3|2mass)_[a-z]+_fPoverfNP$')  # Match only _fPoverfNP
  elif data_type == 'fP':
    photometric_pattern = re.compile(
        r'(sdss|ps|gaiadr3|2mass)_[a-z]+_fP$')  # Match only _fP
  elif data_type == 'fNP':
    photometric_pattern = re.compile(
        r'(sdss|ps|gaiadr3|2mass)_[a-z]+_fNP$')  # Match only _fNP

  # Identify the columns containing the photometry
  names_photometric_columns = [
      col for col in data.columns if photometric_pattern.match(col)
  ]
  print(f"Names of Photometric Passbands: {names_photometric_columns}")

  # Identify all photometric-related columns (fP, fNP, fPoverfNP) to exclude them from stellar labels
  all_photometric_related_pattern = re.compile(
      r'(sdss|ps|gaiadr3|2mass)_[a-z]+_(fP|fNP|fPoverfNP)$')
  all_photometric_related_columns = [
      col for col in data.columns if all_photometric_related_pattern.match(col)
  ]

  # Identify columns with stellar properties
  names_stellar_props_columns = [
      col for col in data.columns if col not in all_photometric_related_columns
  ]
  print(f"Names of Stellar Labels: {names_stellar_props_columns}")

  #=========================================================================================
  #::: Separate the data into Stellar Properties (X) and Photometric Properties (Y)
  #=========================================================================================

  # if log_photometry:
  #   new_names = []
  #   for name in names_photometric_columns:
  #     data[name] = data[name].transform(np.log10)
  #     new_name = "log_" + name
  #     data.rename(columns={name: new_name}, inplace=True)
  #     new_names.append(new_name)
  #   names_photometric_columns = new_names

  # Separate data into feature (X) and label (Y)
  X_cols = names_stellar_props_columns
  Y_cols = names_photometric_columns
  XY_cols = X_cols + Y_cols
  data = data[XY_cols]

  return data, X_cols, Y_cols

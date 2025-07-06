"""Script for tuning a Cecilia model locally."""

import tensorflow as tf
from absl import app, flags, logging

from cecilia import configuration
from cecilia.tuning import study

flags.DEFINE_string("config_file",
                    None,
                    "Path of JSON file containing the study configuration.",
                    required=True)

flags.DEFINE_string("data_dir",
                    None,
                    "Directory containing training and test data files.",
                    required=True)

flags.DEFINE_string("study_dir",
                    None,
                    "Directory to write study results.",
                    required=True)

flags.DEFINE_bool("overwrite", False,
                  "Whether to overwrite existing study directory.")

flags.DEFINE_integer("n_trials", None, "Maximum number of trials to run.")

flags.DEFINE_integer("gpu", None, "Index of GPU device to run on.")

FLAGS = flags.FLAGS


def main(_):
  if FLAGS.gpu is not None:
    gpu_devices = tf.config.get_visible_devices("GPU")
    tf.config.set_visible_devices([gpu_devices[FLAGS.gpu]], "GPU")
    logging.info(
        f"Set logical GPU devices to {tf.config.list_logical_devices('GPU')}")

  study_config = configuration.load(FLAGS.config_file)
  logging.info("Study config: %s", study_config.to_json(indent=2))

  study.run_tuning_study(study_config,
                         data_dir=FLAGS.data_dir,
                         study_dir=FLAGS.study_dir,
                         n_trials=FLAGS.n_trials,
                         overwrite=FLAGS.overwrite)


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  app.run(main)

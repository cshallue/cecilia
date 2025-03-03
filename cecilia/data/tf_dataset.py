import tensorflow as tf


def build(x, y, batch_size, shuffle):
  ds = tf.data.Dataset.from_tensor_slices((x, y))
  if shuffle:
    ds = ds.shuffle(ds.cardinality(), reshuffle_each_iteration=True)
  return ds.batch(batch_size)

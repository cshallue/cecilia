import abc
import itertools

import numpy as np
from tensorboard.plugins.hparams import api as hp


class SearchSpaceAxis(abc.ABC):

  @abc.abstractmethod
  def tensorboard_spec(self):
    ...


class SearchSpace(abc.ABC):

  def __init__(self):
    self.axes = {}

  def get_tensorboard_specs(self):
    return [
        hp.HParam(name, ax.tensorboard_spec())
        for name, ax in self.axes.items()
    ]

  @abc.abstractmethod
  def search(self):
    ...


# Random search


class RandomParam(SearchSpaceAxis):

  def __init__(self, rng):
    self.rng = rng

  @abc.abstractmethod
  def sample(self):
    ...


class DiscreteParam(RandomParam):

  def __init__(self, rng, values):
    super().__init__(rng)
    self.values = values

  def sample(self):
    return self.rng.choice(self.values)

  def tensorboard_spec(self):
    return hp.Discrete(self.values)


class RealParam(RandomParam):

  def __init__(self, rng, low, high):
    super().__init__(rng)
    self.low = low
    self.high = high

  def tensorboard_spec(self):
    return hp.RealInterval(self.low, self.high)


class UniformParam(RealParam):

  def sample(self):
    return self.rng.uniform(self.low, self.high)


class LogUniformParam(RealParam):

  def sample(self):
    log_x = self.rng.uniform(np.log(self.low), np.log(self.high))
    return np.exp(log_x)


class RandomSearchSpace(SearchSpace):

  def __init__(self, rng=None):
    super().__init__()

    if rng is None:
      rng = np.random.default_rng()

    self.rng = rng

  def add_axis(self, name, param):
    self.axes[name] = param

  def add_discrete_param(self, name, values):
    self.add_axis(name, DiscreteParam(self.rng, values))

  def add_uniform_param(self, name, low, high):
    self.add_axis(name, UniformParam(self.rng, low, high))

  def add_log_uniform_param(self, name, low, high):
    self.add_axis(name, LogUniformParam(self.rng, low, high))

  def search(self):
    while True:
      yield {name: ax.sample() for name, ax in self.axes.items()}


# Grid search


class GridSearchAxis(SearchSpaceAxis):

  def __init__(self, values, tb_type):
    self.values = values
    self.tb_type = tb_type

  def tensorboard_spec(self):
    if self.tb_type == "discrete":
      return hp.Discrete(self.values)

    if self.tb_type == "continuous":
      return hp.RealInterval(np.min(self.values), np.max(self.values))

    raise ValueError(self.tb_type)


class GridSearchSpace(SearchSpace):

  def add_axis(self, name, values, tb_type="discrete"):
    self.axes[name] = GridSearchAxis(values, tb_type)

  def search(self):
    for gridpoint in itertools.product(*(ax.values
                                         for ax in self.axes.values())):
      yield {name: value for name, value in zip(self.axes.keys(), gridpoint)}

"""Classes defining hyperparameter search spaces."""

import abc
import itertools

import numpy as np
from tensorboard.plugins.hparams import api as hp


class SearchSpaceAxis(abc.ABC):
  """Base class for an axis of the search space (e.g. a hyperparameter)."""

  @abc.abstractmethod
  def get_domain(self):
    """Returns a hp.Domain object describing the domain of the axis."""
    ...


class SearchSpace(abc.ABC):
  """Base class for a search space."""

  def __init__(self):
    self.axes = {}

  def get_tensorboard_specs(self):
    """Returns a list of hp.HParam objects for all axes in the search space."""
    return [hp.HParam(name, ax.get_domain()) for name, ax in self.axes.items()]

  @abc.abstractmethod
  def search(self):
    """Returns an iterator over the search space."""
    ...


# Random search


class RandomParam(SearchSpaceAxis):
  """A parameter whose value is sampled randomly."""

  def __init__(self, rng):
    self.rng = rng

  @abc.abstractmethod
  def sample(self):
    """Samples a single value from the domain."""
    ...


class DiscreteParam(RandomParam):
  """A parameter whose value is sampled randomly from a discrete set."""

  def __init__(self, rng, values):
    super().__init__(rng)
    self.values = values

  def sample(self):
    return self.rng.choice(self.values)

  def get_domain(self):
    return hp.Discrete(self.values)


class RealParam(RandomParam):
  """A parameter whose value is sampled randomly from a real interval."""

  def __init__(self, rng, low, high):
    super().__init__(rng)
    self.low = low
    self.high = high

  def get_domain(self):
    return hp.RealInterval(self.low, self.high)


class UniformParam(RealParam):
  """A parameter whose value is sampled uniformly from a real interval."""

  def sample(self):
    return self.rng.uniform(self.low, self.high)


class LogUniformParam(RealParam):
  """A parameter whose logarithm is sampled randomly from a real interval."""

  def sample(self):
    log_x = self.rng.uniform(np.log(self.low), np.log(self.high))
    return np.exp(log_x)


class RandomSearchSpace(SearchSpace):
  """A random search space."""

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
  """An axis in a grid search space."""

  def __init__(self, values, tb_type):
    self.values = values
    self.tb_type = tb_type

  def get_domain(self):
    if self.tb_type == "discrete":
      return hp.Discrete(self.values)

    if self.tb_type == "continuous":
      return hp.RealInterval(np.min(self.values), np.max(self.values))

    raise ValueError(self.tb_type)


class GridSearchSpace(SearchSpace):
  """A grid search space."""

  def add_axis(self, name, values, tb_type="discrete"):
    self.axes[name] = GridSearchAxis(values, tb_type)

  def search(self):
    for gridpoint in itertools.product(*(ax.values
                                         for ax in self.axes.values())):
      yield {name: value for name, value in zip(self.axes.keys(), gridpoint)}

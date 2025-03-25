"""Classes defining hyperparameter search spaces."""

import abc
import itertools

import numpy as np
from tensorboard.plugins.hparams import api as hp


class SearchSpaceAxis(abc.ABC):
  """Base class for an axis of the search space (e.g. a hyperparameter)."""

  @property
  @abc.abstractmethod
  def domain(self):
    """Returns a hp.Domain object describing the domain of the axis."""


class SearchSpace(abc.ABC):
  """Base class for a search space."""

  def __init__(self):
    self.axes = {}

  def get_tensorboard_specs(self):
    """Returns a list of hp.HParam objects for all axes in the search space."""
    return [hp.HParam(name, ax.domain) for name, ax in self.axes.items()]

  @abc.abstractmethod
  def search(self):
    """Returns an iterator over the search space."""


# Random search


class RandomParam(SearchSpaceAxis):
  """A parameter whose value is sampled randomly."""

  def __init__(self, rng):
    self.rng = rng

  @abc.abstractmethod
  def sample(self):
    """Samples a single value from the domain."""


class DiscreteParam(RandomParam):
  """A parameter whose value is sampled randomly from a discrete set."""

  def __init__(self, rng, values):
    super().__init__(rng)
    self.values = values
    self._domain = hp.Discrete(self.values)

  def sample(self):
    return self.rng.choice(self.values)

  @property
  def domain(self):
    return self._domain


class ContiguousParam(RandomParam):
  """A parameter whose value is sampled randomly from an interval."""

  def __init__(self, rng, low, high, integer_valued=False):
    super().__init__(rng)
    self.low = low
    self.high = high
    self.integer_valued = integer_valued
    if integer_valued:
      # Conventionally, integer intervals are inclusive on both sides while real
      # intervals are exclusive at the upper end.
      self.high += 1
      self._domain = hp.IntInterval(self.low, self.high)
    else:
      self._domain = hp.RealInterval(self.low, self.high)

  @property
  def domain(self):
    return self._domain

  @abc.abstractmethod
  def _sample_real(self):
    """Samples a real value."""

  def sample(self):
    value = self._sample_real()
    return int(value) if self.integer_valued else value


class UniformParam(ContiguousParam):
  """A parameter whose value is sampled uniformly from an interval."""

  def _sample_real(self):
    return self.rng.uniform(self.low, self.high)


class LogUniformParam(ContiguousParam):
  """A parameter whose logarithm is sampled randomly from an interval."""

  def _sample_real(self):
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

  def add_uniform_param(self, name, low, high, integer_valued=False):
    self.add_axis(name, UniformParam(self.rng, low, high, integer_valued))

  def add_log_uniform_param(self, name, low, high, integer_valued=False):
    self.add_axis(name, LogUniformParam(self.rng, low, high, integer_valued))

  def search(self):
    while True:
      point = {}
      for name, ax in self.axes.items():
        value = ax.sample()
        if isinstance(value, np.integer):
          value = int(value)  # Convert from numpy integer to int
        point[name] = value
      yield point


# Grid search


class GridSearchAxis(SearchSpaceAxis):
  """An axis in a grid search space."""

  def __init__(self, values, tb_type):
    self.values = values
    self.tb_type = tb_type
    if tb_type == "discrete":
      self._domain = hp.Discrete(values)
    elif tb_type == "continuous":
      self._domain = hp.RealInterval(np.min(values), np.max(values))
    else:
      raise ValueError(tb_type)

  def domain(self):
    return self._domain


class GridSearchSpace(SearchSpace):
  """A grid search space."""

  def add_axis(self, name, values, tb_type="discrete"):
    self.axes[name] = GridSearchAxis(values, tb_type)

  def search(self):
    for gridpoint in itertools.product(*(ax.values
                                         for ax in self.axes.values())):
      yield {name: value for name, value in zip(self.axes.keys(), gridpoint)}

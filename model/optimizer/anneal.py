# from https://github.com/perrygeo/simanneal/blob/master/simanneal/anneal.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import copy
import datetime
import math
import pickle
import random
import signal
import sys
import time

from .base import BaseOptimizer
from random import random, randint, choice
from math import floor, ceil


def round_figures(x, n):
  """Returns x rounded to n significant figures."""
  return round(x, int(n - math.ceil(math.log10(abs(x)))))


def time_string(seconds):
  """Returns time in seconds as a string formatted HHHH:MM:SS."""
  s = int(round(seconds))  # round to nearest second
  h, s = divmod(s, 3600)   # get hours and remainder
  m, s = divmod(s, 60)     # split remainder into minutes and seconds
  return '%4i:%02i:%02i' % (h, m, s)


class Annealer(object):

  """Performs simulated annealing by calling functions to calculate
  energy and make moves on a state.  The temperature schedule for
  annealing may be provided manually or estimated automatically.
  """

  __metaclass__ = abc.ABCMeta

  # defaults
  Tmax = 25000.0
  Tmin = 2.5
  steps = 50000
  updates = 100
  copy_strategy = 'deepcopy'
  user_exit = False
  save_state_on_exit = False

  # placeholders
  best_state = None
  best_energy = None
  start = None

  out_file = None

  def __init__(self, initial_state, initial_E, estimated_steps):
    self.state = self.copy_state(initial_state)
    self.E = initial_E
    self.steps = estimated_steps

    signal.signal(signal.SIGINT, self.set_user_exit)

  def save_state(self, fname=None):
    """Saves state to pickle"""
    if not fname:
      date = datetime.datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss")
      fname = date + "_energy_" + str(self.energy()) + ".state"
    with open(fname, "wb") as fh:
      pickle.dump(self.state, fh)

  def load_state(self, fname=None):
    """Loads state from pickle"""
    with open(fname, 'rb') as fh:
      self.state = pickle.load(fh)

  @abc.abstractmethod
  def move(self, state=None):
    """Create a state change"""
    pass

  @abc.abstractmethod
  def energy(self, state=None):
    """Calculate state's energy"""
    pass

  def set_user_exit(self, signum, frame):
    """Raises the user_exit flag, further iterations are stopped
    """
    self.user_exit = True

  def set_schedule(self, schedule):
    """Takes the output from `auto` and sets the attributes
    """
    self.Tmax = schedule['tmax']
    self.Tmin = schedule['tmin']
    self.steps = int(schedule['steps'])
    self.updates = int(schedule['updates'])

  def copy_state(self, state):
    """Returns an exact copy of the provided state
    Implemented according to self.copy_strategy, one of
    * deepcopy : use copy.deepcopy (slow but reliable)
    * slice: use list slices (faster but only works if state is list-like)
    * method: use the state's copy() method
    """
    if self.copy_strategy == 'deepcopy':
      return copy.deepcopy(state)
    elif self.copy_strategy == 'slice':
      return state[:]
    elif self.copy_strategy == 'method':
      return state.copy()
    else:
      raise RuntimeError('No implementation found for ' +
                         'the self.copy_strategy "%s"' %
                         self.copy_strategy)

  def update(self, *args, **kwargs):
    """Wrapper for internal update.
    If you override the self.update method,
    you can chose to call the self.default_update method
    from your own Annealer.
    """
    self.default_update(*args, **kwargs)

  def default_update(self, step, T, E, acceptance, improvement):
    """Default update, outputs to stderr.
    Prints the current temperature, energy, acceptance rate,
    improvement rate, elapsed time, and remaining time.
    The acceptance rate indicates the percentage of moves since the last
    update that were accepted by the Metropolis algorithm.  It includes
    moves that decreased the energy, moves that left the energy
    unchanged, and moves that increased the energy yet were reached by
    thermal excitation.
    The improvement rate indicates the percentage of moves since the
    last update that strictly decreased the energy.  At high
    temperatures it will include both moves that improved the overall
    state and moves that simply undid previously accepted moves that
    increased the energy by thermal excititation.  At low temperatures
    it will tend toward zero as the moves that can decrease the energy
    are exhausted and moves that would increase the energy are no longer
    thermally accessible."""

    elapsed = time.time() - self.start
    if step == 0:
      print(' Temperature        Energy    Accept   Improve     Elapsed   Remaining',
            file=self.out_file)
      print('\r%12.5f  %12.2f                      %s            ' %
            (T, E, time_string(elapsed)), file=self.out_file, end="\r")
      self.out_file.flush()
    else:
      remain = (self.steps - step) * (elapsed / step)
      print('\r%12.5f  %12.2f  %7.2f%%  %7.2f%%  %s  %s\r' %
            (T, E, 100.0 * acceptance, 100.0 * improvement,
             time_string(elapsed), time_string(remain)), file=self.out_file, end="\r")
      self.out_file.flush()

  def anneal_init(self):
    self.step = 0
    self.start = time.time()

    # Precompute factor for exponential cooling from Tmax to Tmin
    if self.Tmin <= 0.0:
      raise Exception('Exponential cooling requires a minimum "\
                "temperature greater than zero.')
    self.Tfactor = -math.log(self.Tmax / self.Tmin)

    # Note initial state
    self.T = self.Tmax
    # self.E = self.energy()
    self.best_state = self.copy_state(self.state)
    self.best_energy = self.E
    self.trials, self.accepts, self.improves = 0, 0, 0

    self.update(self.step, self.T, self.E, None, None)

  def anneal_update(self, ob):
    state, E = ob

    if self.state is None:
      self.state, self.E = state, E
      return

    Tfactor = -math.log(self.Tmax / self.Tmin)

    # Note initial state
    prevEnergy = self.E

    # Attempt moves to new states
    self.step += 1
    # update temperature
    self.T = max(self.Tmin, self.Tmax *
                 math.exp(Tfactor * self.step / self.steps))

    dE = E - prevEnergy
    self.trials += 1
    if dE > 0.0 and math.exp(-dE / self.T) < random():
      # Restore previous state, do nothing
      pass
    else:
      # Accept new state and compare to best state
      self.E, self.state = E, self.copy_state(state)

      self.accepts += 1
      if dE < 0.0:
        self.improves += 1
      if E < self.best_energy:
        self.best_state = self.copy_state(state)
        self.best_energy = E
      self.update(
          self.step, self.T, self.E, self.accepts / self.trials, self.improves / self.trials)

    self.state = self.copy_state(self.best_state)
    if self.save_state_on_exit:
      self.save_state()

    # Return best state and energy
    # return self.best_state, self.best_energy


class AnnealOptimizer(BaseOptimizer):
  def __init__(self, para_setting):
    super().__init__(para_setting)
    self.has_first_state = False
    self.annealer = None

  def get_conf(self):
    if not self.has_first_state:
      return self._random_sample()
    else:
      return self._move(self.annealer.state)

  def add_observation(self, ob):
    # negative energy
    ob = (ob[0], -ob[1])

    if not self.has_first_state:
      # init
      state, E = ob
      self.annealer = Annealer(
          initial_state=state,
          initial_E=E,
          estimated_steps=30  # TODO
      )
      self.annealer.out_file = self.file
      # TODO determine temperature
      self.annealer.Tmax = 2000
      self.annealer.Tmin = 10
      self.annealer.anneal_init()
      self.has_first_state = True
    else:
      self.annealer.anneal_update(ob)

  def set_status_file(self, path):
    self.file = open(path, 'a')

  def _random_sample(self):
    res = {}
    res_numeric = []
    for k, conf in self.para_setting.items():
      numer_range = conf.get('range')
      if numer_range is None:
        minn = conf.get('min')
        maxx = conf.get('max')
        allow_float = conf.get('float', False)

        res[k] = random() * (maxx - minn) + minn \
            if allow_float else randint(minn, maxx)
        res_numeric.append(res[k])
      else:
        choice = randint(0, len(numer_range) - 1)

        res[k] = numer_range[choice]
        res_numeric.append(choice)
      if type(res[k]) is bool:
        # make sure no uppercase 'True/False' literal in result
        res[k] = str(res[k]).lower()
    return res_numeric, res

  def _move(self, state_numeric, ratio=.2):
    res = {}
    res_numeric = []

    for (k, conf), v in zip(self.para_setting.items(), state_numeric):
      #print(k)
      #print(conf)
      numer_range = conf.get('range')
      if numer_range is None:
        # 1/3 neighbor
        minn = conf.get('min')
        maxx = conf.get('max')
        allow_float = conf.get('float', False)

        span = (maxx - minn) * ratio / 2
        minn = max(minn, v - span)
        maxx = min(maxx, v + span)

        res[k] = random() * (maxx - minn) + minn \
            if allow_float else randint(floor(minn), ceil(maxx))
        res_numeric.append(res[k])
      else:
        choice = randint(max(0, v - 1), min(len(numer_range) - 1, v + 1))

        res[k] = numer_range[choice]
        res_numeric.append(choice)
      if type(res[k]) is bool:
        # make sure no uppercase 'True/False' literal in result
        res[k] = str(res[k]).lower()

    return res_numeric, res

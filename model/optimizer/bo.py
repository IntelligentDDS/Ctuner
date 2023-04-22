from ..bayes_opt import BayesianOptimization
from ..bayes_opt.helpers import acq_max, UtilityFunction
from random import random


def noop(*kargs, **kwargs):
  # stub function for bo
  return None


class BayesianOptimizer():
  def __init__(self, space, conf={}):
    conf = {
        **conf,
        'pbounds': space,
    }
    self.space = space
    self.conf = conf
#########conf contains acq, use; else use default
    self.acq = conf.get('acq', 'ucb')
    self.kappa = conf.get('kappa', 2.576)
    self.xi = conf.get('xi', 0.0)
    try:
      del conf['acq'], conf['kappa'], conf['xi']
    except:
      pass
    #print(self.space)
    self.bo = BayesianOptimization(**self._make_config(conf))
    # print('show make config:')
    # print(self._make_config(conf))

  def _make_config(self, conf):
    return {
        **conf,
        'f': noop
    }

  def add_observation(self, ob):
    # ob: (x, y) while x is argument dict

    _x, y = ob
    # print('show ob_x, i.e, sampled_config_numeric:')
    # print(_x)
    # print('show ob.y, i.e, metric_result:')
    # print(y)
    # dict to tuple regarding keys in self.space
    x = []
    for k in self.space.keys():
      x.append(_x[k])

    #print(x,y)
    # add ob into bo space
    try:
      #space.add_observation(x, y) is define in the TargetSpace.py file
      self.bo.space.add_observation(x, y)
    except KeyError as e:
      # get exception message
      msg, = e.args
      raise Exception(msg)
    # print('show ob.space.X，only values no keys:')
    # print(self.bo.space.X)
    # print('show ob.space.Y:')
    # print(self.bo.space.Y)
    self.bo.gp.fit(self.bo.space.X, self.bo.space.Y)

  def get_conf(self):
    acq = self.acq
    kappa = self.kappa
    xi = self.xi
    if self.bo.space.Y is None or len(self.bo.space.Y) == 0:
      x_max = self.bo.space.random_points(1)[0]
    else:
      x_max = acq_max(
          ac=UtilityFunction(
              kind=acq,
              kappa=kappa,
              xi=xi
          ).utility,
          gp=self.bo.gp,
          y_max=self.bo.space.Y.max(),
          bounds=self.bo.space.bounds,
          random_state=self.bo.random_state,
          **self.bo._acqkw
      )
    # check if x_max repeats
    if x_max in self.bo.space:
      x_max = self.bo.space.random_points(1)[0]

    # print('show xmax from acqmax():')
    # print(x_max)
    return self._convert_to_dict(x_max)

  def _convert_to_dict(self, x_array):
    # print('show self.space, not self.bo.space, should be{'':()}:')
    # print(self.space)
    return dict(zip(self.space, x_array))


class ConfigedBayesianOptimizer(BayesianOptimizer):
  def __init__(self, config, bo_conf={}):
    self._config = {**config}
    #print(self._config)
    bo_space = {}
    for k, v in self._config.items():
      v_range = v.get('range')
      if v_range:  # discrete ranged parameter
        bo_space[k] = (0, len(v_range))  # note: right-close range
      else:
        bo_space[k] = (v['min'], v['max'])
    #print(bo_space)
    #print(bo_conf)
    super().__init__(bo_space, bo_conf)

  # get conf and convert to legal config
  def get_conf(self):
    sample = super().get_conf()
    print('show sample from father\'s get_conf:')
    print(sample)
    # first is continuous value, second is translated
    return sample, self._translate(sample)

  def random_sample(self):
    result = {}
    for k, v in self._config.items():
      v_range = v.get('range')
      if v_range:
        result[k] = random() * len(v_range)
      else:
        minn, maxx = v.get('min'), v.get('max')
        result[k] = random() * (maxx - minn) + minn
    return result, self._translate(result)

  def _translate(self, sample):
    result = {}
    # orders in sample are the same as in _config dict
    #   see: https://github.com/fmfn/BayesianOptimization/blob/d531dcab1d73729528afbffd9a9c47c067de5880/bayes_opt/target_space.py#L49
    #   self.bounds = np.array(list(pbounds.values()), dtype=np.float)
    for sample_value, (k, v) in zip(sample.values(), self._config.items()):
      v_range = v.get('range')
      if v_range:
        try:
          index = int(sample_value)
          if index == len(v_range):
            index -= 1
          result[k] = v_range[index]
        except Exception as e:
          print('ERROR!')
          print(k, sample_value)
          print(v_range)
          raise e
      else:
        is_float = v.get('float', False)
        result[k] = sample_value if is_float else int(sample_value)
    #print(result)
    return result

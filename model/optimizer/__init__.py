from .bo import ConfigedBayesianOptimizer
from .anneal import AnnealOptimizer
from .hesbo import HesboOptimizer
from .rhesbo import RHesboOptimizer

available_optimizer = [
    'bo',
    'hesbo',
    'anneal',
    'rhesbo',
]

def create_optimizer(name, configs, extra_vars={}):
  assert name in available_optimizer, f'optimizer [{name}] not supported.'
  if name == 'bo':
    return ConfigedBayesianOptimizer(configs, bo_conf=extra_vars)
  elif name == 'hesbo':
    return HesboOptimizer(configs, rembo_conf=extra_vars)
  elif name == 'rhesbo':
    return RHesboOptimizer(configs, rembo_conf=extra_vars)
  elif name == 'anneal':
    return AnnealOptimizer(configs)

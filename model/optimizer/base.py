class BaseOptimizer:
  def __init__(self, para_setting):
    self.para_setting = para_setting

  def get_conf(self):
    # return None, random_sample(self.para_setting)
    raise NotImplementedError()

  def add_observation(self, ob):
    raise NotImplementedError()

  def dump_state(self, path):
    pass

# -*- coding: utf-8 -*-

import time
import pickle
import logging
import datetime
import sys
import yaml
import re
import traceback
from pathlib import Path
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from environment import configs

def time_start():
    return time.time()


def time_end(start):
    end = time.time()
    delay = end - start
    return delay


def get_timestamp():
    """
    获取UNIX时间戳
    """
    return int(time.time())


def time_to_str(timestamp):
    """
    将时间戳转换成[YYYY-MM-DD HH:mm:ss]格式
    """
    return datetime.datetime.\
        fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


class Logger:

    def __init__(self, name, log_file=''):
        self.log_file = log_file
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        sh = logging.StreamHandler()
        self.logger.addHandler(sh)
        if len(log_file) > 0:
            self.log2file = True
        else:
            self.log2file = False

    def _write_file(self, msg):
        if self.log2file:
            with open(self.log_file, 'a+') as f:
                f.write(msg + '\n')

    def get_timestr(self):
        timestamp = get_timestamp()
        date_str = time_to_str(timestamp)
        return date_str

    def warn(self, msg):
        msg = "%s[WARN] %s" % (self.get_timestr(), msg)
        self.logger.warning(msg)
        self._write_file(msg)

    def info(self, msg):
        msg = "%s[INFO] %s" % (self.get_timestr(), msg)
        #self.logger.info(msg)
        self._write_file(msg)

    def error(self, msg):
        msg = "%s[ERROR] %s" % (self.get_timestr(), msg)
        self.logger.error(msg)
        self._write_file(msg)


def save_state_actions(state_action, filename):
    f = open(filename, 'wb')
    pickle.dump(state_action, f)
    f.close()

def get_value(key, obj):
    if key in obj:
        return obj[key]
    return None

def parse_cmd():
  args = sys.argv
  try:
    assert len(args) > 1, 'too few arguments'
    _, instant, exp, *others = args
    conf = yaml.load(
        Path(configs.PROJECT_DIR + '/environment/target/{}/tests/{}.yml'.format(instant, exp)).resolve().read_text(),  # pylint: disable=E1101
        Loader = yaml.FullLoader
    )

    regexp = re.compile(r'(.+)=(.+)')
    for other in others:
      match = regexp.match(other.strip())
      k, v = match.groups()
      assign(conf, k, v)
    return TestConfig(conf)
  except Exception as e:
    print(e.args)
    print(traceback.print_stack(e))
    print('Usage: python run.py <path_to_conf> [path.to.attr=value]')

class TestConfig:
  def __init__(self, obj):
    self.model_name = get_value('model_name', obj)
    self.memory = get_value('memory', obj)
    self.cm_path = get_value('cm_path', obj)
    self.workload = get_value('workload', obj)
    self.instance = get_value('instance', obj)
    self.method = get_value('method', obj)
    self.batch_size = get_value('batch_size', obj)
    self.epoches = get_value('epoches', obj)
    self.t = get_value('t', obj)
    self.max_steps = get_value('max_steps', obj)
    self.metric_num = get_value('metric_num', obj)
    self.default_knobs = get_value('default_knobs', obj)
    self.tps_weight = get_value('tps_weight', obj)
    self.LowDimSpace = None
    self.isToLowDim = get_value('isToLowDim', obj)
    if self.isToLowDim is True:
        LowDimSpace = {}
        LowDimSpace['target_dim'] = obj['LowDimSpace']['target_dim']
        LowDimSpace['low_method'] = obj['LowDimSpace']['low_method']
        LowDimSpace['seed'] = obj['LowDimSpace']['seed']
        self.LowDimSpace = LowDimSpace
    self.generate_csv = get_value('generate_csv', obj)
    self.n_client = get_value('n_client', obj)
    self.reward_mode = get_value('reward_mode', obj)
    self.iter_limit = get_value('iter_limit', obj)
    self.extra_vars = get_value('extra_vars', obj)
    self.task_name = get_value('task_name', obj)
    self.millde_memory = get_value('millde_memory', obj)

def assign(obj, path, value):
  keys = path.split('.')
  for k in keys[:-1]:
    v = obj.get(k)
    if v is None:
      obj[k] = {}
    elif type(v) is not dict:
      raise Exception(f'error while assigning {path} with {value} on {obj}.')
    obj = obj[k]
  try:
    value = int(value)
  except:
    pass
  if str(value) in ('yes', 'true'):
    value = True
  if str(value) in ('no', 'false'):
    value = False
  obj[keys[-1]] = value
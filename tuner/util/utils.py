# -*- coding: utf-8 -*-

import time
import pickle
import logging
import datetime
import yaml
import re
import traceback
from pathlib import Path
import os
import sys
import numpy as np
import torch
import json
import csv
import torch.optim as optim
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from environment import configs, knobs
from model import util, ddpg, sac, TD3, MTD3

def safemean(xs):
    '''
        Avoid division error when calculate the mean (in our case if
        epinfo is empty returns np.nan, not return an error)
    '''
    return np.nan if len(xs) == 0 else np.mean(xs)

class CSVWriter:

    def __init__(self, fname, fieldnames):

        self.fname = fname
        self.fieldnames = fieldnames
        self.csv_file = open(fname, mode='w')
        self.writer = None

    def write(self, data_stats):

        if self.writer == None:
            self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
            self.writer.writeheader()

        self.writer.writerow(data_stats)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()

def dump_to_json(path, data):
    '''
      Write json file
    '''
    with open(path, 'w') as f:
        json.dump(data, f)

def time_start():
    return time.time()


def time_end(start):
    end = time.time()
    delay = end - start
    return delay


def get_timestamp():
    """
    Get UNIX timestamp
    """
    return int(time.time())


def time_to_str(timestamp):
    """
    Convert timestamp to [YYYY-MM-DD HH:mm:ss] format
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
    self.threshold = get_value('threshold', obj)
    self.save_iter = get_value('save_iter', obj)
    self.seed = get_value('seed', obj)
    self.train_env_workload = get_value('train_env_workload', obj)
    self.test_env_workload = get_value('test_env_workload', obj)
    self.history_length = get_value('history_length', obj)
    self.cur_knobs_path = get_value('cur_knobs_path', obj)
    self.adaptation = get_value('adaptation', obj)
    self.adapt_length = get_value('adapt_length', obj)
    self.max_eval_length = get_value('max_eval_length', obj)
    self.rf_dict_path = get_value('rf_dict_path', obj)

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

def generate_knob(action, isToLowDim, episode, t, workload_name):
    return knobs.gen_continuous(action, isToLowDim, episode, t, workload_name)

def using_actor(model, state, episode, t, workload_name, isMeta, previous_action, previous_reward, previous_obs, isTrain):
    action_step_time = time_start()
    if isMeta is True:
        action = model.choose_action(state, previous_action, previous_reward, previous_obs, isTrain)
    else:
        action = model.choose_action(state, isTrain)
    action_step_time = time_end(action_step_time)
    # Action converted to knobs
    current_knob = generate_knob(action, False, episode, t, workload_name)

    return action_step_time, action, current_knob

def using_causality(model, cm, state, episode, t, workload_name, isMeta, previous_action, previous_reward, previous_obs, cur_knobs_dict, isTrain):
    action_step_time = time_start()
    cm.do_causal()
    action, config = cm.generate_knobs(episode, t, workload_name)
    if len(config) > 0:
        action_step_time = time_end(action_step_time)
        if cur_knobs_dict is not None:
            action = knobs.action_all_to_part(action)
        return action_step_time, np.array(action), config
    else:
        return using_actor(model, state, episode, t, workload_name, isMeta, previous_action, previous_reward, previous_obs, isTrain)


def get_new_konbs(model, cm, state, episode, t, logger, method, threshold, workload_name, isMeta=False, previous_action=None, previous_reward=None, previous_obs=None, cur_knobs_dict=None, isTrain=True):
    e_greed_decrement = 1e-6

    if isTrain and len(model.replay_memory) > threshold:
        sample = np.random.random()
        # Probability less than e Sampling action using causal inference
        if sample < cm.causal_memorys.e_greed:
            action_step_time, action, current_knob = using_causality(model, cm, state, episode, t, workload_name, isMeta, previous_action, previous_reward, previous_obs, cur_knobs_dict, isTrain)
        else:
            # Keep a probability of 0.05 to sample actions using causal inference
            if np.random.random() < 0.05:
                action_step_time, action, current_knob = using_causality(model, cm, state, episode, t, workload_name, isMeta, previous_action, previous_reward, previous_obs, cur_knobs_dict, isTrain)
            # According to the algorithm, select the action with the largest Q value under the current observation obs
            else:
                action_step_time, action, current_knob = using_actor(model, state, episode, t, workload_name, isMeta, previous_action, previous_reward, previous_obs, isTrain)
        # The probability e decays with the number of dynamic interactions between the agent and the environment
        cm.causal_memorys.update_e(max(0.05, cm.causal_memorys.e_greed - e_greed_decrement))
    else:
        action_step_time, action, current_knob = using_actor(model, state, episode, t, workload_name, isMeta, previous_action, previous_reward, previous_obs, isTrain)

    logger.info("[{}] Action: {}".format(method, action))

    return action_step_time, action, current_knob

def get_model(default_knobs, opt):
    return MTD3.meta_TD3(
        state_dim=opt.metric_num,
        action_dim=default_knobs,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device(
                "cpu"),
        buffer_size=100000,
        batch_size=opt.batch_size,
    )

def init_dir(instance):
    if not os.path.exists('log/' + instance):
        os.makedirs('log/' + instance)

    if not os.path.exists('save_knobs/' + instance):
        os.makedirs('save_knobs/' + instance)

    if not os.path.exists('save_state_actions/' + instance):
        os.makedirs('save_state_actions/' + instance)

    if not os.path.exists('model_params/' + instance):
        os.makedirs('model_params/' + instance)

    if not os.path.exists('save_memory/' + instance):
        os.makedirs('save_memory/' + instance)

    if not os.path.exists('causal_memory/' + instance):
        os.makedirs('causal_memory/' + instance)

    if not os.path.exists('save_multi_memory/' + instance):
        os.makedirs('save_multi_memory/' + instance)







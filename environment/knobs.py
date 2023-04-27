# -*- coding: utf-8 -*-
"""
desciption: Knob information

"""

from environment import configs, utils
import yaml
import numpy as np
from pathlib import Path
import ConfigSpace.hyperparameters as CSH
import ConfigSpace as CS
from environment.adapters import *
import random

memory_size = utils.read_machine()
# Application name
instance_name = ''

KNOBS = []


KNOB_DETAILS = None
os_setting = None
app_setting =None
input_space_adapter = None
low_method = None
cur_knobs = None
not_cur_app_setting = dict()
not_cur_os_setting = dict()

PROJECT_DIR = configs.PROJECT_DIR

db_dir = None
result_dir = None

def init_knobs(instance, task_name, lowDimSpace, cur_knobs_dict):
    global instance_name
    global memory_size
    global KNOB_DETAILS
    global app_setting
    global os_setting
    global db_dir
    global result_dir
    global input_space_adapter
    global low_method
    global cur_knobs

    cur_knobs = cur_knobs_dict
    instance_name = instance

    db_dir = PROJECT_DIR + '/environment/target/' + instance_name
    result_dir = Path(db_dir + '/results/' + task_name)

    memory_size = utils.read_machine()

    os_setting_path = PROJECT_DIR + '/environment/target/' + instance_name + '/os_configs_info.yml'
    app_setting_path = PROJECT_DIR + '/environment/target/' + instance_name + '/app_configs_info.yml'

    os_setting = yaml.load(open(os_setting_path, 'r', encoding='UTF-8'),
                           Loader=yaml.FullLoader)  # pylint: disable=E1101
    app_setting = yaml.load(open(app_setting_path, 'r', encoding='UTF-8'),
                            Loader=yaml.FullLoader)  # pylint: disable=E1101

    KNOB_DETAILS = {**(app_setting), **(os_setting)}

    print("Instance: %s Memory: %s" % (instance_name, memory_size))

    # 高维配置参数映射为低维
    if lowDimSpace is not None:
        print('enable mapping mechanism')
        bias_prob_sv = None  # 考虑特殊值才有用
        definition = get_cs_knobs()
        input_space = generate_input_space(
            knobs=definition,
            bias_prob_sv=bias_prob_sv,
            seed=lowDimSpace['seed'],
            ignore_extra_knobs=None)

        low_method = lowDimSpace['low_method']

        input_space_adapter = LinearEmbeddingConfigSpace.create(
            input_space, lowDimSpace['seed'],
            method=lowDimSpace['low_method'],
            target_dim=lowDimSpace['target_dim'],
            bias_prob_sv=bias_prob_sv,
            max_num_values=None)

def get_KNOB_DETAILS():
    global KNOB_DETAILS
    return KNOB_DETAILS

def get_knobs_keys():
    global KNOB_DETAILS
    KNOBS = []
    for k, v in KNOB_DETAILS.items():
        KNOBS.append(k)
    return KNOBS

def generate_input_space(knobs, bias_prob_sv, seed: int, ignore_extra_knobs=None):
    """
    :param knobs: all configuration parameters
    :param adapter_alias: mapping method
    :param _bias_prob_sv: >0&&<1 If special values need to be considered, here is to set the value of the first few percent as special values
    :param le_low_dim: dimension after mapping
    :param seed:
    :param ignore_extra_knobs:
    :return:
    """
    ignore_extra_knobs = ignore_extra_knobs or []
    knob_types = ['enum', 'integer', 'real']

    input_dimensions = []
    for info in knobs:
        name, knob_type = info['name'], info['type']
        if name in ignore_extra_knobs:
            continue

        if knob_type not in knob_types:
            raise NotImplementedError(f'Knob type of "{knob_type}" is not supported :(')

        ## Categorical
        if knob_type == 'enum':
            dim = CSH.CategoricalHyperparameter(
                name=name,
                choices=info['choices'],
                default_value=info['default'])
        ## Numerical
        elif knob_type == 'integer':
            dim = CSH.UniformIntegerHyperparameter(
                name=name,
                lower=info['min'],
                upper=info['max'],
                default_value=info['default'])
        elif knob_type == 'real':
            dim = CSH.UniformFloatHyperparameter(
                name=name,
                lower=info['min'],
                upper=info['max'],
                default_value=info['default'])

        input_dimensions.append(dim)

    input_space = CS.ConfigurationSpace(name="input", seed=seed)
    input_space.add_hyperparameters(input_dimensions)

    if bias_prob_sv is not None:
        # biased sampling
        input_space = PostgresBiasSampling(
            input_space, seed, bias_prob_sv).target

    return input_space

def get_os_app_setting():
    global app_setting
    global os_setting
    return app_setting, os_setting

def get_cs_knobs():
    global KNOB_DETAILS
    definition = []
    for key in KNOB_DETAILS.keys():
        v = KNOB_DETAILS[key]
        item = {}
        item['name'] = key
        item['default'] = v['default']
        if v.get('range'):
            item['type'] = 'enum'
            item['choices'] = v['range']
            if type(item['default']) is bool:
                for idx in range(len(item['choices'])):
                    item['choices'][idx] = str(item['choices'][idx]).lower()
                item['default'] = str(item['default']).lower()
        else:
            if v.get('float'):
                item['type'] = 'real'
            else:
                item['type'] = 'integer'
            item['min'] = v['min']
            item['max'] = v['max']
        if v.get('bucket_num'):
            item['bucket_num'] = v['bucket_num']
        definition.append(item)
    return definition

def format_value(default_value):
    if type(default_value) is bool:
        # make sure no uppercase 'True/False' literal in result
        return str(default_value).lower()
    elif type(default_value) is np.float64:
        return float(default_value)
    else:
        return default_value

def get_init_knobs():
    global app_setting
    global os_setting
    global not_cur_app_setting
    global not_cur_os_setting
    global cur_knobs

    sampled_app_config = {}
    sampled_os_config = {}

    for name, value in app_setting.items():
        sampled_app_config[name] = format_value(value['default'])
        if cur_knobs is not None:
            if name not in cur_knobs['cur_soft']:
                not_cur_app_setting[name] = format_value(value['default'])

    for name, value in os_setting.items():
        sampled_os_config[name] = format_value(value['default'])
        if cur_knobs is not None:
            if name not in cur_knobs['cur_kernel']:
                not_cur_os_setting[name] = format_value(value['default'])

    result_dir.mkdir(parents=True, exist_ok=True)

    # - dump configs
    os_config_path = result_dir / f'0_os_config_init_0.yml'
    os_config_path.write_text(
        yaml.dump(sampled_os_config, default_flow_style=False)
    )
    app_config_path = result_dir / f'0_app_config_init_0.yml'
    app_config_path.write_text(
        yaml.dump(sampled_app_config, default_flow_style=False)
    )
    utils._print(f'os_config & app_config inited.')

    return {**(sampled_app_config), **(sampled_os_config)}


def action_all_to_part(action):
    global cur_knobs

    new_action =[]

    for idx, value in enumerate(KNOB_DETAILS.items()):
        k, v = value
        if cur_knobs is not None:
            if k in cur_knobs['cur_soft'] or k in cur_knobs['cur_kernel']:
                new_action.append(action[idx])
        else:
            new_action.append(action[idx])
    return new_action

def action_part_to_all(action, rf_dict):
    new_action =[]
    count = 0
    for idx, value in enumerate(KNOB_DETAILS.items()):
        k, v = value
        if k in rf_dict['cur_soft'] and k in rf_dict['cur_kernel']:
            new_action.append(action[count])
            count += 1
        else:
            if v.get('range'):
                new_action.append((v['range'].index(v['default']) + random.random()) / len(v['range']))
            else:
                max_val = v['max']
                min_val = v['min']
                if v.get('bucket_num'):
                    new_action.append(((v['default'] - min_val) * v['bucket_num'] / (max_val - min_val) + 0.5) / v['bucket_num'])
                else:
                    new_action.append((v['default'] - min_val) / (max_val - min_val))
    return new_action

def action_to_knobs(action, isToLowDim):
    global app_setting
    global os_setting
    global input_space_adapter
    global low_method
    global cur_knobs
    global not_cur_app_setting
    global not_cur_os_setting

    sampled_app_config = {}
    sampled_os_config = {}

    if isToLowDim is True:
        points = {}
        for idx, value in enumerate(action):
            points[f'{low_method}_{idx}'] = value * 2 - 1  # Convert from (0, 1) to (-1, 1)
        all_knobs = input_space_adapter.unproject_point(points)
        all_knobs_keys = list(all_knobs.keys())
        for k in app_setting:
            if k in all_knobs_keys:
                sampled_app_config[k] = all_knobs[k]
        for k in os_setting:
            if k in all_knobs_keys:
                sampled_os_config[k] = all_knobs[k]
    else:
        action_idx = 0
        for idx, value in enumerate(KNOB_DETAILS.items()):
            k, v = value
            if cur_knobs is not None:
                if k not in cur_knobs['cur_soft'] and k not in cur_knobs['cur_kernel']:
                    if k in app_setting:
                        sampled_app_config[k] = not_cur_app_setting[k]
                    else:
                        sampled_os_config[k] = not_cur_os_setting[k]
                    continue
            v_range = v.get('range')
            if v_range:  # discrete ranged parameter
                enum_size = len(v['range'])
                enum_index = int(enum_size * action[action_idx])
                enum_index = min(enum_size - 1, enum_index)
                eval_value = v['range'][enum_index]
            else:
                max_val = v['max']
                min_val = v['min']
                if v.get('float'):
                    if v.get('bucket_num'):
                        eval_value = (max_val - min_val) / v['bucket_num'] * int(
                            v['bucket_num'] * action[action_idx]) + min_val
                    else:
                        eval_value = (max_val - min_val) * action[action_idx] + min_val
                else:
                    if v.get('bucket_num'):
                        eval_value = int(
                            (max_val - min_val) / v['bucket_num'] * int(v['bucket_num'] * action[action_idx]) + min_val)
                    else:
                        eval_value = int((max_val - v['min']) * action[action_idx] + min_val)

            action_idx += 1

            if k in app_setting:
                sampled_app_config[k] = format_value(eval_value)
            else:
                sampled_os_config[k] = format_value(eval_value)

    return sampled_os_config, sampled_app_config


def gen_continuous(action, isToLowDim, episode, t, workload_name):
    sampled_os_config, sampled_app_config = action_to_knobs(action, isToLowDim)
    # - dump configs
    dump_configs(sampled_os_config, sampled_app_config, episode, t, workload_name)

    return {**(sampled_app_config), **(sampled_os_config)}


def dump_configs(sampled_os_config, sampled_app_config, episode, t, workload_name):
    os_config_path = result_dir / f'{episode}_os_config_{workload_name}_{t}.yml'
    os_config_path.write_text(
        yaml.dump(sampled_os_config, default_flow_style=False)
    )
    app_config_path = result_dir / f'{episode}_app_config_{workload_name}_{t}.yml'
    app_config_path.write_text(
        yaml.dump(sampled_app_config, default_flow_style=False)
    )
    utils._print(f'os_config & app_config generated.')

def get_app_num():
    global app_setting
    return len(app_setting)

def save_knobs(knob, metrics, knob_file):
    """ Save Knobs and their metrics to files
    Args:
        knob: dict, knob content
        metrics: list, tps and latency
        knob_file: str, file path
    """
    # format: tps, latency, knobstr: [#knobname=value#]
    knob_strs = []
    for kv in knob.items():
        knob_strs.append('{}:{}'.format(kv[0], kv[1]))
    result_str = '{},{}'.format(metrics[0], metrics[1])
    knob_str = "#".join(knob_strs)
    result_str += knob_str

    with open(knob_file, 'a+') as f:
        f.write(result_str+'\n')


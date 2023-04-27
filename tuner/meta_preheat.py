import yaml
import re
import sys
import numpy as np
from pathlib import Path
from util import utils
from statistics import mean
import os
import copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from environment.utils import run_playbook, parse_result, _print, get_default, compute_result
from environment import configs, appEnv, knobs
from model.optimizer import create_optimizer
from model.multi_middle_memory import MultiMiddleMemory
import pickle
from CausalModel import causaler


repitition = 1

def find_exist_task_result():
  task_id = -1
  step_id = -1
  regexp = re.compile(r'(\d+)_run_result_(\d+)')
  if result_dir.exists():
    for p in result_dir.iterdir():
      if p.is_file():
        res = regexp.match(p.name)
        if res:
          task_id = max(task_id, int(res.group(1)))
          step_id = max(step_id, int(res.group(2)))
  return task_id, step_id


def divide_config(sampled_config, os_setting, app_setting):
  for k in sampled_config.keys():
    if type(sampled_config[k]) is bool:
      # make sure no uppercase 'True/False' literal in result
      sampled_config[k] = str(sampled_config[k]).lower()
    elif type(sampled_config[k]) is np.float64:
      sampled_config[k] = float(sampled_config[k])

  sampled_os_config = dict(
      ((k, v) for k, v in sampled_config.items() if k in os_setting)
  )
  sampled_app_config = dict(
      ((k, v) for k, v in sampled_config.items() if k in app_setting)
  )
  return sampled_os_config, sampled_app_config

def multi_item(tasks, data):
    return dict([(idx, copy.deepcopy(data)) for idx in tasks])

def main(test_config, env, init_task_id, init_step_id, os_setting, app_setting):
  # print("save_iter: ",test_config.save_iter)
  knobs_head = env.get_knobs_keys()
  knobs_info = knobs.get_KNOB_DETAILS()
  objects_head = ['tps', 'latency']
  cm = None

  multi_middle_memory = MultiMiddleMemory(test_config.train_env_workload)

  if test_config.millde_memory != '':
    multi_middle_memory.load_memory('multi_middle_memory/' + test_config.instance + '/' + test_config.millde_memory + '.pkl')
    print("Load Memory:")
    print(multi_middle_memory.size_all())

  if not os.path.exists('BO_params/' + opt.instance):
    os.makedirs('BO_params/' + test_config.instance)
  if not os.path.exists('save_memory/' + opt.instance):
    os.makedirs('save_memory/' + test_config.instance)
  if not os.path.exists('causal_memory/' + opt.instance):
    os.makedirs('causal_memory/' + test_config.instance)
  if not os.path.exists('multi_middle_memory/' + opt.instance):
    os.makedirs('multi_middle_memory/' + test_config.instance)

  expr_name = 'train_{}_{}'.format(test_config.method, str(utils.get_timestamp()))

  task_id = 0 if init_task_id == -1 else init_task_id

  for tidx in range(task_id, len(test_config.train_env_workload)):
      cur_task = test_config.train_env_workload[tidx]
      env.set_task(cur_task)
      step_id = 0

      default_metric = [0.0, 0.0]
      last_metric = [0.0, 0.0]
      best_metric = [0.0, 0.0]
      best_result = 0
      decline_num = 0
      state = None

      # create optimizer
      optimizer = create_optimizer(
          test_config.method,
          {
              **os_setting, **app_setting,
          },
          extra_vars=test_config.extra_vars
      )

      if test_config.model_name != '':
          if os.path.exists('BO_params/' + test_config.instance + '/' + test_config.model_name + '_' + cur_task + '.pth'):
              with open('BO_params/' + test_config.instance + '/' + test_config.model_name + '_' + cur_task + '.pth', 'rb') as f:
                  data = pickle.load(f)
              optimizer = data['optimizer']
              default_metric = data['default_metric']
              best_metric = data['best_metric']
              best_result = data['best_result']
              decline_num = data['decline_num']
              print("BO_params load secceed!")

      while step_id < test_config.iter_limit and decline_num <= 50:      # Optimization times
        _print(f'{tidx} - {step_id}: decline_num is {decline_num}.')

        workload_name = cur_task
        # - sample config
        if step_id == 0:  # use default config
          sampled_config_numeric, sampled_config = None, get_default(app_setting)   # Get the default configuration
          workload_name = 'init'
        else:
          try:
            sampled_config_numeric, sampled_config = optimizer.get_conf()           # get new configuration
          except StopIteration:
            # all configuration emitted
            return

        # - divide sampled config app & os
        sampled_os_config, sampled_app_config = divide_config(
            sampled_config,
            os_setting=os_setting,
            app_setting=app_setting
        )
        current_knob = {**sampled_app_config, **sampled_os_config}

        # - dump configs
        os_config_path = result_dir / f'{tidx}_os_config_{workload_name}_{step_id}.yml'
        os_config_path.write_text(
            yaml.dump(sampled_os_config, default_flow_style=False)
        )
        app_config_path = result_dir / f'{tidx}_app_config_{workload_name}_{step_id}.yml'
        app_config_path.write_text(
            yaml.dump(sampled_app_config, default_flow_style=False)
        )
        _print(f'{tidx} - {step_id}: os_config & app_config generated.')

        metric_results = []
        skip = False
        fail = False
        for rep in range(repitition):
          next_state, result = single_test(
              env=env,
              task_id=tidx,
              step_id=step_id,
              first=(step_id == 0),
              _skip=skip
          )
          if result['avg'][0] == 0 and result['avg'][1] == 0:
              print("deployment failed")
              fail = True
              break
          # Generate table head
          if cm is None:
            states_head = env.get_stats_keys()
            print("states_head:", len(states_head), next_state)
            cm = causaler.Causality(knobs_head, states_head, objects_head, knobs_info,
                                        'causal_memory/' + test_config.instance + '/' + test_config.cm_path + '.pkl')

          if step_id != 0:
            knob_value = []
            for item in knobs_head:
                knob_value.append(current_knob[item])
            action = cm.system_to_action(knob_value)
            memory_data = (state, action, next_state, False, result)
            # print(memory_data)
            multi_middle_memory.add(cur_task, memory_data)
            datas = knob_value + list(next_state)
            datas.append(cur_task)
            datas.append(result['avg'][0])
            datas.append(result['avg'][1])
            cm.update_data(datas)

          state = next_state

          if step_id != 0 and result['avg'][0] is not None and result['avg'][0] != 0.:
            reward = compute_result(result['avg'], default_metric, last_metric, test_config.tps_weight)
            metric_results.append(reward)


          if step_id != 0:
              if best_result < result['avg'][0]:
                  best_result = result['avg'][0]
                  best_metric = result['avg']
                  decline_num = 0
              else:
                  decline_num += 1
              print('*****************************')
              print("current_performance(tps, lat):", result['avg'][0], result['avg'][1])
              print("default_performance(tps, lat):", default_metric[0], default_metric[1])
              print("last_performance(tps, lat):", last_metric[0], last_metric[1])
              print("best_performance(tps, lat):   ", best_metric[0], best_metric[1])
              print('*****************************')
          else:
              env.default_externam_metrics = result['avg']
              default_metric = result['avg']

          last_metric = result['avg']

        if fail == True:
            continue

        # after
        if step_id != 0:  # not adding default info, 'cause default cannot convert to numeric form
          metric_result = mean(metric_results) if len(metric_results) > 0 else .0

          optimizer.add_observation(
              (sampled_config_numeric, metric_result)
          )


          data = {
            'optimizer': optimizer,
            'default_metric': default_metric,
            'best_metric': best_metric,
            'best_result': best_result,
            'decline_num': decline_num
          }
          f = open('BO_params/' + test_config.instance + '/' + expr_name + '_' + cur_task + '.pth', 'wb')
          pickle.dump(data, f)
          f.close()

          # if step_id in test_config.save_iter:
          #     _print(f'{step_id} memorys have been generated!!')
          #     multi_middle_memory.save('multi_middle_memory/{}/{}.pkl'.format(test_config.instance, expr_name + "_" + str(step_id)))
          #     cm.causal_memorys.save('causal_memory/{}/{}.pkl'.format(test_config.instance, expr_name + "_" + cur_task + "_" + str(step_id)))
          multi_middle_memory.save('multi_middle_memory/{}/{}.pkl'.format(test_config.instance, expr_name))
          cm.causal_memorys.save('causal_memory/{}/{}.pkl'.format(test_config.instance, expr_name))

        step_id += 1


def single_test(env, task_id, step_id, first, _skip=False):
  # for debugging...
  if _skip:
    return
  flag = env._apply_knobs(first, task_id, step_id)

  if not flag:
      return np.array([0] * env.num_metric), {'avg': [0, 0]}

  s = env._get_state(task_id, step_id, generate_keys=first)

  if s is None:
      return np.array([0] * env.num_metric), {'avg': [0, 0]}

  external_metrics, internal_metrics = s
  return internal_metrics, external_metrics


# -------------------------------------------------------------------------------------------------------

PROJECT_DIR = configs.PROJECT_DIR
# calculate paths
proj_root = Path(PROJECT_DIR).resolve()
opt = utils.parse_cmd()

# calculate paths
proj_root = Path(__file__, '../..').resolve()

user = configs.instance_config[opt.instance]['user']
pwd = configs.instance_config[opt.instance]['password']
db_dir = proj_root / f'environment/target/{opt.instance}'
result_dir = db_dir / f'results/{opt.task_name}'
deploy_playbook_path = db_dir / 'playbook/deploy.yml'
tester_playbook_path = db_dir / 'playbook/tester.yml'
osconfig_playbook_path = db_dir / 'playbook/set_os.yml'
reboot_playbook_path = db_dir / 'playbook/reboot.yml'
workload_path = db_dir / f'workload/{opt.workload}'
os_setting_path = db_dir / f'os_configs_info.yml'
app_setting_path = db_dir / f'app_configs_info.yml'


init_task_id = -1
init_step_id = -1


# check existing results, find minimum available task_id
exist_task_id, exist_step_id= find_exist_task_result()
if exist_task_id != -1:
  _print(f'previous results found, with max task_id={exist_task_id}')
  if opt.model_name == '':
    for file in sorted(result_dir.glob('*')):
      file.unlink()
    _print('all deleted')
  else:
    _print(f'continue with task_id={exist_task_id + 1}')
    init_task_id = exist_task_id
    init_step_id = exist_step_id

# create dirs
result_dir.mkdir(parents=True, exist_ok=True)


# read parameters for tuning
os_setting = yaml.load(os_setting_path.read_text(),
                           Loader=yaml.FullLoader)  # pylint: disable=E1101
app_setting = yaml.load(app_setting_path.read_text(),
                           Loader=yaml.FullLoader)  # pylint: disable=E1101

# Create Environment
env = appEnv.Server(
    wk_type=opt.workload,
    instance_name=opt.instance,
    num_metric=opt.metric_num,
    tps_weight=opt.tps_weight,
    lowDimSpace=opt.LowDimSpace,
    task_name=opt.task_name,
    cur_knobs_dict=None,
    n_client=opt.n_client
)

main(
    test_config=opt,
    env=env,
    init_task_id=init_task_id,
    init_step_id=init_step_id,
    os_setting=os_setting,
    app_setting=app_setting
)


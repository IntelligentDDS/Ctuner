# -*- coding: utf-8 -*-
"""
description: App Environment
"""
import re
import os
import time
import math
import numpy as np
import asyncio
from pathlib import Path
from environment import configs, utils, knobs

class AppEnv(object):

    def __init__(self, wk_type='small', num_metric=48, tps_weight = 0.4, alpha=1.0, beta1=0.5, beta2=0.5, time_decay1=1.0, time_decay2=1.0):
        self.db_info = None
        self.wk_type = wk_type
        self.score = 0.0
        self.steps = 0
        self.terminate = False
        self.last_external_metrics = None
        self.default_externam_metrics = None

        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.time_decay_1 = time_decay1
        self.time_decay_2 = time_decay2
        self.num_metric = num_metric
        self.tps_weight = tps_weight
        self.stat_keys = []
        self.test_time = 0

        self._period = 2       # how often to get status
        self.count = 12 / 2   # How many times to get the state for each step

    @staticmethod
    def _get_external_metrics(path):
        with open(path) as f:
            lines = f.read()
        latency = re.compile(r'95thPercentileLatency\(us\), (\d+(?:\.\d+)?)').findall(lines)
        latency_num = 0
        latency_avg = 0
        for i in latency:
            latency_num += 1
            latency_avg += float(i)
        latency_avg /= latency_num
        tps = float(re.compile(r'\[OVERALL\], Throughput\(ops/sec\), (\d+(?:\.\d+)?)').search(lines).group(1))
        return [tps, latency_avg]

    async def _get_internal_metrics(self, internal_metrics, counter):
        """
        Args:
            internal_metrics: list,
        Return:

        """
        counter += 1
        # print(counter, "get_internal_metrics")
        if counter != 1 and counter < self.count:
            await asyncio.sleep(self._period)
            await self._get_internal_metrics(internal_metrics, counter)
        try:
            data = {}
            await utils.get_os_metrics(self.db_info, data, self.db_dir)
            if self.test_mode is 'synchronization':
                await utils.get_app_metrics(self.db_info, data, self.instance_name, self.db_dir)
            internal_metrics.append(data)
        except Exception as err:
            print("[GET Metrics]Exception:", err)

    def _post_handle(self, metrics):
        result = np.zeros(self.num_metric)

        def do(metric_values):
            if type(metric_values[0]) == int:
                return int(sum(metric_values)/len(metric_values))
            else:
                return float(sum(metric_values)/len(metric_values))

        keys = metrics[0].keys()
        # print("keys:", len(list(keys)))
        for idx, key in enumerate(keys):
            data = [x[key] for x in metrics]
            result[idx] = do(data)
        return result

    def initialize(self):
        """Initialize the mysql instance environment
        """
        pass

    def eval(self, episode, t, logger):
        """ Evaluate the knobs
        Args:
            knob: dict, mysql parameters
        Returns:
            result: {tps, latency}
        """
        flag = self._apply_knobs(False, episode, t)
        if not flag:
            return {"tps": 0, "latency": 0}

        external_metrics, _ = self._get_state(episode, t, logger)
        return {"tps": external_metrics[0],
                "latency": external_metrics[1]}

    def _get_best_now(self):
        filename = self.task_name + '_' + self.wk_type + '_bestnow.log'
        with open(self.result_dir  + filename) as f:
            lines = f.readlines()
        best_now = lines[0].split(',')
        return [float(best_now[0]), float(best_now[1]), float(best_now[0])]

    def record_best(self, external_metrics):
        filename = self.task_name + '_' + self.wk_type + '_bestnow.log'
        best_flag = False
        if os.path.exists( self.result_dir + filename):
            tps_best = external_metrics[0]
            lat_best = external_metrics[1]
            if int(lat_best) != 0:
                rate = float(tps_best)/lat_best
                with open(self.result_dir + filename) as f:
                    lines = f.readlines()
                best_now = lines[0].split(',')
                rate_best_now = float(best_now[0])/float(best_now[1])
                if rate > rate_best_now:
                    best_flag = True
                    with open(self.result_dir + filename, 'w') as f:
                        f.write(str(tps_best) + ',' + str(lat_best) + ',' + str(rate))
        else:
            file = open(self.result_dir + filename, 'w+')
            tps_best = external_metrics[0]
            lat_best = external_metrics[1]
            if int(lat_best) == 0:
                rate = 0
            else:
                rate = float(tps_best)/lat_best
            file.write(str(tps_best) + ',' + str(lat_best) + ',' + str(rate))
        return best_flag

    def step(self, episode, t, reward_mode, logger):
        """step
        """
        restart_time = utils.time_start()
        knobs_start_times = utils.time_start()
        flag = self._apply_knobs(False, episode, t)
        knobs_end_times = utils.time_end(knobs_start_times)

        restart_time = utils.time_end(restart_time)

        if not flag:
            return -1000.0, np.array([0] * self.num_metric), True, self.score - 1000, {'0': [0.0, 0.0], 'avg': [0.0, 0.0]}, restart_time
        test_start_times = utils.time_start()
        s = self._get_state(episode, t, logger)
        test_end_times = utils.time_end(test_start_times)

        if s is None:
            return -1000.0, np.array([0] * self.num_metric), True, self.score - 1000, {'0': [0.0, 0.0], 'avg': [0.0, 0.0]}, restart_time
        external_metrics, internal_metrics = s

        reward = self._get_reward(external_metrics['avg'], reward_mode)

        next_state = internal_metrics
        terminate = self._terminate()
        return reward, next_state, terminate, self.score, external_metrics, restart_time

    def set_task(self, wk_type):
        utils._print(f'Current workload type: {wk_type}')
        self.wk_type = wk_type

    def setting(self, first, episode, t):
        self._apply_knobs(first, episode, t)

    async def ycsb_test(self, episode, step, rep, workload_name):
        utils._print(f'{episode} - {step} - {rep}: testing...')
        self.test_time = utils.time_start()
        await utils.run_playbook(
            self.tester_playbook_path,
            host=self.tester,
            target=self.testee,
            project_dir=self.PROJECT_DIR,
            task_id=episode,
            workload_name=workload_name,
            task_name=self.task_name,
            task_step=step,
            task_rep=rep,
            user=configs.instance_config[self.instance_name]['user'],
            pwd=configs.instance_config[self.instance_name]['password'],
            workload_path=self.db_dir + '/workload/' + self.wk_type,
            n_client=self.n_client
        )
        self.test_time = utils.time_end(self.test_time)
        utils._print(f'{episode} - {step} - {rep}: done.')

    async def load_status(self, internal_metrics, episode, step, rep, workload_name):
        _counter = 0
        await self._get_internal_metrics(internal_metrics, _counter)
        _counter += 1
        tasks_list = [
            asyncio.get_event_loop().create_task(self.ycsb_test(episode, step, rep, workload_name)),
            asyncio.get_event_loop().create_task(self._get_internal_metrics(internal_metrics, _counter))
        ]
        await asyncio.gather(*tasks_list)

    def _get_state(self, episode, step, logger=None, generate_keys=False):
        """Collect the Internal State and External State
        """
        workload_name = self.wk_type
        if generate_keys is True:
            workload_name = 'init'

        init_external_metrics = None
        init_internal_metrics = None

        iteration = 1
        decline = 0.5
        all_external_metrics = dict()

        for rep in range(iteration):
            internal_metrics = []
            asyncio.get_event_loop().run_until_complete(self.load_status(internal_metrics, episode, step, rep, workload_name))

            if generate_keys is True:
                self.stat_keys = list(internal_metrics[0].keys())

            # print("states:",len(self.stat_keys))

            run_result_path = self.PROJECT_DIR + "/environment/target/" + self.instance_name + "/results/" + self.task_name + "/" +str(episode) +"_run_result_"  + workload_name + '_' + str(step) + '_' + str(rep)

            external_metrics = self._get_external_metrics(run_result_path)
            internal_metrics = self._post_handle(internal_metrics)

            if self.test_mode is not 'synchronization':
                data = {}
                asyncio.get_event_loop().run_until_complete(utils.get_app_metrics(self.db_info, data, self.instance_name, self.db_dir))
                keys = data.keys()
                for idx, key in enumerate(keys):
                    v = data[key]
                    if generate_keys is True:
                        self.stat_keys.append(key)
                    internal_metrics[7 + idx] = v

            if logger is not None:
                logger.info(
                    "\n[Episode: {}][Workload:{}][Step: {}][Metric tps_{}:{} lat_{}:{}]".format(
                        episode, self.wk_type, step, rep, external_metrics[0], rep, external_metrics[1]
                    ))

            all_external_metrics[str(rep)] = external_metrics
            if rep == 0:
                init_external_metrics = np.array(external_metrics, dtype=float)
                init_internal_metrics = np.array(internal_metrics, dtype=float)
            else:
                init_external_metrics += np.array(external_metrics, dtype=float)
                init_internal_metrics += np.array(internal_metrics, dtype=float)

            if generate_keys is False:
                if self.default_externam_metrics[1] * (1 + decline) <= external_metrics[1]:
                    break

        init_external_metrics /= len(all_external_metrics)
        init_internal_metrics /= len(all_external_metrics)

        asyncio.get_event_loop().run_until_complete(utils.clean_os_knobs(episode, step, self.instance_name, workload_name))

        all_external_metrics['avg'] = list(init_external_metrics)

        return all_external_metrics, init_internal_metrics

    def get_stats_keys(self):
        return self.stat_keys

    def get_test_time(self):
        return self.test_time

    def _apply_knobs(self, first, episode, t, logger):
        """ Apply Knobs to the instance
        """
        pass

    @staticmethod
    def _calculate_CDB(delta0, deltat):

        if delta0 > 0:
            _reward = ((1+delta0)**2-1) * math.fabs(1+deltat)
        else:
            _reward = - ((1-delta0)**2-1) * math.fabs(1-deltat)

        if _reward > 0 and deltat < 0:
            _reward = 0
        return _reward

    def _get_CDBTune_reward(self, external_metrics):
        # tps
        delta_0_tps = float((external_metrics[0] - self.default_externam_metrics[0])) / self.default_externam_metrics[0]
        delta_t_tps = float((external_metrics[0] - self.last_external_metrics[0])) / self.last_external_metrics[0]

        tps_reward = self._calculate_CDB(delta_0_tps, delta_t_tps)

        # latency
        delta_0_lat = float((-external_metrics[1] + self.default_externam_metrics[1])) / self.default_externam_metrics[
            1]
        delta_t_lat = float((-external_metrics[1] + self.last_external_metrics[1])) / self.last_external_metrics[1]

        lat_reward = self._calculate_CDB(delta_0_lat, delta_t_lat)

        return tps_reward, lat_reward


    def _get_sample_reward(self, external_metrics):
        tps_reward = float(external_metrics[0] / self.default_externam_metrics[0]) - 1
        lat_reward = float(self.default_externam_metrics[1] / external_metrics[1]) - 1

        return tps_reward, lat_reward

    @staticmethod
    def _calculate_CDL(delta):
        if delta > 0:
            return 10 * (2**delta - 1)
        else:
            return 10 * (-1 * (1/2)**delta + 1)

    def _get_CDLTune_reward(self, external_metrics):
        if int(external_metrics[0]) == 0 and int(external_metrics[1]) == 0:
            return -50.0, -50.0

        delta_0_tps = float(external_metrics[0] / self.default_externam_metrics[0]) - 1
        tps_reward = self._calculate_CDL(delta_0_tps)

        delta_0_lat = float(self.default_externam_metrics[1] / external_metrics[1]) - 1
        lat_reward = self._calculate_CDL(delta_0_lat)

        return tps_reward, lat_reward

    def _get_reward(self, external_metrics, mode, prehead=False):
        """
        Args:
            external_metrics: list, external metric info, including `tps` and `qps`
        Return:
            reward: float, a scalar reward
        """
        flag = self.record_best(external_metrics)
        if flag == True:
            print('Better performance changed!')
        else:
            print('Performance remained!')
        # get the best performance so far to calculate the reward
        best_now_performance = self._get_best_now()
        self.last_external_metrics = best_now_performance

        if prehead is True:
            print('***********prehead***********')
        else:
            print('************train************')
        print("current_performance(tps, lat):", external_metrics[0], external_metrics[1])
        print("default_performance(tps, lat):", self.default_externam_metrics[0], self.default_externam_metrics[1])
        print("best_performance(tps, lat):   ", best_now_performance[0], best_now_performance[1])
        print('*****************************')
        if mode == 1:
            tps_reward, lat_reward = self._get_CDBTune_reward(external_metrics);
        elif mode == 2:
            tps_reward, lat_reward = self._get_sample_reward(external_metrics);
        elif mode == 3:
            tps_reward, lat_reward = self._get_CDLTune_reward(external_metrics);

        reward = tps_reward * self.tps_weight + (1 - self.tps_weight) * lat_reward
        self.score += reward
        print('$$$$$$$$$$$$$$$$$$$$$$')
        print("tps_reward:", tps_reward)
        print("lat_reward:", lat_reward)
        print("reward:", reward)
        print('$$$$$$$$$$$$$$$$$$$$$$')
        self.last_external_metrics = external_metrics
        return reward

    def _terminate(self):
        return self.terminate

    def find_exist_task_result(self):
        result_dir = Path(self.result_dir)
        task_id = -1
        regexp = re.compile(r'(\d+)_.+')
        if result_dir.exists():
            for p in result_dir.iterdir():
                if p.is_file():
                    res = regexp.match(p.name)
                    if res:
                        task_id = max(task_id, int(res.group(1)))
        return (None if task_id == -1 else task_id), result_dir


class Server(AppEnv):
    """ Build an environment directly on Server
    """

    def __init__(self, wk_type, instance_name, num_metric, tps_weight, lowDimSpace, task_name, cur_knobs_dict, n_client = 16):
        AppEnv.__init__(self, wk_type, num_metric, tps_weight)

        self.wk_type = wk_type
        self.task_name = task_name
        self.score = 0.0
        self.steps = 0
        self.terminate = False
        self.last_external_metrics = None
        self.instance_name = instance_name

        self.PROJECT_DIR = configs.PROJECT_DIR
        self.n_client = n_client
        self.db_dir = self.PROJECT_DIR + '/environment/target/' + self.instance_name
        self.BEST_NOW = self.db_dir + '/'
        self.result_dir = self.db_dir + '/results/' + task_name + '/'
        self.setting_path = self.db_dir + '/os_configs_info.yml'
        self.osconfig_playbook_path = self.db_dir + '/playbook/set_os.yml'
        self.tester_playbook_path = self.db_dir + '/playbook/tester.yml'


        self.db_info = configs.instance_config[instance_name]
        self.tester = self.db_info['tester']
        self.testee = self.db_info['testee']
        self.test_mode = self.db_info['test_mode']
        self.alpha = 1.0
        knobs.init_knobs(instance_name, task_name, lowDimSpace,  cur_knobs_dict)

    def get_knobs_keys(self):
        return knobs.get_knobs_keys()

    def initialize(self, logger):
        """ Initialize the environment when an episode starts
        Returns:
            state: np.array, current state
        """
        self.score = 0.0
        self.last_external_metrics = []
        self.steps = 0
        self.terminate = False
        self.init_flag_num = 0

        init_flag = self._apply_knobs(True, 0, 0)


        while not init_flag and self.init_flag_num < 3:
            init_flag = self._apply_knobs(True, 0, 0)
            self.init_flag_num += 1

        external_metrics, internal_metrics = self._get_state(0, 0, logger, True)

        self.last_external_metrics = external_metrics['avg']
        self.default_externam_metrics = external_metrics['avg']

        utils._print(f'initialize done.')
        return internal_metrics, external_metrics['avg']

    def _apply_knobs(self, first, epoch, t):
        """ Apply the knobs to the mysql
        Args:
            knob: dict, mysql parameters
        Returns:
            flag: whether the setup is valid
        """
        self.steps += 1
        if first is True:
            cur_wk = 'init'
        else:
            cur_wk = self.wk_type
        asyncio.get_event_loop().run_until_complete(utils.modify_configurations(
            epoch=epoch,
            rep=t,
            testee=self.testee,
            instance_name=self.instance_name,
            workload_name=cur_wk,
            first=first,
            user=configs.instance_config[self.instance_name]['user'],
            pwd=configs.instance_config[self.instance_name]['password'],
            task_name=self.task_name
        ))

        steps = 0
        max_steps = 5

        flag = True
        time.sleep(3)
        if self.instance_name == 'mongodb':
            flag = utils.test_mongodb(self.instance_name)
            while not flag and steps < max_steps:
                time.sleep(5)
                flag = utils.test_mongodb(self.instance_name)
                steps += 1
        elif self.instance_name == 'hbase':
            flag = utils.test_hbase(self.instance_name)
            while not flag and steps < max_steps:
                time.sleep(5)
                flag = utils.test_hbase(self.instance_name)
                steps += 1

        if not flag:
            return False
        else:
            return True




DockerServer = Server
# -*- coding: utf-8 -*-

"""
description: MySQL Env Utils
"""

import time
import json
import pymongo
import asyncio
from datetime import datetime
import re
from pathlib import Path
from environment import configs
import os
import requests
PROJECT_DIR = configs.PROJECT_DIR


def parse_result(result_dir, task_id, rep):
    result_file = str(result_dir / f'{task_id}_run_result_{rep}')
    with open(result_file) as f:
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

def compute_result(current_result, default_result, last_metric, tps_weight):
    tps_reward = float(current_result[0] / default_result[0]) - 1
    lat_reward = float(default_result[1] / current_result[1]) - 1
    reward = tps_reward * tps_weight + (1 - tps_weight) * lat_reward

    return reward


def _calculate_CDL(delta):
    if delta > 0:
        return 2 ** delta
    else:
        return -1 * (1 / 2) ** delta

def time_start():
    return time.time()


def time_end(start):
    end = time.time()
    delay = end - start
    return delay


async def get_app_metrics(config, value, instance, db_dir):
    if instance == 'mongodb':
        await get_mongo_metrics(config, value, db_dir)
    elif instance == 'cassandra':
        await get_cassandra_metrics(config, value, db_dir)
    # TODO
    elif instance == 'elasticsearch':
        await get_elasticsearch_metrics(config, value, db_dir)
    elif instance == 'hbase':
        await get_hbase_metrics(config, value, db_dir)
    elif instance == 'redis':
        await get_redis_metrics(config, value, db_dir)


def get_default(config):
  res = {}
  for k, conf in config.items():
    res[k] = conf['default']
    if type(res[k]) is bool:
      # make sure no uppercase 'True/False' literal in result
      res[k] = str(res[k]).lower()
  return res

def _print(msg):
  print(f'[{datetime.now()}] {msg}')

def time_start():
    return int(round(time.time() * 1000))

def time_end(start):
    end = int(round(time.time() * 1000))
    delay = end - start
    return delay / 1000

async def modify_configurations(epoch, rep, testee, instance_name, workload_name, first, user, pwd, task_name):
    """ Modify the configurations by restarting the mongodb
    Args:
        epoch: number, number of runs
        rep: number, rep of runs
        testee: str, instance's server IP Addr
        instance_name: str, instance's name
        first: bool, whether it is the first deploy
    """
    db_dir = PROJECT_DIR + '/environment/target/' + instance_name
    deploy_playbook_path = db_dir + '/playbook/deploy.yml'
    osconfig_playbook_path = db_dir + '/playbook/set_os.yml'
    result_dir = Path(db_dir + '/results/' + task_name)

    _print(f'{epoch}: carrying out #{rep} repetition test...')
    try:
        _print(f'{epoch} - {rep}: deploying...')
        stdout, stderr = await run_playbook(
            deploy_playbook_path,
            host=testee,
            task_id=epoch,
            task_name=task_name,
            workload_name=workload_name,
            user=user,
            pwd=pwd,
            project_dir=PROJECT_DIR,
            task_rep=rep,
        )
        if len(stderr) != 0:
            (result_dir / f'{epoch}_deploy_err_{workload_name}_{rep}').write_text(stderr)
        (result_dir / f'{epoch}_deploy_log_{workload_name}_{rep}').write_text(stdout)
        _print(f'{epoch} - {rep}: done.')

        # if not first:
            # os parameters need to be changed
        _print(f'{epoch} - {rep}: setting os parameters...')
        await run_playbook(
                osconfig_playbook_path,
                host=testee,
                task_id=epoch,
                task_name=task_name,
                workload_name=workload_name,
                user=user,
                pwd=pwd,
                project_dir=PROJECT_DIR,
                task_rep=rep,
        )
        # else:
        #     # - no need to change, for default testing or os test is configured to be OFF
        #     _print(
        #         f'{epoch} - {rep}: resetting os parameters...')
        #     await run_playbook(
        #         osconfig_playbook_path,
        #         host=testee,
        #         task_id=epoch,
        #         task_name=task_name,
        #         workload_name=workload_name,
        #         user=user,
        #         pwd=pwd,
        #         project_dir=PROJECT_DIR,
        #         task_rep=rep,
        #         tags='cleanup'
        #     )
        _print(f'{epoch} - {rep}: done.')
    except RuntimeError as e:
        errlog_path = result_dir / f'{epoch}_error_{workload_name}_{rep}.log'
        errlog_path.write_text(str(e))
        print(e)

async def clean_os_knobs(episode, t, db_name, workload_name):
    # - cleanup os config
    _print(f'{episode} - {t}: cleaning up os config...')
    db_dir = PROJECT_DIR + '/environment/target/' + db_name
    osconfig_playbook_path = db_dir + '/playbook/set_os.yml'
    await run_playbook(
        osconfig_playbook_path,
        host=configs.instance_config[db_name]['testee'],
        task_id=episode,
        workload_name=workload_name,
        user=configs.instance_config[db_name]['user'],
        pwd=configs.instance_config[db_name]['password'],
        project_dir=PROJECT_DIR,
        task_rep=t,
        tags='cleanup'
    )
    _print(f'{episode} - {t}: done.')

def test_mongodb(instance_name):
    """ Test the mongodb instance to see whether if it has been restarted
    Args
        instance_name: str, instance's name
    """
    db_config = configs.instance_config[instance_name]
    try:
        myclient = pymongo.MongoClient(host=db_config['host'], port=db_config['port'], serverSelectionTimeoutMS=2000,
                                       socketTimeoutMS=2000)
        myclient.list_database_names()
    except:
        return False
    myclient.close()
    return True

def test_hbase(instance_name):
    db_config = configs.instance_config[instance_name]
    try:
        res = requests.get("http://{}:{}".format(db_config['host'], db_config['port']))
        return True
    except Exception as err:
        return False


def read_machine():
    """ Get the machine information, such as memory and disk

    Return:

    """
    if os.path.exists("/proc/meminfo"):
        f = open("/proc/meminfo", 'r')
        line = f.readlines()[0]
        f.close()
        line = line.strip('\r\n')
        total = int(line.split(':')[1].split()[0])*1024
        return total
    else:
        return 100000

async def run_playbook(playbook_path, tags='all', **extra_vars):
  vars_json_str = json.dumps(extra_vars)
  command = f'ansible-playbook {playbook_path} -vvv --extra-vars=\'{vars_json_str}\' --tags={tags}'
  process = await asyncio.create_subprocess_shell(
      cmd=command,
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE,
      stdin=asyncio.subprocess.PIPE,
  )
  stdout, stderr = await process.communicate()
  if process.returncode != 0:
    raise RuntimeError(
        f'Error running playbook. Below is stdout:\n {stdout.decode("utf-8")}\nand stderr: {stderr.decode("utf-8")}\n')
  return stdout.decode('utf-8'), stderr.decode('utf-8')


def mkdir(path):
    '''
    Create specified folder
    :param path: Folder path, in string format
    :return: True (created successfully) or False (folder already exists, failed to create)
    '''
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False

async def get_os_metrics(config, value, db_dir):
    get_vmstat_playbook_path = db_dir + "/playbook/get_os_stat.yml"

    await run_playbook(
        get_vmstat_playbook_path,
        host=config['testee'],
        user=config['user'],
        pwd=config['password'],
        project_dir=PROJECT_DIR,
    )

    vmstat_path = db_dir + "/vmstat"

    vm_f = open(vmstat_path)
    for i in range(3):
        vm_lines = vm_f.readline()
    vm_row = vm_lines.split()
    vm_array = ['procs_r', 'procs_b', 'memory_swpd', 'memory_free', 'memory_buff',
                'memory_cache', 'swap_si', 'swap_so', 'io_bi', 'io_bo', 'system_in',
                'system_cs', 'cpu_us', 'cpu_sy', 'cpu_id', 'cpu_wa', 'cpu_st']
    for i in range(len(vm_row)):
        value[vm_array[i]] = int(vm_row[i])

    value.pop('memory_swpd')
    value.pop('swap_si')
    value.pop('swap_so')
    # -------------------------
    value.pop('memory_buff')
    value['memory_free'] = value['memory_free'] / 1000000
    value['memory_cache'] = value['memory_cache'] / 1000000
    value['cpu_id'] /= 100
    value.pop('io_bi')
    value.pop('io_bo')
    value.pop('cpu_us')
    value.pop('cpu_sy')
    value.pop('cpu_wa')
    value.pop('cpu_st')



def combine_dict(value, append_value):
    for k in append_value.keys():
        value[k] = append_value[k]

async def get_mongo_metrics(config, value, db_dir):
    myclient = pymongo.MongoClient(host=config['host'], port=config['port'])
    serverStatus = myclient.admin.command("serverStatus")
    serverStatus['mem'].pop('supported')
    serverStatus['mem'].pop('bits')
    serverStatus['mem'].pop('mapped')
    serverStatus['mem'].pop('mappedWithJournal')

    serverStatus['tcmalloc']['tcmalloc'].pop('formattedString')
    serverStatus['tcmalloc']['tcmalloc'].pop('max_total_thread_cache_bytes')
    serverStatus['tcmalloc']['tcmalloc'].pop('aggressive_memory_decommit')
    serverStatus['tcmalloc']['tcmalloc'].pop('pageheap_decommit_count')

    serverStatus['wiredTiger']['connection'].pop('detected system time went backwards')

    mongodb_value = {**serverStatus['mem'], **serverStatus['tcmalloc']['tcmalloc'],
             **serverStatus['wiredTiger']['connection']}
    combine_dict(value, mongodb_value)

    value.pop('files currently open')
    value.pop('thread_cache_free_bytes')
    value.pop('pageheap_total_reserve_bytes')

    value.pop('pthread mutex shared lock read-lock calls')
    value.pop('pthread mutex shared lock write-lock calls')
    value.pop('pthread mutex condition wait calls')
    value.pop('pageheap_committed_bytes')
    value.pop('central_cache_free_bytes')
    value.pop('transfer_cache_free_bytes')
    value.pop('pageheap_free_bytes')
    value.pop('auto adjusting condition resets')
    value.pop('auto adjusting condition wait calls')
    value.pop('memory re-allocations')



async def get_cassandra_metrics(config, value, db_dir):
    get_stat_playbook_path = db_dir + "/playbook/get_app_stat.yml"

    await run_playbook(
        get_stat_playbook_path,
        host=config['testee'],
        user=config['user'],
        pwd=config['password'],
        project_dir=PROJECT_DIR,
    )

    nodetool_tablestats_path = Path(db_dir + "/nodetool_tablestats")
    nodetool_info_path = Path(db_dir + "/nodetool_info")

    read_regexp = re.compile(
        r'Read Latency: (\d+(?:\.\d+)?)', )
    write_regexp = re.compile(
        r'Write Latency: (\d+(?:\.\d+)?)', )
    content = nodetool_tablestats_path.read_text()
    read_match = read_regexp.search(content)
    write_match = write_regexp.search(content)

    value['read_lat'] = float(read_match.group(1)) * 1000
    value['write_lat'] = float(write_match.group(1)) * 1000

    hit_rate_regexp = re.compile(
            r'(\d+(?:\.\d+)?) recent hit rate')

    content = nodetool_info_path.read_text()
    hit_rate_match = hit_rate_regexp.findall(content)

    value['Key_Cache_hit_rate'] = float(hit_rate_match[0])
    value['Chunk_Cache_hit_rate'] = float(hit_rate_match[1])

async def get_elasticsearch_metrics(config, value, db_dir):
    get_stat_playbook_path = db_dir + "/playbook/get_app_stat.yml"


async def get_hbase_metrics(config, value, db_dir):
    OSIndex = requests.get("http://{}:16010/jmx?qry=java.lang:type=OperatingSystem".format(config['host'])).json()['beans'][0]

    value['FreePhysicalMemorySize'] = OSIndex['FreePhysicalMemorySize']
    value['ProcessCpuLoad'] = OSIndex['ProcessCpuLoad']
    value['SystemCpuLoad'] = OSIndex['SystemCpuLoad']
    value['AvailableProcessors'] = OSIndex['AvailableProcessors']

    JVMIndex = requests.get("http://{}:16010/jmx?qry=Hadoop:service=HBase,name=JvmMetrics".format(config['host'])).json()['beans'][
        0]
    value['MemNonHeapUsedM'] = JVMIndex['MemNonHeapUsedM']
    value['MemHeapUsedM'] = JVMIndex['MemHeapUsedM']
    value['GcTimeMillis'] = JVMIndex['GcTimeMillis']

async def get_redis_metrics(config, value, db_dir):
    get_stat_playbook_path = db_dir + "/playbook/get_app_stat.yml"

    await run_playbook(
        get_stat_playbook_path,
        host=config['tester'],
        user=config['user'],
        pwd=config['password'],
        target=config['testee'],
        project_dir=PROJECT_DIR,
    )

    stat_path = Path(db_dir + "/stat")

    fp = open(stat_path, 'r')

    need = ['used_memory_rss', 'used_memory_peak_perc', 'used_memory_dataset_perc',
            'mem_fragmentation_ratio', 'used_cpu_sys', 'used_cpu_user', 'used_cpu_sys_children',
            'used_cpu_user_children']
    i = 0
    lines = fp.readlines()

    for line in lines:
        line = line.strip('\n')
        s = line.split(':')
        if len(s) > 1:
            if s[0] == need[i]:
                i = i + 1
                if i == len(need):
                    break
                value[s[0]] = float(s[1].replace('%', ''))

    value['used_memory_rss'] /= 1000000
    value['used_memory_peak_perc'] /= 100
    value['used_memory_dataset_perc'] /= 100
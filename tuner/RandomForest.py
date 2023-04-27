import sys
from util import utils
import os
import yaml
import pickle
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from environment import appEnv, knobs, configs
from CausalModel.causal_memory import CausalMemory

opt = utils.parse_cmd()
path = './causal_memory/mongodb/train_hesbo_1670665225.pkl'
instance_name = 'mongodb'

env = appEnv.Server(
    wk_type=opt.workload,
    instance_name=opt.instance,
    num_metric=opt.metric_num,
    tps_weight=opt.tps_weight,
    lowDimSpace=opt.LowDimSpace,
    task_name=opt.task_name,
    n_client=opt.n_client,
    cur_knobs_dict=None,
)
knobs_head = env.get_knobs_keys() + ['wk_type']
objects_head = ['tps', 'latency']
# objects_head = ['tps']

cm = CausalMemory()
cm.load_memory(path)
data = cm.data

workload_list = ['workloada', 'workloadb', 'workloadc', 'workloadd', 'workloade']

for i in range(len(data)):
    data.loc[i, 'wk_type'] = np.int64(workload_list.index(data.loc[i, 'wk_type']))

knobs_data = np.array(data[knobs_head].values.tolist())
objects_data = data[objects_head].values.tolist()
objects_data = np.array(list(map(list, zip(*objects_data)))[0])

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
x_train, x_test, y_train, y_test = train_test_split(knobs_data, objects_data, test_size = 0.25, random_state = 0)
feat_labels = data[knobs_head].columns[:]
forest = RandomForestRegressor(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(x_train, y_train)

print("Traing Score:%f" % forest.score(x_train, y_train))
print("Testing Score:%f" % forest.score(x_test, y_test))

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(x_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

threshold = int((x_train.shape[1]-1) * 0.4)
PROJECT_DIR = configs.PROJECT_DIR

db_dir = PROJECT_DIR + '/environment/target/' + instance_name

os_setting_path = db_dir + '/os_configs_info.yml'
app_setting_path = db_dir + '/app_configs_info.yml'
app_setting = yaml.load(open(app_setting_path, 'r', encoding='UTF-8'),
                            Loader=yaml.FullLoader)
os_setting = yaml.load(open(os_setting_path, 'r', encoding='UTF-8'),
                           Loader=yaml.FullLoader)
soft_columns = list(app_setting.keys())
kernel_columns = list(os_setting.keys())
soft_count = 0
kernel_count = 0
cur_soft = []
cur_kernel = []

for f in range(1, threshold):
    if feat_labels[indices[f]] in soft_columns:
        soft_count += 1
        cur_soft.append(feat_labels[indices[f]])
    else:
        kernel_count += 1
        cur_kernel.append(feat_labels[indices[f]])

print("soft_count:", soft_count)
print("kernel_count:", kernel_count)

if not os.path.exists('knobs_choose/' + instance_name):
    os.makedirs('knobs_choose/' + instance_name)

cur = {
    'cur_soft': cur_soft,
    'cur_kernel': cur_kernel,
}
cur_path = 'knobs_choose/' + instance_name + '/' + 'train_randomforest_{}.pkl'.format(str(utils.get_timestamp()))
f = open(cur_path, 'wb')
pickle.dump(cur, f)
f.close()


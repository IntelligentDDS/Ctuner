import os
import time
import sys
from ananke.graphs import ADMG
from networkx import DiGraph
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),"CausalModel"))
from CausalModel.causal_model import CausalModel
from CausalModel.causal_memory import CausalMemory
from environment import knobs


class Causality():
    def __init__(self, conf_opt, perf_columns, obj_columns, knobs_info):
        self.columns = conf_opt + perf_columns + obj_columns  # 所有参数
        self.conf_opt = conf_opt  # 配置参数
        self.perf_columns = perf_columns
        self.obj_columns = obj_columns
        self.knobs_info = knobs_info
        self.NUM_PATHS = 25
        self.query = "best"
        # initialize causal model object
        self.CM = CausalModel(self.columns)
        self.g = DiGraph()
        # add node
        self.g.add_nodes_from(self.columns)
        # edge constraints
        self.tabu_edges = self.CM.get_tabu_edges(self.columns, self.conf_opt, obj_columns)

        self.option_values = self.get_option_values()

        self.var_types = {}
        for col in self.columns:
            self.var_types[col] = "c"
        self.isGenerate = False

    def init_causality(self, cm_path):
        if not self.isGenerate:
            self.causal_memorys = CausalMemory()

            if os.path.exists(cm_path):
                self.causal_memorys.load_memory(cm_path)
                print("--------------------------------------------------------------")
                print("Connections discovered by the causal graph")
                print(self.causal_memorys.di_edges)
                print("--------------------------------------------------------------")

            else:
                self.causal_memorys.init_data(self.columns)
                self.causal_memorys.save(cm_path)

            # add edge
            self.g.add_edges_from(self.causal_memorys.di_edges + self.causal_memorys.bi_edges)

            self.isGenerate = True

    def do_causal(self):
        G, di_edges, bi_edges = self.run_unicorn_loop()
        self.causal_memorys.G = G
        self.causal_memorys.di_edges = di_edges
        self.causal_memorys.bi_edges = bi_edges

    def update_data(self, row):
        self.causal_memorys.update_data(row, self.columns,  self.knob_info)

    def generate_knobs(self, episode, t):
        ref_index = self.causal_memorys.data[[self.obj_columns[0]]].idxmax()
        ref_df = self.causal_memorys.data.loc[ref_index]
        ref = ref_df.iloc[0]

        start = time.time()

        # identify causal paths
        previous_config = ref[self.conf_opt].copy()
        paths = self.CM.get_causal_paths(self.columns, self.causal_memorys.di_edges, self.causal_memorys.bi_edges,
                                        self.obj_columns)

        # compute causal paths
        if len(self.obj_columns) < 2:
            # single objective
            for key, val in paths.items():
                if len(paths[key]) > self.NUM_PATHS:
                    s = self.CM.compute_path_causal_effect(self.causal_memorys.data, paths[key], self.causal_memorys.G,
                                                        self.NUM_PATHS)
                else:
                    paths = paths[self.obj_columns[0]]

            # compute individual treatment effect in a path
            config = self.CM.compute_individual_treatment_effect(self.causal_memorys.data, paths,
                                                                self.query,  self.obj_columns, ref[self.obj_columns[0]],
                                                                previous_config, self.conf_opt, self.var_types,  self.option_values)

        else:
            # multi objective
            paths = paths[self.obj_columns[0]]
            # compute individual treatment effect in a path
            config = self.CM.compute_individual_treatment_effect(self.causal_memorys.data, paths,
                                                                self.query, self.obj_columns, ref[self.obj_columns],
                                                                previous_config, self.conf_opt, self.var_types,  self.option_values)
        end = time.time() - start
        print("Time", end)

        # perform intervention. This updates the init_data
        if config is not None:
            return self.value_to_action(config), self.value_to_system(config, episode, t),
        else:
            return [], {}


    def value_to_action(self, config):
        action = []
        for i, value in enumerate(self.knobs_info.items()):
            key, v = value
            if v.get('range'):  # discrete ranged parameter
                enum_size = len(v['range'])
                action.append((config[i]+0.5) / enum_size)
            else:
                max_val = v['max']
                min_val = v['min']
                if v.get('bucket_num'):
                    action.append(((config[i] - min_val) * v['bucket_num'] / (max_val - min_val) + 0.5) / v['bucket_num'])
                else:
                    action.append((config[i] - min_val) / (max_val - min_val))
        return action

    def value_to_system(self, config, episode, t):
        app_setting, os_setting = knobs.get_os_app_setting()
        sampled_app_config = {}
        sampled_os_config = {}
        for i,v in enumerate(self.knobs_info.items()):
            k, value = v
            if value.get('range'):  # discrete ranged parameter
                config[i] = value['range'][config[i]]

            if k in app_setting:
                if type(config[i]) is bool:
                    # make sure no uppercase 'True/False' literal in result
                    sampled_app_config[k] = str(config[i]).lower()
                elif type(config[i]) is np.float64:
                    sampled_app_config[k] = float(config[i])
                else:
                    sampled_app_config[k] = config[i]
            else:
                if type(config[i]) is bool:
                    # make sure no uppercase 'True/False' literal in result
                    sampled_os_config[k] = str(config[i]).lower()
                elif type(config[i]) is np.float64:
                    sampled_os_config[k] = float(config[i])
                else:
                    sampled_os_config[k] = config[i]

        knobs.dump_configs(sampled_app_config, sampled_os_config, episode, t)

        return {**(sampled_app_config), **(sampled_os_config)}


    def run_unicorn_loop(self):
        """This function is used to run unicorn in a loop"""

        fci_edges = self.CM.learn_fci(self.causal_memorys.data, self.tabu_edges, -1)
        edges = []
        # resolve notears_edges and fci_edges and update
        di_edges, bi_edges = self.CM.resolve_edges(edges, fci_edges, self.columns,
                                              self.tabu_edges, self.NUM_PATHS, self.obj_columns)
        # construct mixed graph ADMG
        G = ADMG(self.columns, di_edges=di_edges, bi_edges=bi_edges)
        return G, di_edges, bi_edges

    def get_option_values(self):
        option_values_num = 10
        option_values = {}
        for name, value in self.knobs_info.items():
            ll = []
            if value.get('range'):  # discrete ranged parameter
                for item in value['range']:
                    ll.append(item)
                option_values[name] = ll
            else:
                if value.get('float'):
                    max_val = value['max']
                    min_val = value['min']
                    item_rep = (max_val - min_val) / option_values_num
                    while min_val <= max_val:
                        ll.append(min_val)
                        min_val += item_rep
                    option_values[name] = ll
                else:
                    max_val = value['max']
                    min_val = value['min']
                    item_rep = int((max_val - min_val) / option_values_num)
                    if item_rep == 0:
                        item_rep = 1
                    while min_val <= max_val:
                        ll.append(min_val)
                        min_val += item_rep
                    option_values[name] = ll

        return option_values





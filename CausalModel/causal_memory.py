import pickle
import pandas as pd
import numpy as np

class CausalMemory(object):

    def __init__(self):
        self.G = None
        self.di_edges = None
        self.bi_edges = None
        self.e_greed = 0.3
        self.data = None
        self.causalData = None
        self.rc = []

    def add_rc(self, data):
        self.rc.append(data)

    def init_data(self, columns):
        # print("colunms_len:",len(columns))
        # print("columns:",columns)
        self.data = pd.DataFrame([], columns=columns)
        for j in columns:
            if self.data[j].dtype == 'object':
                self.data[j] = self.data[j].astype('int64')

    def update_data(self, row, columns, knob_info):
        # print("row_len:", len(row))
        # print("row:",row)
        self.data.loc[len(self.data)] = row
        index = len(self.data) - 1
        for j in columns:
            if type(self.data.loc[index, j]) is str or type(self.data.loc[index, j]) is np.bool_ or type(self.data.loc[index, j]) is bool:
                if type(knob_info[j]['range'][0]) is bool:
                    self.data.loc[index, j] = bool(self.data.loc[index, j])
                self.data.loc[index, j] = np.int64(knob_info[j]['range'].index(self.data.loc[index, j]))

    def get_causalData(self, columns):
        self.causalData = self.data[columns]

    def update_cg(self, G, di_edges, bi_edges):
        self.G = G
        self.di_edges = di_edges
        self.bi_edges = bi_edges

    def update_e(self, e_greed):
        self.e_greed = e_greed

    def __len__(self):
        return len(self.data)

    def save(self, path):
        cm = {
            'G': self.G,
            'di_edges': self.di_edges,
            'bi_edges': self.bi_edges,
            'data': self.data,
            'e_greed': self.e_greed,
            'rc': self.rc,
        }
        f = open(path, 'wb')
        pickle.dump(cm, f)
        f.close()

    def load_memory(self, path):
        with open(path, 'rb') as f:
            cm = pickle.load(f)
        self.G = cm['G']
        self.di_edges = cm['di_edges']
        self.bi_edges = cm['bi_edges']
        self.data = cm['data']
        self.e_greed = cm['e_greed']
        self.rc = cm['rc']
        # self.e_greed = 0.5


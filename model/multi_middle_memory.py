import pickle

class MultiMiddleMemory(object):

    def __init__(self, tasks):
        self.task_mms = dict([(idx, list()) for idx in tasks])

    def size_all(self):
        tasks_len = dict()
        for key in self.task_mms.keys():
            tasks_len[key] = len(self.task_mms[key])
        return tasks_len

    def size_rb(self,task):
        return len(self.task_mms[task])

    def get_mm(self, task):
        return self.task_mms[task]

    def add(self, task, data):
        self.task_mms[task].append(data)

    def save(self, path):
        data = {
            'task_mms': self.task_mms,
        }
        f = open(path, 'wb')
        pickle.dump(data, f)
        f.close()

    def load_memory(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.task_mms = data['task_mms']
from model.prioritized_replay_memory import PrioritizedReplayMemory
import pickle
import numpy as np
import random

class MultiTaskPrioritizedReplayBuffer(object):
    def __init__(
            self,
            max_size
    ):
        """
        :param max_replay_buffer_size:
        """
        self.max_replay_buffer_size = max_size

    def init(self, tasks=None):
        '''
            init buffers for all tasks
            :param tasks: for multi-task setting
        '''
        self.task_buffers = dict([(idx, PrioritizedReplayMemory(
            capacity=self.max_replay_buffer_size,
        )) for idx in tasks])

    def __len__(self):
        tasks_len = dict()
        for key in self.task_buffers.keys():
            tasks_len[key] = len(self.task_buffers[key])
        return tasks_len

    def size_rb(self,task):
        return len(self.task_buffers[task])

    def add_sample(self, task, error, sample):
        # sample包括state, action, reward, next_state, done, external_metrics
        self.task_buffers[task].add(error, sample)

    def get_buffer(self, task_id):
        return self.task_buffers[task_id]

    def format_sample(self, batch):
        state = list(map(lambda x: x[0], batch))
        next_state = list(map(lambda x: x[1], batch))
        action = list(map(lambda x: x[2], batch))
        reward = list(map(lambda x: x[3], batch))
        done = list(map(lambda x: x[4], batch))
        previous_action = list(map(lambda x: x[5], batch))
        previous_reward = list(map(lambda x: x[6], batch))
        previous_state = list(map(lambda x: x[7], batch))
        next_actions = list(map(lambda x: x[8], batch))
        next_rewards = list(map(lambda x: x[9], batch))
        next_states = list(map(lambda x: x[10], batch))
        return  np.array(state), np.array(next_state), np.array(action), \
                np.array(reward).reshape(-1, 1), np.array(done).reshape(-1, 1), \
                np.array(previous_action), np.array(previous_reward), np.array(previous_state), \
                np.array(next_actions), np.array(next_rewards), np.array(next_states)

    def getAll(self, tasks):
        all_task = dict()
        for task in tasks:
            batch, idx = self.task_buffers[task].sample(len(self.task_buffers[task]))
            all_task[task] = batch
        return all_task

    def sample(self, task_ids, batch_size):
        # X ==> state,
        # Y ==> next_state
        # U ==> action
        # r ==> reward
        # d ==> done
        # pu ==> previous action
        # pr ==> previous reward
        # px ==> previous state
        # nu ==> next actions
        # nr ==> next rewards
        # nx ==> next states
        if len(task_ids) == 1:
            batch, idx = self.task_buffers[task_ids[0]].sample(batch_size)
            xx, _, _, _, _, pu, pr, px, _, _, _ = self.format_sample(batch)
            return pu, pr, px, xx

        mb_actions = []
        mb_rewards = []
        mb_obs = []
        mb_x = []

        for tid in task_ids:
            batch, idx = self.task_buffers[tid].sample(batch_size)
            xx, _, _, _, _, pu, pr, px, _, _, _ = self.format_sample(batch)
            mb_actions.append(pu)  # batch_size x D1
            mb_rewards.append(pr)  # batch_size x D2
            mb_obs.append(px)  # batch_size x D3
            mb_x.append(xx)

        mb_actions = np.asarray(mb_actions, dtype=np.float32)  # task_ids x batch_size x D1
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)  # task_ids x batch_size x D2
        mb_obs = np.asarray(mb_obs, dtype=np.float32)  # task_ids x batch_size x D2
        mb_x = np.asarray(mb_x, dtype=np.float32)

        return mb_actions, mb_rewards, mb_obs, mb_x

    def sample_tasks(self, task_ids, batch_size):
        '''
            Returns tuples of (state, next_state, action, reward, done,
                              previous_action, previous_reward, previous_state
                              )
        '''
        mb_xx = []
        mb_yy = []
        mb_u = []
        mb_r = []
        mb_d = []
        mb_pu = []
        mb_pr = []
        mb_px = []
        mb_nu = []
        mb_nr = []
        mb_nx = []

        # shuffle task lists
        shuffled_task_ids = random.sample(task_ids, len(task_ids))

        for tid in shuffled_task_ids:
            batch, idx = self.task_buffers[tid].sample(batch_size)
            xx, yy, u, r, d, pu, pr, px, nu, nr, nx = self.format_sample(batch)
            mb_xx.append(xx)  # batch_size x D1
            mb_yy.append(yy)  # batch_size x D2
            mb_u.append(u)  # batch_size x D3
            mb_r.append(r)
            mb_d.append(d)
            mb_pu.append(pu)
            mb_pr.append(pr)
            mb_px.append(px)
            mb_nu.append(nu)
            mb_nr.append(nr)
            mb_nx.append(nx)

        mb_xx = np.asarray(mb_xx, dtype=np.float32).reshape(len(task_ids) * batch_size,
                                                            -1)  # task_ids x batch_size x D1
        mb_yy = np.asarray(mb_yy, dtype=np.float32).reshape(len(task_ids) * batch_size,
                                                            -1)  # task_ids x batch_size x D1
        mb_u = np.asarray(mb_u, dtype=np.float32).reshape(len(task_ids) * batch_size, -1)  # task_ids x batch_size x D1
        mb_r = np.asarray(mb_r, dtype=np.float32).reshape(len(task_ids) * batch_size, -1)  # task_ids x batch_size x D1
        mb_d = np.asarray(mb_d, dtype=np.float32).reshape(len(task_ids) * batch_size, -1)  # task_ids x batch_size x D1
        mb_pu = np.asarray(mb_pu, dtype=np.float32).reshape(len(task_ids) * batch_size,
                                                            -1)  # task_ids x batch_size x D1
        mb_pr = np.asarray(mb_pr, dtype=np.float32).reshape(len(task_ids) * batch_size,
                                                            -1)  # task_ids x batch_size x D1
        mb_px = np.asarray(mb_px, dtype=np.float32).reshape(len(task_ids) * batch_size,
                                                            -1)  # task_ids x batch_size x D1
        mb_nu = np.asarray(mb_nu, dtype=np.float32).reshape(len(task_ids) * batch_size,
                                                            -1)  # task_ids x batch_size x D1
        mb_nr = np.asarray(mb_nr, dtype=np.float32).reshape(len(task_ids) * batch_size,
                                                            -1)  # task_ids x batch_size x D1
        mb_nx = np.asarray(mb_nx, dtype=np.float32).reshape(len(task_ids) * batch_size,
                                                            -1)  # task_ids x batch_size x D1

        return mb_xx, mb_yy, mb_u, mb_r, mb_d, mb_pu, mb_pr, mb_px, mb_nu, mb_nr, mb_nx

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def add_samples(self, task, paths):
        for error, sample in paths:
            self.task_buffers[task].add(error, sample)

    def save(self, path):
        f = open(path, 'wb')
        tasks_tree = dict()
        for key in self.task_buffers.keys():
            tasks_tree[key] = self.task_buffers[key].get_tree()
        pickle.dump(tasks_tree, f)
        f.close()

    def load_memory(self, path):
        with open(path, 'rb') as f:
            tasks_tree = pickle.load(f)
        for key in self.task_buffers.keys():
            self.task_buffers[key].set_tree(tasks_tree[key])
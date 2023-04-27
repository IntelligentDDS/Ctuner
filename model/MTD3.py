import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from copy import deepcopy
from sklearn.linear_model import LogisticRegression as logistic
from model.prioritized_replay_memory import PrioritizedReplayMemory
from model.multi_prioritized_replay_memory import MultiTaskPrioritizedReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class Context(nn.Module):
    """
      This layer just does non-linear transformation(s)
    """

    def __init__(self,
                 hidden_sizes=[50],
                 input_dim=None,
                 hidden_activation=F.relu,
                 action_dim=None,
                 obsr_dim=None,
                 device='cpu'
                 ):

        super(Context, self).__init__()
        self.hid_act = hidden_activation
        self.fcs = []  # list of linear layer
        self.hidden_sizes = hidden_sizes
        self.input_dim = input_dim
        self.device = device
        self.action_dim = action_dim
        self.obsr_dim = obsr_dim

        #### build LSTM or multi-layers FF
        # use LSTM or GRU
        self.recurrent = nn.GRU(self.input_dim,
                                self.hidden_sizes[0],
                                bidirectional=False,
                                batch_first=True,
                                num_layers=1)

    def init_recurrent(self, bsize=None):
        '''
            init hidden states
            Batch size can't be none
        '''
        # The order is (num_layers, minibatch_size, hidden_dim)
        # LSTM ==> return (torch.zeros(1, bsize, self.hidden_sizes[0]),
        #        torch.zeros(1, bsize, self.hidden_sizes[0]))
        return torch.zeros(1, bsize, self.hidden_sizes[0]).to(self.device)

    def forward(self, data):
        '''
            pre_x : B * D where B is batch size and D is input_dim
            pre_a : B * A where B is batch size and A is input_dim
            previous_reward: B * 1 where B is batch size and 1 is input_dim
        '''
        previous_action, previous_reward, pre_x = data[0], data[1], data[2]

        # first prepare data for LSTM
        bsize, dim = previous_action.shape  # previous_action is B* (history_len * D)
        pacts = previous_action.view(bsize, -1, self.action_dim)  # view(bsize, self.hist_length, -1)
        prews = previous_reward.view(bsize, -1, 1)  # reward dim is 1, view(bsize, self.hist_length, 1)
        pxs = pre_x.view(bsize, -1, self.obsr_dim)  # view(bsize, self.hist_length, -1)
        pre_act_rew = torch.cat([pacts, prews, pxs], dim=-1)  # input to LSTM is [action, reward]

        # init lstm/gru
        hidden = self.init_recurrent(bsize=bsize)

        # lstm/gru
        _, hidden = self.recurrent(pre_act_rew, hidden)  # hidden is (1, B, hidden_size)
        out = hidden.squeeze(0)  # (1, B, hidden_size) ==> (B, hidden_size)

        return out

class Actor(nn.Module):
    def __init__(self,
                 n_states,
                 n_actions,
                 hiddens_dim_conext = [30],
                 hidden_sizes=[128, 128, 64]):

        super(Actor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_states + hiddens_dim_conext[0], hidden_sizes[0]),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(hidden_sizes[2]),
            nn.Linear(hidden_sizes[2], n_actions)
        )
        self.act = nn.Sigmoid()
        self._init_weights()

        # context network
        self.context = Context(hidden_sizes=hiddens_dim_conext,
                                input_dim=n_states + n_actions + 1,
                                action_dim=n_actions,
                                obsr_dim=n_states,
                                device=device
                                )

    def _init_weights(self):
        for m in self.layers:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 1e-2)
                m.bias.data.uniform_(-0.1, 0.1)

    def forward(self, states, pre_act_rew = None, ret_context = False):  # pylint: disable=arguments-differ
        combined = self.context(pre_act_rew)
        states = torch.cat([states, combined], dim=-1)
        actions = self.act(self.layers(states))
        if ret_context == True:
            return actions, combined
        else:
            return actions

    def get_conext_feats(self, pre_act_rew):
        '''
            pre_act_rew: B * (A + 1) where B is batch size and A + 1 is input_dim
            return combine features
        '''
        combined = self.context(pre_act_rew)

        return combined

class Critic(nn.Module):
    def __init__(self,
                 n_states,
                 n_actions,
                 hiddens_dim_conext=[30],
                 hidden_sizes=[128, 128, 64]):
        super(Critic, self).__init__()
        # Q1 architecture
        self.layers_1 = nn.Sequential(
            nn.Linear(n_states + n_actions + hiddens_dim_conext[0], hidden_sizes[0] * 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_sizes[0] * 2, hidden_sizes[1]),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.3),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(hidden_sizes[2]),
            nn.Linear(hidden_sizes[2], 1)
        )

        self.layers_2 = nn.Sequential(
            nn.Linear(n_states + n_actions + hiddens_dim_conext[0], hidden_sizes[0] * 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_sizes[0] * 2, hidden_sizes[1]),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.3),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(hidden_sizes[2]),
            nn.Linear(hidden_sizes[2], 1)
        )

        self._init_weights()

        # context network
        self.context = Context(hidden_sizes=hiddens_dim_conext,
                               input_dim=n_states + n_actions + 1,
                               action_dim=n_actions,
                               obsr_dim=n_states,
                               device=device
                               )

    def _init_weights(self):
        for m in self.layers_1:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 1e-2)
                m.bias.data.uniform_(-0.1, 0.1)

        for m in self.layers_2:
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 1e-2)
                m.bias.data.uniform_(-0.1, 0.1)

    def forward(self, states, actions, pre_act_rew = None, ret_context = False):
        xu = torch.cat([states, actions], 1)
        combined = self.context(pre_act_rew)
        xu = torch.cat([xu, combined], dim=-1)
        q1 = self.layers_1(xu)

        q2 = self.layers_2(xu)

        if ret_context == True:
            return q1, q2, combined

        else:
            return q1, q2

    def Q1(self, states, actions, pre_act_rew = None, ret_context = False):
        xu = torch.cat([states, actions], 1)
        combined = self.context(pre_act_rew)
        xu = torch.cat([xu, combined], dim=-1)

        q1 = self.layers_1(xu)
        if ret_context == True:
            return q1, combined

        else:
            return q1

    def get_conext_feats(self, pre_act_rew):
        '''
            pre_act_rew: B * (A + 1) where B is batch size and A + 1 is input_dim
            return combine features
        '''
        combined = self.context(pre_act_rew)

        return combined

class meta_TD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            device,
            buffer_size,
            batch_size=64,
            action_noise=0.1,
            discount=0.2,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            prox_coef=1,
            use_normalized_beta=True,
            beta_clip=1,
            use_ess_clipping=True
    ):

        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.batch_size = batch_size
        self.device = device
        self.action_noise = action_noise
        self.replay_memory = PrioritizedReplayMemory(capacity=buffer_size)
        self.tasks_buffer = MultiTaskPrioritizedReplayBuffer(max_size=buffer_size)
        self.eval_task_buffer = MultiTaskPrioritizedReplayBuffer(max_size=buffer_size)
        self.prox_coef = prox_coef
        self.prox_coef_init = prox_coef
        self.use_normalized_beta = use_normalized_beta
        self.beta_clip = beta_clip
        self.r_eps = np.float32(1e-7)  # this is used to avoid inf or nan in calculations
        self.use_ess_clipping = use_ess_clipping

        self.copy_model_params()
        self.total_it = 0

    def init_tasks_buffer(self, tasks):
        self.tasks_buffer.init(tasks)

    def init_eval_tasks_buffer(self, tasks):
        self.eval_task_buffer.init(tasks)

    @staticmethod
    def totensor(x):
        return Variable(torch.FloatTensor(x))

    def get_score(self, state, action, np_pre_actions, np_pre_rewards, np_pre_obsers):
        batch_state = self.totensor([state.tolist()])
        previous_action = self.totensor([np_pre_actions.tolist()])
        previous_reward = self.totensor([np_pre_rewards.tolist()])
        previous_obs = self.totensor([np_pre_obsers.tolist()])
        pre_act_rew = [previous_action, previous_reward, previous_obs]

        current_Q1, current_Q2 = self.critic(batch_state, self.totensor([action.tolist()]), pre_act_rew)
        return torch.min(current_Q1, current_Q2)

    def add_sample(self, cur_task, state, next_state, action, reward, done,
                   np_pre_actions, np_pre_rewards, np_pre_obsers,
                   np_next_hacts, np_next_hrews, np_next_hobvs,
                   external_metrics, is_train=True):
        # print("state:", type(state), state)
        # print("action:", type(action), action)
        # print("reward:", type(reward), reward)
        # print("next_state:", type(next_state),next_state)
        # print("done:", type(done), done)
        # print("external_metrics:", type(external_metrics), external_metrics)
        threshold = 0.4
        self.critic.eval()
        self.actor.eval()
        self.critic_target.eval()
        self.actor_target.eval()
        batch_state = self.totensor([state.tolist()])
        batch_next_state = self.totensor([next_state.tolist()])

        previous_action = self.totensor([np_pre_actions.tolist()])
        previous_reward = self.totensor([np_pre_rewards.tolist()])
        previous_obs = self.totensor([np_pre_obsers.tolist()])
        hist_actions = self.totensor([np_next_hacts.tolist()])
        hist_rewards = self.totensor([np_next_hrews.tolist()])
        hist_obs = self.totensor([np_next_hobvs.tolist()])

        act_rew = [hist_actions, hist_rewards, hist_obs]
        pre_act_rew = [previous_action, previous_reward, previous_obs]

        current_Q1, current_Q2 = self.critic(batch_state, self.totensor([action.tolist()]), pre_act_rew)
        current_value = torch.min(current_Q1, current_Q2)

        noise = (torch.randn_like(torch.tensor(action, dtype=torch.float)) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        target_action = (self.actor_target(batch_next_state, act_rew) + noise).clamp(0, 1)

        target_Q1, target_Q2 = self.critic_target(batch_next_state, target_action, act_rew)
        target_value = torch.min(target_Q1, target_Q2)
        target_value = reward + (not done) * self.discount * target_value

        error = float(torch.abs(current_value - target_value).data.numpy()[0]) + threshold * math.tanh(reward) + threshold

        self.actor_target.train()
        self.actor.train()
        self.critic.train()
        self.critic_target.train()
        if is_train:
            self.replay_memory.add(error, (state, next_state, action, reward, done,
                       np_pre_actions, np_pre_rewards, np_pre_obsers,
                       np_next_hacts, np_next_hrews, np_next_hobvs, external_metrics))
            self.tasks_buffer.add_sample(cur_task, error, (state, next_state, action, reward, done,
                                           np_pre_actions, np_pre_rewards, np_pre_obsers,
                                           np_next_hacts, np_next_hrews, np_next_hobvs, external_metrics))
        else:
            self.eval_task_buffer.add_sample(cur_task, error, (state, next_state, action, reward, done,
                                           np_pre_actions, np_pre_rewards, np_pre_obsers,
                                           np_next_hacts, np_next_hrews, np_next_hobvs, external_metrics))

        return current_value

    def choose_action(self, state, previous_action, previous_reward, previous_obs, train=True):
        self.actor.eval()
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)

        previous_action = torch.FloatTensor(previous_action.reshape(1, -1)).to(self.device)
        previous_reward = torch.FloatTensor(previous_reward.reshape(1, -1)).to(self.device)
        previous_obs = torch.FloatTensor(previous_obs.reshape(1, -1)).to(self.device)
        pre_act_rew = [previous_action, previous_reward, previous_obs]

        action = self.actor(state, pre_act_rew)
        if train:
            # exploration noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            action = torch.clamp(action + noise, 0, 1)

        self.actor.train()
        return action.squeeze().detach().cpu().numpy()

    def update(self):
        self.total_it += 1

        # Sample replay buffer
        batch, idx = self.replay_memory.sample(self.batch_size)
        state = torch.tensor(list(map(lambda x: x[0], batch)), dtype=torch.float).to(self.device)
        next_state = torch.tensor(list(map(lambda x: x[1], batch)), dtype=torch.float).to(self.device)
        action = torch.tensor(list(map(lambda x: x[2], batch)), dtype=torch.float).to(self.device)
        reward = torch.tensor(list(map(lambda x: x[3], batch)), dtype=torch.float).view(-1, 1).to(self.device)
        done = torch.tensor(list(map(lambda x: x[4], batch)), dtype=torch.float).view(-1, 1).to(self.device)
        previous_action = torch.tensor(list(map(lambda x: x[5], batch)), dtype=torch.float).to(self.device)
        previous_reward = torch.tensor(list(map(lambda x: x[6], batch)), dtype=torch.float).to(self.device)
        previous_obs = torch.tensor(list(map(lambda x: x[7], batch)), dtype=torch.float).to(self.device)
        hist_actions = torch.tensor(list(map(lambda x: x[8], batch)), dtype=torch.float).to(self.device)
        hist_rewards = torch.tensor(list(map(lambda x: x[9], batch)), dtype=torch.float).to(self.device)
        hist_obs = torch.tensor(list(map(lambda x: x[10], batch)), dtype=torch.float).to(self.device)

        act_rew = [hist_actions, hist_rewards, hist_obs]  # torch.cat([action, reward], dim = -1)
        pre_act_rew = [previous_action, previous_reward, previous_obs]

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state, act_rew) + noise).clamp(0, 1)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action, act_rew)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * (-1)).add(1) * (-1) * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action, pre_act_rew)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state, pre_act_rew), pre_act_rew).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.copy_model_params()

            return critic_loss.data, actor_loss.data

        self.copy_model_params()

        return critic_loss.data, None

    def train_cs(self, train_tasks_list=None, task_id=None, enable_beta_obs_cxt=True, max_iter_logistic=2000, lam_csc=0.50):
        '''
            This function trains covariate shift correction model
        '''
        ######
        # fetch all_data
        ######
        # step 1: calculate how many samples per classes we need
        # in adaption step, all train task can be used
        task_bsize = int(self.eval_task_buffer.size_rb(task_id) / (len(train_tasks_list))) + 2
        neg_tasks_ids = train_tasks_list

        # collect examples from other tasks and consider them as one class
        # view --> len(neg_tasks_ids),task_bsize, D ==> len(neg_tasks_ids) * task_bsize, D
        pu, pr, px, xx = self.tasks_buffer.sample(task_ids=neg_tasks_ids, batch_size=task_bsize)
        neg_actions = torch.FloatTensor(pu).view(task_bsize * len(neg_tasks_ids), -1).to(self.device)
        neg_rewards = torch.FloatTensor(pr).view(task_bsize * len(neg_tasks_ids), -1).to(self.device)
        neg_obs = torch.FloatTensor(px).view(task_bsize * len(neg_tasks_ids), -1).to(self.device)
        neg_xx = torch.FloatTensor(xx).view(task_bsize * len(neg_tasks_ids), -1).to(self.device)

        # sample cuurent task and consider it as another class
        # returns size: (task_bsize, D)
        ppu, ppr, ppx, pxx = self.eval_task_buffer.sample(task_ids=[task_id], batch_size=self.eval_task_buffer.size_rb(task_id))
        pos_actions = torch.FloatTensor(ppu).to(self.device)
        pos_rewards = torch.FloatTensor(ppr).to(self.device)
        pos_obs = torch.FloatTensor(ppx).to(self.device)
        pos_pxx = torch.FloatTensor(pxx).to(self.device)

        # combine reward and action and previous states for context network.
        pos_act_rew_obs = [pos_actions, pos_rewards, pos_obs]
        neg_act_rew_obs = [neg_actions, neg_rewards, neg_obs]

        ######
        # extract features: context features
        ######
        with torch.no_grad():
            # batch_size X context_hidden
            # self.actor.get_conext_feats outputs, [batch_size , context_size]
            # torch.cat ([batch_size , obs_dim], [batch_size , context_size]) ==> [batch_size, obs_dim + context_size ]
            if enable_beta_obs_cxt == True:
                snap_ctxt = torch.cat([pos_pxx, self.actor.get_conext_feats(pos_act_rew_obs)],
                                      dim=-1).cpu().data.numpy()
                neg_ctxt = torch.cat([neg_xx, self.actor.get_conext_feats(neg_act_rew_obs)], dim=-1).cpu().data.numpy()
            else:
                snap_ctxt = self.actor.get_conext_feats(pos_act_rew_obs).cpu().data.numpy()
                neg_ctxt = self.actor.get_conext_feats(neg_act_rew_obs).cpu().data.numpy()

        ######
        # Train logistic classifiers
        ######
        x = np.concatenate((snap_ctxt, neg_ctxt))  # [b1 + b2] X D
        y = np.concatenate((-np.ones(snap_ctxt.shape[0]), np.ones(neg_ctxt.shape[0])))

        # model params : [1 , D] wehere D is context_hidden
        model = logistic(solver='lbfgs', max_iter=max_iter_logistic, C=lam_csc).fit(x, y)
        # keep track of how good is the classifier
        predcition_score = model.score(x, y)

        info = (snap_ctxt.shape[0], neg_ctxt.shape[0], model.score(x, y))
        # print(info)
        return model, info

    def get_prox_penalty(self, model_t, model_target):
        '''
            This function calculates ||theta - theta_t||
        '''
        param_prox = []
        for p, q in zip(model_t.parameters(), model_target.parameters()):
            # q should ne detached
            param_prox.append((p - q.detach()).norm()**2)

        result = sum(param_prox)

        return result

    def copy_model_params(self):
        '''
            Keep a copy of actor and critic for proximal update
        '''
        self.ckpt = {
            'actor': deepcopy(self.actor),
            'critic': deepcopy(self.critic)
        }

    def update_prox_w_ess_factor(self, cs_model, x, beta=None):
        '''
            This function calculates effective sample size (ESS):
            ESS = ||w||^2_1 / ||w||^2_2  , w = pi / beta
            ESS = ESS / n where n is number of samples to normalize
            x: is (n, D)
        '''
        n = x.shape[0]
        if beta is not None:
            # beta results should be same as using cs_model.predict_proba(x)[:,0] if no clipping
            w = ((torch.sum(beta)**2) /(torch.sum(beta**2) + self.r_eps) )/n
            ess_factor = np.float32(w.numpy())

        else:
            # step 1: get prob class 1
            p0 = cs_model.predict_proba(x)[:,0]
            w =  p0 / ( 1 - p0 + self.r_eps)
            w = (np.sum(w)**2) / (np.sum(w**2) + self.r_eps)
            ess_factor = np.float32(w) / n

        # since we assume task_i is class -1, and replay buffer is 1, then
        ess_prox_factor = 1.0 - ess_factor

        if np.isnan(ess_prox_factor) or np.isinf(ess_prox_factor) or ess_prox_factor <= self.r_eps: # make sure that it is valid
            self.prox_coef = self.prox_coef_init

        else:
            self.prox_coef = ess_prox_factor

    def get_propensity(self, cs_model, curr_pre_act_rew, curr_obs, enable_beta_obs_cxt=True):
        '''
            This function returns propensity for current sample of data
            simply: exp(f(x))
        '''

        ######
        # extract features: context features
        ######
        with torch.no_grad():
            # batch_size X context_hidden
            if enable_beta_obs_cxt == True:
                ctxt = torch.cat([curr_obs, self.actor.get_conext_feats(curr_pre_act_rew)], dim = -1).cpu().data.numpy()
            else:
                ctxt = self.actor.get_conext_feats(curr_pre_act_rew).cpu().data.numpy()

        # step 0: get f(x)
        f_prop = np.dot(ctxt, cs_model.coef_.T) + cs_model.intercept_

        # step 1: convert to torch
        f_prop = torch.from_numpy(f_prop).float()

        # To make it more stable, clip it
        f_prop = f_prop.clamp(min=-self.beta_clip)

        # step 2: exp(-f(X)), f_score: N * 1
        f_score = torch.exp(-f_prop)
        f_score[f_score < 0.1]  = 0 # for numerical stability

        if self.use_normalized_beta == True:
            #get logistic regression prediction of class [-1] for current task
            lr_prob = cs_model.predict_proba(ctxt)[:,0]
            # normalize using logistic_probs
            d_pmax_pmin = np.float32(np.max(lr_prob) - np.min(lr_prob))
            f_score = ( d_pmax_pmin * (f_score - torch.min(f_score)) )/( torch.max(f_score) - torch.min(f_score) + self.r_eps ) + np.float32(np.min(lr_prob))

        # update prox coeff with ess.
        if self.use_ess_clipping == True:
            self.update_prox_w_ess_factor(cs_model, ctxt, beta=f_score)

        return f_score, None

    def do_training(self, train_replay_buffer, iterations=None, csc_model=None, apply_prox=False, current_batch_size=None):
        '''
            inputs:
                replay_buffer
                iterations episode_timesteps
        '''

        actor_loss_out = 0.0
        critic_loss_out = 0.0
        critic_prox_out = 0.0
        actor_prox_out = 0.0
        list_prox_coefs = [self.prox_coef]

        for it in range(iterations):
            ########
            # Sample replay buffer
            ########
            batch, idx = train_replay_buffer.sample(current_batch_size)
            obs = torch.tensor(list(map(lambda x: x[0], batch)), dtype=torch.float).to(self.device)
            next_obs = torch.tensor(list(map(lambda x: x[1], batch)), dtype=torch.float).to(self.device)
            action = torch.tensor(list(map(lambda x: x[2], batch)), dtype=torch.float).to(self.device)
            reward = torch.tensor(list(map(lambda x: x[3], batch)), dtype=torch.float).view(-1, 1).to(self.device)
            done = torch.tensor(list(map(lambda x: x[4], batch)), dtype=torch.float).view(-1, 1).to(self.device)
            previous_action = torch.tensor(list(map(lambda x: x[5], batch)), dtype=torch.float).to(self.device)
            previous_reward = torch.tensor(list(map(lambda x: x[6], batch)), dtype=torch.float).to(self.device)
            previous_obs = torch.tensor(list(map(lambda x: x[7], batch)), dtype=torch.float).to(self.device)
            hist_actions = torch.tensor(list(map(lambda x: x[8], batch)), dtype=torch.float).to(self.device)
            hist_rewards = torch.tensor(list(map(lambda x: x[9], batch)), dtype=torch.float).to(self.device)
            hist_obs = torch.tensor(list(map(lambda x: x[10], batch)), dtype=torch.float).to(self.device)

            # combine reward and action
            act_rew = [hist_actions, hist_rewards, hist_obs]  # torch.cat([action, reward], dim = -1)
            pre_act_rew = [previous_action, previous_reward,
                           previous_obs]  # torch.cat([previous_action, previous_reward], dim = -1)

            if csc_model is None:
                # propensity_scores dim is batch_size
                # no csc_model, so just do business as usual
                beta_score = torch.ones((current_batch_size, 1)).to(self.device)

            else:
                # propensity_scores dim is batch_size
                beta_score, clipping_factor = self.get_propensity(csc_model, pre_act_rew, obs)
                beta_score = beta_score.to(self.device)
                list_prox_coefs.append(self.prox_coef)

            ########
            # Select action according to policy and add clipped noise
            # mu'(s_t) = mu(s_t | \theta_t) + N (Eq.7 in https://arxiv.org/abs/1509.02971)
            # OR
            # Eq. 15 in TD3 paper:
            # e ~ clip(N(0, \sigma), -c, c)
            ########
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_obs, act_rew) + noise).clamp(0, 1)
            ########
            #  Update critics
            #  1. Compute the target Q value
            #  2. Get current Q estimates
            #  3. Compute critic loss
            #  4. Optimize the critic
            ########

            # 1. y = r + \gamma * min{Q1, Q2} (s_next, next_action)
            # if done , then only use reward otherwise reward + (self.gamma * target_Q)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action, act_rew)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * (-1)).add(1) * (-1) * self.discount * target_Q

            # 2.  Get current Q estimates
            current_Q1, current_Q2 = self.critic(obs, action, pre_act_rew)

            # 3. Compute critic loss
            # even we picked min Q, we still need to backprob to both Qs
            critic_loss_temp = F.mse_loss(current_Q1, target_Q, reduction='none') + F.mse_loss(current_Q2, target_Q, reduction='none')

            assert critic_loss_temp.shape == beta_score.shape, (
            'shape critic_loss_temp and beta_score shoudl be the same', critic_loss_temp.shape, beta_score.shape)

            critic_loss = (critic_loss_temp * beta_score).mean()
            critic_loss_out += critic_loss.item()

            if apply_prox:
                # calculate proximal term
                critic_prox = self.get_prox_penalty(self.critic, self.ckpt['critic'])
                critic_loss = critic_loss + self.prox_coef * critic_prox
                critic_prox_out += critic_prox.item()

            # 4. Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            ########
            # Delayed policy updates
            ########
            if it % self.policy_freq == 0:

                # Compute actor loss
                actor_loss_temp = -1 * beta_score * self.critic.Q1(obs, self.actor(obs, pre_act_rew), pre_act_rew)
                actor_loss = actor_loss_temp.mean()
                actor_loss_out += actor_loss.item()

                if apply_prox:
                    # calculate proximal term
                    actor_prox = self.get_prox_penalty(self.actor, self.ckpt['actor'])
                    actor_loss = actor_loss + self.prox_coef * actor_prox
                    actor_prox_out += actor_prox.item()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        out = {}
        out['critic_loss'] = critic_loss_out / iterations
        out['actor_loss'] = self.policy_freq * actor_loss_out / iterations
        out['prox_critic'] = critic_prox_out / iterations
        out['prox_actor'] = self.policy_freq * actor_prox_out / iterations
        out['beta_score'] = beta_score.cpu().data.numpy().mean()

        # if csc_model and self.use_ess_clipping == True:
        out['avg_prox_coef'] = np.mean(list_prox_coefs)

        return out

    def adapt(self,
            batch_size,
            train_tasks_list,
            task_id=None,
            snap_iter_nums=5,
            main_snap_iter_nums=15,
            sample_mult=1

    ):
        '''
            inputs:
                replay_buffer
                iterations episode_timesteps
        '''
        #######
        # Reset optim at the beginning of the adaptation
        #######
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        #######
        # Adaptaion step:
        # learn a model to correct covariate shift
        #######
        # train covariate shift correction model
        csc_model, csc_info = self.train_cs(train_tasks_list=train_tasks_list,
                                            task_id=task_id)

        # train td3 for a single task
        out_single = self.do_training(train_replay_buffer=self.eval_task_buffer.get_buffer(task_id),
                                       iterations=snap_iter_nums,
                                       csc_model=None,
                                       apply_prox=False,
                                       current_batch_size=self.eval_task_buffer.size_rb(task_id))
        # self.copy_model_params()

        # keep a copy of model params for task task_id
        out_single['csc_info'] = csc_info
        out_single['snap_iter'] = snap_iter_nums

        # sampling_style is based on 'replay'
        # each train task has own buffer, so sample from each of them
        out = self.do_training(train_replay_buffer=self.replay_memory,
                                iterations=main_snap_iter_nums,
                                csc_model=csc_model,
                                apply_prox=True,
                                current_batch_size=sample_mult * batch_size)

        return out, out_single

    def rollback(self):
        '''
            This function rollback everything to state before test-adaptation
        '''

        ####### ####### ####### Super Important ####### ####### #######
        # It is very important to make sure that we rollback everything to
        # Step 0
        ####### ####### ####### ####### ####### ####### ####### #######
        self.actor.load_state_dict(self.actor_copy.state_dict())
        self.actor_target.load_state_dict(self.actor_target_copy.state_dict())
        self.critic.load_state_dict(self.critic_copy.state_dict())
        self.critic_target.load_state_dict(self.critic_target_copy.state_dict())
        self.actor_optimizer.load_state_dict(self.actor_optimizer_copy.state_dict())
        self.critic_optimizer.load_state_dict(self.critic_optimizer_copy.state_dict())



    def save_model_states(self):
        ####### ####### ####### Super Important ####### ####### #######
        # Step 0: It is very important to make sure that we save model params before
        # do anything here
        ####### ####### ####### ####### ####### ####### ####### #######
        self.actor_copy = deepcopy(self.actor)
        self.actor_target_copy = deepcopy(self.actor_target)
        self.critic_copy = deepcopy(self.critic)
        self.critic_target_copy = deepcopy(self.critic_target)
        self.actor_optimizer_copy  = deepcopy(self.actor_optimizer)
        self.critic_optimizer_copy = deepcopy(self.critic_optimizer)

    def save_model(self, model_dir, title):
        torch.save(self.critic.state_dict(), '{}/{}_critic.pth'.format(model_dir, title))
        torch.save(self.critic_optimizer.state_dict(), '{}/{}_critic_optimizer.pth'.format(model_dir, title))

        torch.save(self.actor.state_dict(), '{}/{}_actor.pth'.format(model_dir, title))
        torch.save(self.actor_optimizer.state_dict(), '{}/{}_actor_optimizer.pth'.format(model_dir, title))


    def load_model(self, model_name):
        self.critic.load_state_dict(torch.load('{}_critic.pth'.format(model_name)))
        self.critic_optimizer.load_state_dict(torch.load('{}_critic_optimizer.pth'.format(model_name)))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load('{}_actor.pth'.format(model_name)))
        self.actor_optimizer.load_state_dict(torch.load('{}_actor_optimizer.pth'.format(model_name)))
        self.actor_target = copy.deepcopy(self.actor)
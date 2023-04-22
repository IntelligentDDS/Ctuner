from __future__ import  print_function, division
import torch.optim as optim
from copy import deepcopy
from sklearn.linear_model import LogisticRegression as logistic
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


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
                                num_layers=2)

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

class meta_TD3:
    def __init__(self,
                state_dim,
                action_dim,
                lr = 0.001,
                action_noise = 0.1,
                gamma = 0.99,
                ptau = 0.005,
                policy_freq = 2,                 # 更新频率
                batch_size = 128,
                max_iter_logistic = 2000,
                beta_clip = 1,
                enable_beta_obs_cxt = False,
                prox_coef = 0.1,
                lam_csc = 0.50,
                use_ess_clipping = False,
                use_normalized_beta = True,    # 可试试False
                reset_optims = False,
                ):

        '''
            lr:   learning rate for RMSProp
            gamma: reward discounting parameter
            ptau:  Interpolation factor in polyak averaging
            policy_freq: delayed policy updates
            beta_clip: Range to clip beta term in CSC
            enable_beta_obs_cxt:  decide whether to concat obs and ctx for logistic regresstion
            prox_coef: Prox lambda
            lam_csc: logisitc regression reg, samller means stronger reg
            reset_optims: init optimizers at the start of adaptation
        '''
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.action_noise = action_noise
        self.gamma = gamma
        self.ptau = ptau
        self.policy_freq = policy_freq
        self.batch_size = batch_size
        self.max_iter_logistic = max_iter_logistic
        self.beta_clip = beta_clip
        self.enable_beta_obs_cxt = enable_beta_obs_cxt
        self.prox_coef = prox_coef
        self.prox_coef_init = prox_coef
        self.device = device
        self.lam_csc = lam_csc
        self.use_ess_clipping = use_ess_clipping
        self.r_eps = np.float32(1e-7)  # this is used to avoid inf or nan in calculations
        self.use_normalized_beta = use_normalized_beta
        self.lr = lr
        self.reset_optims = reset_optims

        # keep a copy of model params which will be used for proximal point
        self.copy_model_params()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = lr)

        print('-----------------------------')
        print('Optim Params')
        print("Actor:\n ",  self.actor_optimizer)
        print("Critic:\n ", self.critic_optimizer )
        print('********')
        print("reset_optims: ", reset_optims)
        print("use_ess_clipping: ", use_ess_clipping)
        print("use_normalized_beta: ", use_normalized_beta)
        print("enable_beta_obs_cxt: ", enable_beta_obs_cxt)
        print('********')
        print('-----------------------------')

    def copy_model_params(self):
        '''
            Keep a copy of actor and critic for proximal update
        '''
        self.ckpt = {
                        'actor': deepcopy(self.actor),
                        'critic': deepcopy(self.critic)
                    }

    def set_tasks_list(self, tasks_idx):
        '''
            Keep copy of task lists
        '''
        self.train_tasks_list = set(tasks_idx.copy())


    def select_action(self, obs, previous_action, previous_reward, previous_obs, train=True):
        '''
            return action
        '''
        obs = torch.FloatTensor(obs.reshape(1, -1)).to(self.device)
        previous_action = torch.FloatTensor(previous_action.reshape(1, -1)).to(self.device)
        previous_reward = torch.FloatTensor(previous_reward.reshape(1, -1)).to(self.device)
        previous_obs = torch.FloatTensor(previous_obs.reshape(1, -1)).to(self.device)

        # combine all other data here before send them to actor
        # torch.cat([previous_action, previous_reward], dim = -1)
        pre_act_rew = [previous_action, previous_reward, previous_obs]

        action = self.actor(obs, pre_act_rew)
        if train:
            # exploration noise
            noise = torch.tensor(np.random.normal(loc=0.0, scale=self.action_noise),
                                 dtype=torch.float).to(self.device)
            action = torch.clamp(action + noise, 0, 1)

        return action.squeeze().detach().cpu().numpy()


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

    def train_cs(self, task_id = None, snap_buffer = None, train_tasks_buffer = None, adaptation_step = False):
        '''
            This function trains covariate shift correction model
        '''

        ######
        # fetch all_data
        ######
        if adaptation_step == True:
            # step 1: calculate how many samples per classes we need
            # in adaption step, all train task can be used
            task_bsize = int(snap_buffer.size_rb(task_id) / (len(self.train_tasks_list))) + 2
            neg_tasks_ids = self.train_tasks_list

        else:
            # step 1: calculate how many samples per classes we need
            task_bsize = int(snap_buffer.size_rb(task_id) / (len(self.train_tasks_list) - 1)) + 2
            neg_tasks_ids = list(self.train_tasks_list.difference(set([task_id])))

        # collect examples from other tasks and consider them as one class
        # view --> len(neg_tasks_ids),task_bsize, D ==> len(neg_tasks_ids) * task_bsize, D
        pu, pr, px, xx = train_tasks_buffer.sample(task_ids = neg_tasks_ids, batch_size = task_bsize)
        neg_actions = torch.FloatTensor(pu).view(task_bsize * len(neg_tasks_ids), -1).to(self.device)
        neg_rewards = torch.FloatTensor(pr).view(task_bsize * len(neg_tasks_ids), -1).to(self.device)
        neg_obs = torch.FloatTensor(px).view(task_bsize * len(neg_tasks_ids), -1).to(self.device)
        neg_xx = torch.FloatTensor(xx).view(task_bsize * len(neg_tasks_ids), -1).to(self.device)

        # sample cuurent task and consider it as another class
        # returns size: (task_bsize, D)
        ppu, ppr, ppx, pxx = snap_buffer.sample(task_ids = [task_id], batch_size = snap_buffer.size_rb(task_id))
        pos_actions = torch.FloatTensor(ppu).to(self.device)
        pos_rewards = torch.FloatTensor(ppr).to(self.device)
        pos_obs = torch.FloatTensor(ppx).to(self.device)
        pos_pxx = torch.FloatTensor(pxx).to(self.device)

        # combine reward and action and previous states for context network.
        pos_act_rew_obs  = [pos_actions, pos_rewards, pos_obs]
        neg_act_rew_obs  = [neg_actions, neg_rewards, neg_obs]

        ######
        # extract features: context features
        ######
        with torch.no_grad():
            # batch_size X context_hidden
            # self.actor.get_conext_feats outputs, [batch_size , context_size]
            # torch.cat ([batch_size , obs_dim], [batch_size , context_size]) ==> [batch_size, obs_dim + context_size ]
            if self.enable_beta_obs_cxt == True:
                snap_ctxt = torch.cat([pos_pxx, self.actor.get_conext_feats(pos_act_rew_obs)], dim = -1).cpu().data.numpy()
                neg_ctxt = torch.cat([neg_xx, self.actor.get_conext_feats(neg_act_rew_obs)], dim = -1).cpu().data.numpy()
            else:
                snap_ctxt = self.actor.get_conext_feats(pos_act_rew_obs).cpu().data.numpy()
                neg_ctxt = self.actor.get_conext_feats(neg_act_rew_obs).cpu().data.numpy()


        ######
        # Train logistic classifiers
        ######
        x = np.concatenate((snap_ctxt, neg_ctxt)) # [b1 + b2] X D
        y = np.concatenate((-np.ones(snap_ctxt.shape[0]), np.ones(neg_ctxt.shape[0])))

        # model params : [1 , D] wehere D is context_hidden
        model = logistic(solver='lbfgs', max_iter = self.max_iter_logistic, C = self.lam_csc).fit(x,y)
        # keep track of how good is the classifier
        predcition_score = model.score(x, y)

        info = (snap_ctxt.shape[0], neg_ctxt.shape[0],  model.score(x, y))
        #print(info)
        return model, info

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

    def get_propensity(self, cs_model, curr_pre_act_rew, curr_obs):
        '''
            This function returns propensity for current sample of data
            simply: exp(f(x))
        '''

        ######
        # extract features: context features
        ######
        with torch.no_grad():

            # batch_size X context_hidden
            if self.enable_beta_obs_cxt == True:
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
            # get logistic regression prediction of class [-1] for current task
            lr_prob = cs_model.predict_proba(ctxt)[:,0]
            # normalize using logistic_probs
            d_pmax_pmin = np.float32(np.max(lr_prob) - np.min(lr_prob))
            f_score = ( d_pmax_pmin * (f_score - torch.min(f_score)) )/( torch.max(f_score) - torch.min(f_score) + self.r_eps ) + np.float32(np.min(lr_prob))

        # update prox coeff with ess.
        if self.use_ess_clipping == True:
            self.update_prox_w_ess_factor(cs_model, ctxt, beta=f_score)

        return f_score, None

    def do_training(self,
                    replay_buffer = None,
                    iterations = None,
                    csc_model = None,
                    apply_prox = False,
                    current_batch_size = None,
                    src_task_ids = []):

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
            if len(src_task_ids) > 0:
                x, y, u, r, d, pu, pr, px, nu, nr, nx = replay_buffer.sample_tasks(task_ids = src_task_ids, batch_size = current_batch_size)

            else:
                x, y, u, r, d, pu, pr, px, nu, nr, nx = replay_buffer.sample(current_batch_size)

            obs = torch.FloatTensor(x).to(self.device)
            next_obs = torch.FloatTensor(y).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)
            mask = torch.FloatTensor(1 - d).to(self.device)
            previous_action = torch.FloatTensor(pu).to(self.device)
            previous_reward = torch.FloatTensor(pr).to(self.device)
            previous_obs = torch.FloatTensor(px).to(self.device)

            # list of hist_actions and hist_rewards which are one time ahead of previous_ones
            # example:
            # previous_action = [t-3, t-2, t-1]
            # hist_actions    = [t-2, t-1, t]
            hist_actions = torch.FloatTensor(nu).to(self.device)
            hist_rewards = torch.FloatTensor(nr).to(self.device)
            hist_obs     = torch.FloatTensor(nx).to(self.device)


            # combine reward and action
            act_rew = [hist_actions, hist_rewards, hist_obs] # torch.cat([action, reward], dim = -1)
            pre_act_rew = [previous_action, previous_reward, previous_obs] #torch.cat([previous_action, previous_reward], dim = -1)

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
            noise = torch.tensor(np.random.normal(loc=0.0, scale=self.action_noise),
                                 dtype=torch.float).to(self.device)
            next_action = torch.clamp(self.actor_target(next_obs, act_rew) + noise, 0, 1)

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
            target_Q = reward + (mask * self.gamma * target_Q).detach()

            # 2.  Get current Q estimates
            current_Q1, current_Q2 = self.critic(obs, action, pre_act_rew)


            # 3. Compute critic loss
            # even we picked min Q, we still need to backprob to both Qs
            critic_loss_temp = F.mse_loss(current_Q1, target_Q, reduction='none') + F.mse_loss(current_Q2, target_Q, reduction='none')
            assert critic_loss_temp.shape == beta_score.shape, ('shape critic_loss_temp and beta_score shoudl be the same', critic_loss_temp.shape, beta_score.shape)

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
                    target_param.data.copy_(self.ptau * param.data + (1 - self.ptau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.ptau * param.data + (1 - self.ptau) * target_param.data)

        out = {}
        if iterations == 0:
            out['critic_loss'] = 0
            out['actor_loss']  = 0
            out['prox_critic'] = 0
            out['prox_actor']  = 0
            out['beta_score']  = 0

        else:
            out['critic_loss'] = critic_loss_out/iterations
            out['actor_loss']  = self.policy_freq * actor_loss_out/iterations
            out['prox_critic'] = critic_prox_out/iterations
            out['prox_actor']  = self.policy_freq * actor_prox_out/iterations
            out['beta_score']  = beta_score.cpu().data.numpy().mean()

        #if csc_model and self.use_ess_clipping == True:
        out['avg_prox_coef'] = np.mean(list_prox_coefs)

        return out

    def train_TD3(
                self,
                replay_buffer=None,
                iterations=None):

        '''
            inputs:
                replay_buffer
                iterations episode_timesteps
            outputs:

        '''
        actor_loss_out = 0.0
        critic_loss_out = 0.0

        ### if there is no eough data in replay buffer, then reduce size of iteration to 20:
        #if replay_buffer.size_rb() < iterations or replay_buffer.size_rb() <  self.batch_size * iterations:
        #    temp = int( replay_buffer.size_rb()/ (self.batch_size) % iterations ) + 1
        #    if temp < iterations:
        #        iterations = temp

        for it in range(iterations):
            ########
            # Sample replay buffer
            ########
            x, y, u, r, d, pu, pr, px, nu, nr, nx = replay_buffer.sample(self.batch_size)
            obs = torch.FloatTensor(x).to(self.device)
            next_obs = torch.FloatTensor(y).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)
            mask = torch.FloatTensor(1 - d).to(self.device)
            previous_action = torch.FloatTensor(pu).to(self.device)
            previous_reward = torch.FloatTensor(pr).to(self.device)
            previous_obs = torch.FloatTensor(px).to(self.device)

            # list of hist_actions and hist_rewards which are one time ahead of previous_ones
            # example:
            # previous_action = [t-3, t-2, t-1]
            # hist_actions    = [t-2, t-1, t]
            hist_actions = torch.FloatTensor(nu).to(self.device)
            hist_rewards = torch.FloatTensor(nr).to(self.device)
            hist_obs     = torch.FloatTensor(nx).to(self.device)

            # combine reward and action
            act_rew = [hist_actions, hist_rewards, hist_obs] # torch.cat([action, reward], dim = -1)
            pre_act_rew = [previous_action, previous_reward, previous_obs] #torch.cat([previous_action, previous_reward], dim = -1)

            ########
            # Select action according to policy and add clipped noise
            # mu'(s_t) = mu(s_t | \theta_t) + N (Eq.7 in https://arxiv.org/abs/1509.02971)
            # OR
            # Eq. 15 in TD3 paper:
            # e ~ clip(N(0, \sigma), -c, c)
            ########
            noise = torch.tensor(np.random.normal(loc=0.0, scale=self.action_noise),
                                 dtype=torch.float).to(self.device)
            next_action = torch.clamp(self.actor_target(next_obs, act_rew) + noise, 0, 1)

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
            target_Q = reward + (mask * self.gamma * target_Q).detach()

            # 2.  Get current Q estimates
            current_Q1, current_Q2 = self.critic(obs, action, pre_act_rew)

            # 3. Compute critic loss
            # even we picked min Q, we still need to backprob to both Qs
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            critic_loss_out += critic_loss.item()

            # 4. Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            ########
            # Delayed policy updates
            ########
            if it % self.policy_freq == 0:

                # Compute actor loss
                actor_loss = -self.critic.Q1(obs, self.actor(obs, pre_act_rew), pre_act_rew).mean()
                actor_loss_out += actor_loss.item()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()


                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.ptau * param.data + (1 - self.ptau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.ptau * param.data + (1 - self.ptau) * target_param.data)

        out = {}
        out['critic_loss'] = critic_loss_out/iterations
        out['actor_loss'] = self.policy_freq * actor_loss_out/iterations

        # keep a copy of models' params
        self.copy_model_params()
        return out, None

    def adapt(self,
            train_replay_buffer = None,
            train_tasks_buffer = None,
            eval_task_buffer = None,
            task_id = None,
            snap_iter_nums = 5,
            main_snap_iter_nums = 15,
            sample_mult = 1
            ):
        '''
            inputs:
                replay_buffer
                iterations episode_timesteps
        '''
        #######
        # Reset optim at the beginning of the adaptation
        #######
        self.actor_optimizer = optim.Adam(self.actor.parameters())
        self.critic_optimizer = optim.Adam(self.critic.parameters())

        #######
        # Adaptaion step:
        # learn a model to correct covariate shift
        #######
        # train covariate shift correction model
        csc_model, csc_info = self.train_cs(task_id = task_id,
                                            snap_buffer = eval_task_buffer,
                                            train_tasks_buffer = train_tasks_buffer,
                                            adaptation_step = True)

        # train td3 for a single task
        out_single = self.do_training(replay_buffer = eval_task_buffer.get_buffer(task_id),
                                      iterations = snap_iter_nums,
                                      csc_model = None,
                                      apply_prox = False,
                                      current_batch_size = eval_task_buffer.size_rb(task_id))
        #self.copy_model_params()

        # keep a copy of model params for task task_id
        out_single['csc_info'] = csc_info
        out_single['snap_iter'] = snap_iter_nums

        # sampling_style is based on 'replay'
        # each train task has own buffer, so sample from each of them
        out = self.do_training(replay_buffer = train_replay_buffer,
                                   iterations = main_snap_iter_nums,
                                   csc_model = csc_model,
                                   apply_prox = True,
                                   current_batch_size = sample_mult * self.batch_size)

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


    def train(self,
              replay_buffer = None,
              iterations = None):
        '''
         This starts type of desired training
        '''
        return self.train_TD3(  replay_buffer = replay_buffer,
                                    iterations = iterations
                                )

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

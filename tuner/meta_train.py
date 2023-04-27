# -*- coding: utf-8 -*-
"""
Train the model
"""

import os
import sys
from util import utils
import numpy as np
import math
from collections import deque
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from environment import utils as eu
from environment import appEnv, knobs
from CausalModel import causaler
from model.multi_middle_memory import MultiMiddleMemory

if __name__ == '__main__':
    opt = utils.parse_cmd()
    default_knobs = opt.default_knobs
    cur_knobs_dict = None

    if opt.cur_knobs_path != '':
        with open('knobs_choose/' + opt.instance + '/' + opt.cur_knobs_path + '.pkl', 'rb') as f:
            cur = pickle.load(f)
        cur_knobs_dict = cur
        print(cur)
        default_knobs = len(cur_knobs_dict['cur_soft']) + len(cur_knobs_dict['cur_kernel'])

    if opt.rf_dict_path != '':
        with open('knobs_choose/' + opt.instance + '/' + opt.rf_dict_path + '.pkl', 'rb') as f:
            rf_dict = pickle.load(f)
        print(rf_dict)


    if opt.isToLowDim is True:
        default_knobs = opt.LowDimSpace['target_dim']

    # Create Environment
    env = appEnv.Server(wk_type=opt.workload, instance_name=opt.instance, num_metric=opt.metric_num,
                        tps_weight=opt.tps_weight, lowDimSpace=opt.LowDimSpace, task_name=opt.task_name,
                        cur_knobs_dict=cur_knobs_dict, n_client=opt.n_client)



    # Build models
    model = utils.get_model(default_knobs, opt)
    train_tasks, eval_tasks = opt.train_env_workload, opt.test_env_workload
    model.init_tasks_buffer(train_tasks)

    # 判断是否继续训练
    init_episode = 0
    exist_task_episode, result_dir = env.find_exist_task_result()
    eu._print(f'previous results found, with max task_episode={exist_task_episode}')
    if opt.model_name == '':
        for file in sorted(result_dir.glob('*')):
            file.unlink()
        eu._print('all deleted')
    else:
        model.load_model('model_params/' + opt.instance + '/' + opt.model_name)
        eu._print(f'continue with task_episode={exist_task_episode}')
        init_episode = exist_task_episode

    # 初始化训练文件夹
    utils.init_dir(opt.instance)

    expr_name = 'train_{}_{}'.format(opt.method, str(utils.get_timestamp()))

    logger = utils.Logger(
        name=opt.method,
        log_file='log/{}/{}.log'.format(opt.instance, expr_name)
    )

    current_knob = knobs.get_init_knobs()
    knobs_info = knobs.get_KNOB_DETAILS()

    # decay rate
    step_counter = 0
    train_step = 0
    accumulate_loss = [0, 0]

    fine_state_actions = []

    if opt.memory != '':
        model.replay_memory.load_memory('save_memory/' + opt.instance + '/' + opt.memory + '.pkl')
        model.tasks_buffer.load_memory('save_multi_memory/' + opt.instance + '/' + opt.memory + '.pkl')
        print("Load Memory: {}".format(len(model.replay_memory)))



    # time for every step
    step_times = []
    # time for training
    train_step_times = []
    # time for setup, restart, test
    env_step_times = []
    # restart time
    env_restart_times = []
    # choose_action_time
    action_step_times = []

    sigma = 0.2
    theta = 0.15

    # ------------------------form generation----------------------------
    knobs_head = env.get_knobs_keys()
    objects_head = ['tps', 'latency']

    multi_middle_memory = None
    if opt.millde_memory != '':
        multi_middle_memory = MultiMiddleMemory(train_tasks)
        multi_middle_memory.load_memory('multi_middle_memory/' + opt.instance + '/' + opt.millde_memory + '.pkl')
        print("Load Memory:")
        print(multi_middle_memory.size_all())

    # --------------------------start training------------------------
    cm = None
    by = False
    for episode in range(opt.epoches):
        for tidx in train_tasks:
            env.set_task(tidx)
            # Initialize the environment
            current_state, external_metrics = env.initialize(logger)
            logger.info("\n[Env initialized][Metric tps: {} lat: {}]".format(
                external_metrics[0], external_metrics[1]))

            # model reset
            if hasattr(model, 'reset'):
                model.reset(sigma, theta)

            # Dynamically set performance test timeout
            test_time = env.get_test_time()
            test_async_value = math.ceil(test_time * 2)
            test_poll_value = math.ceil(test_time / 2)

            ########
            ## create a queue to keep track of past rewards and actions
            ########
            rewards_hist = deque(maxlen=opt.history_length)
            actions_hist = deque(maxlen=opt.history_length)
            obsvs_hist = deque(maxlen=opt.history_length)

            next_hrews = deque(maxlen=opt.history_length)
            next_hacts = deque(maxlen=opt.history_length)
            next_hobvs = deque(maxlen=opt.history_length)

            # Given batching schema, I need to build a full seq to keep in replay buffer
            # Add to all zeros.
            zero_action = np.zeros(default_knobs)
            zero_obs = np.zeros(opt.metric_num)

            for _ in range(opt.history_length):
                rewards_hist.append(0)
                actions_hist.append(zero_action.copy())
                obsvs_hist.append(zero_obs.copy())

                # same thing for next_h*
                next_hrews.append(0)
                next_hacts.append(zero_action.copy())
                next_hobvs.append(zero_obs.copy())

            if episode == 0:
                if cm is None:
                    states_head = env.get_stats_keys()
                    cm = causaler.Causality(knobs_head, states_head, objects_head, knobs_info,
                                              'causal_memory/' + opt.instance + '/' + opt.cm_path + '.pkl')

                # ---------------------------warm up-----------------------
                if multi_middle_memory is not None:
                    for item in multi_middle_memory.get_mm(tidx):
                        # ----------------prologue-------------------
                        np_pre_actions = np.asarray(actions_hist,
                                                    dtype=np.float32).flatten()  # (hist, action_dim) => (hist *action_dim,)
                        np_pre_rewards = np.asarray(rewards_hist, dtype=np.float32)  # (hist, )
                        np_pre_obsers = np.asarray(obsvs_hist, dtype=np.float32).flatten()
                        # --------------------------------------

                        state, action, next_state, done, result = item
                        if type(result) is dict:
                            reward = env._get_reward(result['avg'], opt.reward_mode, True)
                        else:
                            reward = env._get_reward(result, opt.reward_mode, True)


                        if opt.cur_knobs_path != '':
                            if len(action) > default_knobs:
                                action = np.array(knobs.action_all_to_part(action))

                        if opt.rf_dict_path != '':
                            if len(action) < default_knobs:
                                action = np.array(knobs.action_part_to_all(action, rf_dict))

                        # ----------------Subsequent-------------------
                        print("action:", action)
                        next_hrews.append(reward)
                        next_hacts.append(action.copy())
                        next_hobvs.append(state.copy())
                        # np_next_hacts and np_next_hrews are required for TD3 alg
                        np_next_hacts = np.asarray(next_hacts, dtype=np.float32).flatten()  # (hist, action_dim) => (hist *action_dim,)
                        np_next_hrews = np.asarray(next_hrews, dtype=np.float32)  # (hist, )
                        np_next_hobvs = np.asarray(next_hobvs, dtype=np.float32).flatten()  # (hist, )
                        # --------------------------------------

                        current_value = model.add_sample(tidx, state, next_state, action, reward, done,
                                                         np_pre_actions, np_pre_rewards, np_pre_obsers,
                                                         np_next_hacts, np_next_hrews, np_next_hobvs,
                                                         result)

                        # new becomes old
                        rewards_hist.append(reward)
                        actions_hist.append(action.copy())
                        obsvs_hist.append(state.copy())

                        # ------------------------form generation----------------------------
                        if opt.generate_csv:
                            sampled_os_config, sampled_app_config = knobs.action_to_knobs(action, False)
                            current_knob = {**(sampled_app_config), **(sampled_os_config)}
                            knob_value = []
                            for item in knobs_head:
                                knob_value.append(current_knob[item])

                            if by == False:
                                if knob_value[12] == 0:
                                    knob_value[12] = 1
                                else:
                                    knob_value[12] = 0

                                if knob_value[20] == 0:
                                    knob_value[20] = 1
                                else:
                                    knob_value[20] = 0
                                by = True


                            datas = knob_value + list(next_state)

                            datas.append(tidx)
                            if type(result) is dict:
                                datas.append(result['avg'][0])
                                datas.append(result['avg'][1])
                            else:
                                datas.append(result[0])
                                datas.append(result[1])
                            cm.update_data(datas)
                            cm.add_rc((reward, current_value.tolist()[0][0], result))
                        # --------------------------------------------------------

                        if reward > 10:
                            fine_state_actions.append((state, action))

                        if len(model.replay_memory) > opt.batch_size:
                            losses = []
                            for i in range(6):
                                losses.append(model.update())
                                train_step += 1

                            accumulate_loss[0] += sum([x[0] for x in losses])
                            accumulate_loss[1] += sum([x[1] if x[1] is not None else 0 for x in losses])
                            logger.info('[{}][preheat] Critic: {} Actor: {}'.format(
                                opt.method, accumulate_loss[0] / train_step, accumulate_loss[1] / train_step
                            ))

                # --------------------------------------------------------

            step = 1
            done = False
            while not done:
                # ----------------prologue-------------------
                np_pre_actions = np.asarray(actions_hist,
                                            dtype=np.float32).flatten()  # (hist, action_dim) => (hist *action_dim,)
                np_pre_rewards = np.asarray(rewards_hist, dtype=np.float32)  # (hist, )
                np_pre_obsers = np.asarray(obsvs_hist, dtype=np.float32).flatten()
                # --------------------------------------
                step_time = utils.time_start()
                state = current_state

                action_step_time, action, current_knob = utils.get_new_konbs(model, cm, state, episode, step, logger,
                                                                             opt.method, opt.threshold, tidx, True,
                                                                             np.array(np_pre_actions), np.array(np_pre_rewards), np.array(np_pre_obsers),
                                                                             cur_knobs_dict)

                print("action:", action)
                print("current_knob:", current_knob)

                # Deploy to an environment to view status and rewards
                env_step_time = utils.time_start()
                reward, state_, done, score, metrics, restart_time = env.step(episode, step, opt.reward_mode, logger)

                env_step_time = utils.time_end(env_step_time)

                # ----------------Subsequent-------------------
                next_hrews.append(reward)
                next_hacts.append(action.copy())
                next_hobvs.append(state.copy())
                # np_next_hacts and np_next_hrews are required for TD3 alg
                np_next_hacts = np.asarray(next_hacts,
                                           dtype=np.float32).flatten()  # (hist, action_dim) => (hist *action_dim,)
                np_next_hrews = np.asarray(next_hrews, dtype=np.float32)  # (hist, )
                np_next_hobvs = np.asarray(next_hobvs, dtype=np.float32).flatten()  # (hist, )
                # --------------------------------------

                # 日志记录
                logger.info(
                    "\n[{}][Episode: {}][Workload:{}][Step: {}][Metric tps:{} lat:{}]Reward: {} Score: {} Done: {}".format(
                        opt.method, episode, tidx, step, metrics['avg'][0], metrics['avg'][1], reward, score, done
                    ))

                env_restart_times.append(restart_time)

                next_state = state_



                new_memory_data = (state, action, next_state, False, metrics)
                # print(memory_data)
                multi_middle_memory.add(tidx, new_memory_data)
                current_value = model.add_sample(tidx, state, next_state, action, reward, done,
                                                 np_pre_actions, np_pre_rewards, np_pre_obsers,
                                                 np_next_hacts, np_next_hrews, np_next_hobvs,
                                                 metrics)

                if (episode != 0 and score < -100 and step > opt.t / 2) or step > opt.t:
                    done = True

                # new becomes old
                rewards_hist.append(reward)
                actions_hist.append(action.copy())
                obsvs_hist.append(state.copy())

                # ------------------------form generation----------------------------
                if opt.generate_csv:
                    knob_value = []
                    for item in knobs_head:
                        knob_value.append(current_knob[item])
                    datas = knob_value + list(state_)
                    datas.append(tidx)
                    datas.append(metrics['avg'][0])
                    datas.append(metrics['avg'][1])
                    cm.update_data(datas)
                    cm.add_rc((reward, current_value.tolist()[0][0], metrics))
                # --------------------------------------------------------

                if reward > 10:
                    fine_state_actions.append((state, action))

                current_state = next_state
                train_step_time = 0.0
                if len(model.replay_memory) > opt.batch_size:
                    losses = []
                    train_step_time = utils.time_start()
                    for i in range(6):
                        losses.append(model.update())
                        train_step += 1
                    train_step_time = utils.time_end(train_step_time) / 2.0

                    accumulate_loss[0] += sum([x[0] for x in losses])
                    accumulate_loss[1] += sum([x[1] if x[1] is not None else 0 for x in losses])
                    logger.info('[{}][Episode: {}][Workload:{}][Step: {}] Critic: {} Actor: {}'.format(
                        opt.method, episode, tidx, step, accumulate_loss[0] / train_step, accumulate_loss[1] / train_step
                    ))

                # all_step time
                step_time = utils.time_end(step_time)
                step_times.append(step_time)
                # env_step_time
                env_step_times.append(env_step_time)
                # training step time
                train_step_times.append(train_step_time)
                # action step times
                action_step_times.append(action_step_time)

                logger.info("[{}][Episode: {}][Workload:{}][Step: {}] step: {}s env step: {}s train step: {}s restart time: {}s "
                            "action time: {}s"
                            .format(opt.method, episode, tidx, step, step_time, env_step_time, train_step_time, restart_time,
                                    action_step_time))

                logger.info("[{}][Episode: {}][Workload:{}][Step: {}][Average] step: {}s env step: {}s train step: {}s "
                            "restart time: {}s action time: {}s"
                            .format(opt.method, episode, tidx, step, np.mean(step_time), np.mean(env_step_time),
                                    np.mean(train_step_time), np.mean(restart_time), np.mean(action_step_times)))

                step += 1
                step_counter += 1

                # save replay memory
                if step_counter % 10 == 0:
                    model.replay_memory.save('save_memory/{}/{}.pkl'.format(opt.instance, expr_name))
                    model.tasks_buffer.save('save_multi_memory/{}/{}.pkl'.format(opt.instance, expr_name))
                    multi_middle_memory.save('multi_middle_memory/{}/{}.pkl'.format(opt.instance, expr_name))
                    cm.causal_memorys.save('causal_memory/{}/{}.pkl'.format(opt.instance, expr_name))
                    utils.save_state_actions(fine_state_actions,
                                             'save_state_actions/{}/{}.pkl'.format(opt.instance, expr_name))

                # save network
                if step_counter % 5 == 0:
                    model.save_model('model_params/{}'.format(opt.instance),
                                     title='{}_{}'.format(expr_name, step_counter))

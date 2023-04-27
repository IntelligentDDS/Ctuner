# -*- coding: utf-8 -*-
"""
evaluate the model
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

def compute_percentage(default, current):
    """ compute metrics percentage versus default settings
    Args:
        default: dict, metrics from default settings
        current: dict, metrics from current settings
    """
    delta_tps = 100 * (current[0] - default[0]) / default[0]
    delta_latency = 100 * (-current[1] + default[1]) / default[1]
    return delta_tps, delta_latency

if __name__ == '__main__':
    opt = utils.parse_cmd()

    default_knobs = opt.default_knobs

    if opt.isToLowDim is True:
        default_knobs = opt.LowDimSpace['target_dim']

    cur_knobs_dict = None
    if opt.cur_knobs_path != '':
        with open('knobs_choose/' + opt.instance + '/' + opt.cur_knobs_path +'.pkl', 'rb') as f:
            cur = pickle.load(f)
        cur_knobs_dict = cur
        print(cur)

    # Create Environment
    env = appEnv.Server(wk_type=opt.workload, instance_name=opt.instance, num_metric=opt.metric_num,
                        tps_weight=opt.tps_weight, lowDimSpace=opt.LowDimSpace, task_name=opt.task_name,
                        cur_knobs_dict=cur_knobs_dict, n_client=opt.n_client)

    # Build models
    model = utils.get_model(default_knobs, opt)

    if not os.path.exists('test_log/' + opt.instance):
        os.makedirs('test_log/' + opt.instance)

    if not os.path.exists('test_knob/' + opt.instance):
        os.makedirs('test_knob/' + opt.instance)

    expr_name = 'eval_{}_{}'.format(opt.method, str(utils.get_timestamp()))

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
    train_tasks = opt.train_env_workload
    model.init_tasks_buffer(train_tasks)

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

    current_value_list = []
    reward_list = []

    sigma = 0.2
    theta = 0.15
    # In evaluation, if the cur_score is lower than the default_score, skip deployment. default_score can be set by yourself.
    default_score = 1

    # --------------------------start evaluating------------------------
    ############# adaptation step #############
    if opt.adaptation:
        model.save_model_states()
    ############# ############### #############

    all_task_rewards = []
    expl_noise = 0.1
    episode = 0

    model.init_eval_tasks_buffer([opt.workload])

    current_state, default_metrics = env.initialize(logger)
    logger.info("\n[Env initialized][Metric tps: {} lat: {}]".format(
        default_metrics[0], default_metrics[1]))
    print("[Environment Intialize]Tps: {} Lat:{}".format(default_metrics[0], default_metrics[1]))
    print("------------------- Starting to Test -----------------------")

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

    step = 1
    done = False
    have_adapt = False

    max_length = opt.max_eval_length
    if opt.adaptation:
        max_length += opt.adapt_length

    cm = None
    best_now = None
    best_knob = {}
    generate_knobs = []
    max_score = 0
    max_idx = -1

    while step < max_length:
        if opt.adaptation and step > opt.adapt_length and not have_adapt:
            model.save_model_states()
            adapt_start_time = utils.time_start()
            stats_main, stats_csv = model.adapt(
                opt.batch_size,  # 64
                train_tasks,
                task_id=opt.workload,
                snap_iter_nums=int(max_length / 4),
                main_snap_iter_nums=max_length * 3
            )
            print("Adaptation time：", utils.time_end(adapt_start_time))
            print('--------Adaptation-----------')
            print('Task: ', opt.workload)
            print(("critic_loss: %.4f \n\ractor_loss: %.4f \n\rNo beta_score: %.4f ") %
                  (stats_csv['critic_loss'], stats_csv['actor_loss'], stats_csv['beta_score']))

            print(("\rsamples for CSC: (%d, %d) \n\rAccuracy on train: %.4f \n\rsnap_iter: %d ") %
                  (
                      stats_csv['csc_info'][0], stats_csv['csc_info'][1], stats_csv['csc_info'][2],
                      stats_csv['snap_iter']))
            print(("\rmain_critic_loss: %.4f \n\rmain_actor_loss: %.4f \n\rmain_beta_score: %.4f ") %
                  (stats_main['critic_loss'], stats_main['actor_loss'], stats_main['beta_score']))
            print(("\rmain_prox_critic %.4f \n\rmain_prox_actor: %.4f") % (
                stats_main['prox_critic'], stats_main['prox_actor']))

            if 'avg_prox_coef' in stats_main:
                print(("\ravg_prox_coef: %.4f" % (stats_main['avg_prox_coef'])))

            print('-----------------------------')
            have_adapt = True
        # ----------------prologue-------------------
        np_pre_actions = np.asarray(actions_hist,
                                    dtype=np.float32).flatten()  # (hist, action_dim) => (hist *action_dim,)
        np_pre_rewards = np.asarray(rewards_hist, dtype=np.float32)  # (hist, )
        np_pre_obsers = np.asarray(obsvs_hist, dtype=np.float32).flatten()
        # --------------------------------------
        step_time = utils.time_start()
        state = current_state

        action_start_time = utils.time_start()
        action_step_time, action, current_knob = utils.get_new_konbs(model, cm, state, episode, step, logger,
                                                                     opt.method, opt.threshold, opt.workload, True,
                                                                     np.array(np_pre_actions), np.array(np_pre_rewards),
                                                                     np.array(np_pre_obsers), False)
        print("Configuration generation time：", utils.time_end(action_start_time))
        logger.info("[{}] Action: {}".format(opt.method, action))

        # -------Get cur_score by critic-----------
        current_value = model.get_score(state, action, np_pre_actions, np_pre_rewards, np_pre_obsers)
        cur_score = current_value.tolist()[0][0]
        if cur_score < default_score:
            continue
        #-----------------------------------

        # Deploy to an environment to view status and rewards
        env_step_time = utils.time_start()
        reward, state_, done, score, metrics, restart_time = env.step(episode, step, opt.reward_mode, logger)

        env_step_time = utils.time_end(env_step_time)

        _tps, _lat = compute_percentage(default_metrics, metrics['avg'])
        cur_score = _tps * opt.tps_weight + _lat * (1 - opt.tps_weight)
        if cur_score > max_score:
            max_score = cur_score
            max_idx = step_counter
            best_knob = current_knob
            best_now = metrics['avg']


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
        # logging
        logger.info("\n[{}][Step: {}][Metric tps:{} lat:{}]Reward: {} Score: {} Done: {}".format(
            opt.method, step_counter, metrics['avg'][0], metrics['avg'][1], reward, score, done
        ))
        logger.info("[{}][Knob Idx: {}] tps increase: {}% lat decrease: {}%".format(
            opt.method, step_counter, _tps, _lat
        ))
        env_restart_times.append(restart_time)


        next_state = state_
        # {"tps_inc":xxx, "lat_dec": xxx, "metrics": xxx, "knob": xxx}
        current_value = model.add_sample(opt.workload, state, next_state, action, reward, done,
                         np_pre_actions, np_pre_rewards, np_pre_obsers,
                         np_next_hacts, np_next_hrews, np_next_hobvs,
                         metrics, False)
        current_value_list.append(current_value.tolist()[0][0])
        reward_list.append(reward)

        generate_knobs.append({"tps_inc": _tps, "lat_dec": _lat, "metrics": metrics['avg'], "knob": current_knob})
        with open('test_knob/' + opt.instance + '/' + expr_name + '.pkl', 'wb') as f:
            pickle.dump(generate_knobs, f)

        # new becomes old
        rewards_hist.append(reward)
        actions_hist.append(action.copy())
        obsvs_hist.append(state.copy())

        current_state = next_state
        train_step_time = 0.0

        # all_step time
        step_time = utils.time_end(step_time)
        step_times.append(step_time)
        # env_step_time
        env_step_times.append(env_step_time)
        # training step time
        train_step_times.append(train_step_time)
        # action step times
        action_step_times.append(action_step_time)

        logger.info("[{}][Step: {}] step: {}s env step: {}s train step: {}s restart time: {}s "
                    "action time: {}s"
                    .format(opt.method, step_counter, step_time, env_step_time, train_step_time, restart_time,
                            action_step_time))

        logger.info("[{}][Step: {}][Average] step: {}s env step: {}s train step: {}s "
                    "restart time: {}s action time: {}s"
                    .format(opt.method, step_counter, np.mean(step_time), np.mean(env_step_time),
                            np.mean(train_step_time), np.mean(restart_time), np.mean(action_step_times)))

        step += 1

    if opt.adaptation:
        model.rollback()


    print("------------------- Testing Finished -----------------------")
    print("-----------------------------------------")
    print("Knobs are saved at: {}".format('test_knob/' + opt.instance + '/' + expr_name + '.pkl'))
    print("Proposal Knob At {}".format(max_idx))
    print("best_performance(tps, lat): ", best_now[0], best_now[1])
    print("--------------------------------------------------")
    print("--------------------------------------------------")
    print("             App Recommended Configuration            ")
    app_setting, os_setting = knobs.get_os_app_setting()
    for k in best_knob.keys():
        v = best_knob[k]
        if k in list(app_setting.keys()):
            print(k+": ",v)
    print("--------------------------------------------------")
    print("--------------------------------------------------")
    print()
    print("--------------------------------------------------")
    print("--------------------------------------------------")
    print("             OS Recommended Configuration            ")
    for k in best_knob.keys():
        v = best_knob[k]
        if k in list(os_setting.keys()):
            print(k+": ",v)
    print("--------------------------------------------------")
    print("--------------------------------------------------")


    print(current_value_list)
    print(reward_list)

    if not os.path.exists('value_reward/' + opt.instance):
        os.makedirs('value_reward/' + opt.instance)

    cur = {
        'current_value_list': current_value_list,
        'reward_list': reward_list,
    }
    cur_path = 'value_reward/' + opt.instance + '/' + 'train_value_reward_{}.pkl'.format(str(utils.get_timestamp()))
    f = open(cur_path, 'wb')
    pickle.dump(cur, f)
    f.close()

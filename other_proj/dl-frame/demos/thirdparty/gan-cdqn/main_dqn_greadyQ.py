# coding: utf-8
import numpy as np
import os
import tensorflow as tf
from collections import deque
from itertools import chain
import sys
from ganrl.experiment.rl.utils_dqn_gan import dqn_gan_hyperpara, compute_average_reward
from ganrl.common.cmd_args import cmd_args
from ganrl.experiment.rl.utils import TestData, save_dr_results, DRandIPS, BehaviorPolicy, compute_reward_and_next_state


def construct_placeholder():
    global disp_action_feature, Xs_clicked, news_size, user_size, disp_indices
    global disp_2d_split_user_ind, history_order_indices, history_user_indices

    disp_action_feature = tf.placeholder(dtype=tf.float32, shape=[None, _f_dim])
    Xs_clicked = tf.placeholder(dtype=tf.float32, shape=[None, _f_dim])

    news_size = tf.placeholder(dtype=tf.int64, shape=[])
    user_size = tf.placeholder(dtype=tf.int64, shape=[])
    disp_indices = tf.placeholder(dtype=tf.int64, shape=[None, 2])

    disp_2d_split_user_ind = tf.placeholder(dtype=tf.int64, shape=[None])

    history_order_indices = tf.placeholder(dtype=tf.int64, shape=[None])
    history_user_indices = tf.placeholder(dtype=tf.int64, shape=[None])


def construct_p():
    global u_disp, p_disp, agg_variables, position_weight, user_states

    denseshape = [user_size, news_size]

    # (1) history feature --- net ---> clicked_feature
    # (1) construct cumulative history
    click_history = [[] for _ in range(_weighted_dim)]
    position_weight = [[] for _ in range(_weighted_dim)]
    for ii in range(_weighted_dim):
        position_weight[ii] = tf.get_variable('p_w'+str(ii), [_band_size], initializer=tf.constant_initializer(0.0001))
        position_weight_values = tf.gather(position_weight[ii], history_order_indices)
        weighted_feature = tf.multiply(Xs_clicked, tf.reshape(position_weight_values, [-1, 1]))  # Xs_clicked: section by _f_dim
        click_history[ii] = tf.segment_sum(weighted_feature, history_user_indices)
    user_states = tf.concat(click_history, axis=1)
    disp_history_feature = tf.gather(user_states, disp_2d_split_user_ind)

    # (4) combine features
    concat_disp_features = tf.reshape(tf.concat([disp_history_feature, disp_action_feature], axis=1), [-1, _f_dim * _weighted_dim + _f_dim])

    # (5) compute utility
    n1 = 256
    y1 = tf.layers.dense(concat_disp_features, n1, trainable=True, kernel_initializer=tf.truncated_normal_initializer(stddev=_E3_sd))  #, kernel_initializer=tf.truncated_normal_initializer(stddev=1e-4))
    y1 = tf.nn.elu(y1)
    # create layer2

    n2 = 32
    y2 = tf.layers.dense(y1, n2, trainable=True, kernel_initializer=tf.truncated_normal_initializer(stddev=_E3_sd))
    y2 = tf.nn.elu(y2)

    # output layer
    u_disp = tf.layers.dense(y2, 1, trainable=True, kernel_initializer=tf.truncated_normal_initializer(stddev=_E3_sd))

    # (5)
    u_disp = tf.reshape(u_disp, [-1])
    exp_u_disp = tf.exp(u_disp)
    sum_exp_disp = tf.segment_sum(exp_u_disp, disp_2d_split_user_ind) + float(np.exp(_noclick_weight))
    scatter_sum_exp_disp = tf.gather(sum_exp_disp, disp_2d_split_user_ind)
    p_disp = tf.div(exp_u_disp, scatter_sum_exp_disp)

    agg_variables = tf.global_variables()


def construct_reward():
    # 1. reward
    Reward_r = tf.segment_sum(tf.multiply(u_disp, p_disp), disp_2d_split_user_ind)
    Reward_1 = tf.segment_sum(p_disp, disp_2d_split_user_ind)

    reward_feed_dict = {Xs_clicked: [], history_order_indices: [], history_user_indices: [], disp_2d_split_user_ind: [], disp_action_feature:[]}

    trans_p = tf.reshape(p_disp, [-1, _k])

    return Reward_r, Reward_1, reward_feed_dict, trans_p


def construct_Q_and_loss():
    global current_action_space, y_label
    global action_states, action_space_mean, action_space_std
    global action_k_id

    # placeholder
    current_action_space = tf.placeholder(dtype=tf.float32, shape=[None, _f_dim])
    action_space_mean = tf.placeholder(dtype=tf.float32, shape=[None, _f_dim])
    action_space_std = tf.placeholder(dtype=tf.float32, shape=[None, _f_dim])
    y_label = tf.placeholder(dtype=tf.float32, shape=[None])
    # 1. user_states, use the global variable
    action_k_id = [[] for _ in range(_k)]
    for ii in range(_k):
        action_k_id[ii] = tf.placeholder(dtype=tf.int64, shape=[None])

    # 2. candidate action states
    action_states = tf.concat([action_space_mean, action_space_std], axis=1)

    # 3. action
    action_k_feature_gather = [[] for _ in range(_k)]
    for ii in range(_k):
        action_k_feature_gather[ii] = tf.gather(current_action_space, action_k_id[ii])

    # 3. Q: states, candidate, action
    concate_input_k = [[] for _ in range(_k)]
    # action_feature_list = []
    q_value_k = [[] for _ in range(_k)]
    reuse_k = False
    current_variables = tf.trainable_variables()
    for ii in range(_k):
        if ii>0:
            reuse_k = True
        action_feature_list = []
        action_feature_list.append(action_k_feature_gather[ii])
        concate_input_k[ii] = tf.concat([user_states, action_states] + action_feature_list, axis=1)
        concate_input_k[ii] = tf.reshape(concate_input_k[ii], [-1, _weighted_dim * _f_dim + 2 * _f_dim + _f_dim])

        with tf.variable_scope('Q-function', reuse=reuse_k):
            q_y1 = tf.layers.dense(concate_input_k[ii], 256, activation=tf.nn.elu, trainable=True, kernel_initializer=tf.truncated_normal_initializer(stddev=_q_sd))
            q_y2 = tf.layers.dense(q_y1, 32, activation=tf.nn.elu, trainable=True, kernel_initializer=tf.truncated_normal_initializer(stddev=_q_sd))
            q_value_k[ii] = tf.layers.dense(q_y2, 1, trainable=True, kernel_initializer=tf.truncated_normal_initializer(stddev=_q_sd))

        q_value_k[ii] = tf.reshape(q_value_k[ii], [-1])

    q_value = tf.reduce_sum(tf.stack(q_value_k), 0)
    loss = tf.reduce_mean(tf.squared_difference(q_value, y_label))
    opt_q = tf.train.AdamOptimizer(learning_rate=_lr)
    train_variable_q = list(set(tf.trainable_variables()) - set(current_variables))
    train_op_q = opt_q.minimize(loss, var_list=train_variable_q)

    sess.run(tf.variables_initializer(list(set(tf.global_variables()) - set(agg_variables))))

    q_feed_dict = {current_action_space: [], action_space_mean: [], action_space_std: [], Xs_clicked: [], history_order_indices: [], history_user_indices: [], y_label: []}
    for ii in range(_k):
        q_feed_dict[action_k_id[ii]] = []

    return q_feed_dict, loss, train_op_q


def construct_max_Q():

    global all_action_user_indices, all_action_tensor_indices, all_action_tensor_shape, action_count, action_space_count, all_action_id

    all_action_user_indices = tf.placeholder(dtype=tf.int64, shape=[None])
    all_action_tensor_indices = tf.placeholder(dtype=tf.int64, shape=[None, 2])
    all_action_tensor_shape = tf.placeholder(dtype=tf.int64, shape=[2])
    action_count = tf.placeholder(dtype=tf.int64, shape=[None])
    action_space_count = tf.placeholder(dtype=tf.int64, shape=[None])
    #
    all_action_id = tf.placeholder(dtype=tf.int64, shape=[None])

    # (1) first solve max Q1
    all_action_feature_gather = tf.gather(current_action_space, all_action_id)
    # action_features = tf.reshape(action_feature_gather, [-1, _k * _f_dim])

    user_states_scatter = tf.gather(user_states, all_action_user_indices)
    action_states_scatter = tf.gather(action_states, all_action_user_indices)

    max_action_k = [[] for _ in range(_k)]
    max_action_feature_k = [[] for _ in range(_k)]
    to_avoid_repeat_tensor = tf.zeros(tf.cast(all_action_tensor_shape, tf.int32))
    q_value_k = [[] for _ in range(_k)]
    for ii in range(_k):
        concate_input = tf.concat([user_states_scatter, action_states_scatter]+[all_action_feature_gather], axis=1)
        concate_input = tf.reshape(concate_input, [-1, _weighted_dim * _f_dim + 2 * _f_dim + _f_dim])
        with tf.variable_scope('Q-function', reuse=True):
            q1_y1 = tf.layers.dense(concate_input, 256, activation=tf.nn.elu, kernel_initializer=tf.truncated_normal_initializer(stddev=_q_sd))
            q1_y2 = tf.layers.dense(q1_y1, 32, activation=tf.nn.elu, kernel_initializer=tf.truncated_normal_initializer(stddev=_q_sd))
            q_value_k[ii] = tf.layers.dense(q1_y2, 1, kernel_initializer=tf.truncated_normal_initializer(stddev=_q_sd))

        q_value_k[ii] = tf.reshape(q_value_k[ii], [-1])
        q1_tensor = tf.sparse_to_dense(all_action_tensor_indices, all_action_tensor_shape, q_value_k[ii], default_value=-1000000000.0)
        q1_tensor += to_avoid_repeat_tensor
        # max_q1_value = tf.segment_max(q1_value_all, all_action_user_indices)
        max_action_k[ii] = tf.argmax(q1_tensor, axis=1)
        to_avoid_repeat_tensor += tf.one_hot(max_action_k[ii], tf.cast(all_action_tensor_shape[1], tf.int32), on_value=-1000000000.0, off_value=0.0)
        max_action_k[ii] = tf.add(max_action_k[ii], action_count)
        max_action_k[ii] = tf.gather(all_action_id, max_action_k[ii])
        max_action_feature_k[ii] = tf.gather(current_action_space, max_action_k[ii])
        max_action_k[ii] = max_action_k[ii] - action_space_count

    q_value_all = tf.reduce_sum(tf.stack(q_value_k), 0)
    max_q_value = tf.segment_max(q_value_all, all_action_user_indices)

    max_action = tf.stack(max_action_k, axis=1)
    max_action_disp_features = tf.concat(max_action_feature_k, axis=1)
    max_action_disp_features = tf.reshape(max_action_disp_features, [-1, _f_dim])

    max_q_feed_dict = {all_action_id: [], all_action_user_indices: [], all_action_tensor_indices: [], all_action_tensor_shape: [],
                         current_action_space: [], Xs_clicked: [], history_order_indices: [], history_user_indices: [],
                       action_count: [], action_space_count: [], action_space_mean: [], action_space_std: []}

    return max_q_value, max_action, max_action_disp_features, max_q_feed_dict


def repeated_test(_repeated_best_reward, n_test, behavior_pol):

    score = DRandIPS(n_test)

    for i_th in range(n_test):
        sim_vali_user = _users_to_test
        states = [[] for _ in range(len(sim_vali_user))]

        for t in range(_time_horizon):
            action_mean_tr, action_std_tr, action_user_indice_tr, action_tensor_indice_tr, action_shape_tr, \
            action_space_tr, states_tr, history_order_tr, history_user_tr, action_cnt_tr, action_space_cnt_tr, action_id_tr = form_max_q_feed_dict(sim_vali_user, states)

            max_q_feed_dict[all_action_id] = action_id_tr
            max_q_feed_dict[all_action_user_indices] = action_user_indice_tr
            max_q_feed_dict[all_action_tensor_indices] = action_tensor_indice_tr
            max_q_feed_dict[all_action_tensor_shape] = action_shape_tr
            max_q_feed_dict[current_action_space] = action_space_tr
            max_q_feed_dict[action_space_mean] = action_mean_tr
            max_q_feed_dict[action_space_std] = action_std_tr
            max_q_feed_dict[Xs_clicked] = states_tr
            max_q_feed_dict[history_order_indices] = history_order_tr
            max_q_feed_dict[history_user_indices] = history_user_tr
            max_q_feed_dict[action_count] = action_cnt_tr
            max_q_feed_dict[action_space_count] = action_space_cnt_tr
            # 1. find best recommend action
            best_action = sess.run([max_action, max_action_disp_features], feed_dict=max_q_feed_dict)
            best_action[0] = best_action[0].tolist()
            # 2. compute reward
            disp_2d_split_user = np.kron(np.arange(len(sim_vali_user)), np.ones(_k))
            reward_feed_dict[Xs_clicked] = states_tr
            reward_feed_dict[history_order_indices] = history_order_tr
            reward_feed_dict[history_user_indices] = history_user_tr
            reward_feed_dict[disp_2d_split_user_ind] = disp_2d_split_user
            reward_feed_dict[disp_action_feature] = best_action[1]
            [reward_u, transition_p] = sess.run([u_disp, trans_p], feed_dict=reward_feed_dict)
            reward_u = np.reshape(reward_u, [-1, _k])
            # 3. COMPUTE Behavior Policy
            p_weight = behavior_pol.weight(best_action[0], sim_vali_user)
            # 4. Return rewards and next states
            sim_vali_user, states, expected_reward, actual_reward, expected_click, actual_click\
                = compute_reward_and_next_state(sim_vali_user, states,  transition_p, reward_u, feature_space, best_action[0], _k)
            # 5. COMPUTE score
            score.inner_update(i_th, expected_reward, actual_reward, expected_click, actual_click, p_weight)

            if len(sim_vali_user) == 0:
                break

        score.outer_update(i_th)

    filename = 'DQN'+str(_k)+'_'+str(_noclick_weight)+'_greedyQ.pickle'
    save_dr_results(_time_horizon, _users_to_test, filename, score)

    return _repeated_best_reward


def form_max_q_feed_dict(user_set, states_id):

    # states_feature = np.zeros([len(user_set), _f_dim], dtype=np.float32)

    states_feature = [[] for _ in range(len(user_set))]
    history_order = [[] for _ in range(len(user_set))]  # np.zeros([len(user_set)], dtype=np.int64)
    history_user = [[] for _ in range(len(user_set))]  # np.arange(len(user_set), dtype=np.int64)

    action_space = []

    # action_indicate_u = [[] for _ in range(len(user_set))]

    # action_indicate = []

    action_id = []

    action_user_indice = []
    action_tensor_indice = []

    max_act_size = 0

    candidate_action_mean = [[] for _ in range(len(user_set))]
    candidate_action_std = [[] for _ in range(len(user_set))]

    action_cnt = [0 for _ in range(len(user_set))]

    action_space_cnt = [0 for _ in range(len(user_set))]

    for uu in range(len(user_set)):
        user = user_set[uu]

        candidate_action = []

        if len(states_id[uu]) == 0:
            states_feature[uu] = np.zeros([1, _f_dim], dtype=np.float32).tolist()
            history_order[uu].append(0)
            history_user[uu].append(uu)
        else:
            states_feature[uu] = deque(maxlen=_band_size)
            for idd in states_id[uu]:
                states_feature[uu].append(feature_space[user][idd])

            states_feature[uu] = list(states_feature[uu])
            id_cnt = len(states_feature[uu])
            history_order[uu] = np.arange(id_cnt, dtype=np.int64).tolist()
            history_user[uu] = list(uu * np.ones(id_cnt, dtype=np.int64))

        action_candidate = np.array(list(set(np.arange(len(feature_space[user]))) - set(states_id[uu])))

        for idd in action_candidate:
            candidate_action.append(feature_space[user][idd])

        candidate_action_mean[uu] = np.mean(np.array(candidate_action), axis=0)
        candidate_action_std[uu] = np.std(np.array(candidate_action), axis=0)

        action_space_cnt[uu] = len(action_space)

        action_id_u = list(action_candidate + action_space_cnt[uu])
        action_id += action_id_u
        # all possible actions
        # action_indicate_u[uu] = list(chain.from_iterable(combinations(action_candidate + action_space_cnt[uu], _k)))
        # action_indicate += action_indicate_u[uu]

        action_cnt[uu] = len(action_id_u)
        action_user_indice += [uu for _ in range(action_cnt[uu])]

        if action_cnt[uu] == 0:
            print('action_cnt 0')
            print(action_candidate)
            print(action_candidate + action_space_cnt[uu])
            print(states_id[uu])
        max_act_size = max(max_act_size, action_cnt[uu])
        # action_user_indice += [uu for _ in range(action_cnt[uu])]
        action_tensor_indice += map(lambda x: [uu, x], np.arange(action_cnt[uu]))

        # action space
        action_space += feature_space[user]

    action_cnt = np.cumsum(action_cnt)
    action_cnt = [0] + list(action_cnt[:-1])

    action_shape = [len(user_set), max_act_size]

    states_feature = list(chain.from_iterable(states_feature))
    history_order = list(chain.from_iterable(history_order))
    history_user = list(chain.from_iterable(history_user))

    return candidate_action_mean, candidate_action_std, action_user_indice, action_tensor_indice, action_shape, \
           action_space, states_feature, history_order, history_user, action_cnt, action_space_cnt, action_id


_band_size, _weighted_dim, _regularity, _E3_sd, _candidate_sd, _q_sd, _lr = dqn_gan_hyperpara()
_noclick_weight = cmd_args.nonclick_weight
_k = cmd_args.display_size
_f_dim = cmd_args.f_dim
_training_batch_size = cmd_args.train_batch_size
_sample_batch_size = cmd_args.sample_batch_size
_time_horizon = cmd_args.length_time


if __name__ == '__main__':
    np.random.seed(cmd_args.seed)
    #  (1) ===========
    #  LOAD..... test user model
    construct_placeholder()
    construct_p()
    sess = tf.Session()
    sess.run(tf.variables_initializer(agg_variables))
    full_path = os.path.realpath(__file__)
    path = os.path.abspath(os.path.dirname(full_path))
    save_dir = os.path.join(path, '../')+'/saved_models/'
    vali_path = save_dir+'/GAN_PW_best/'
    saver = tf.train.Saver(max_to_keep=None)
    save_path = os.path.join(vali_path, 'convert_best-loss')
    saver.restore(sess, save_path)

    #  (2) ===========
    #  LOAD..... saved pre-trained DQN
    Reward_r, Reward_1, reward_feed_dict, trans_p = construct_reward()
    q_feed_dict, loss_k, train_op_k = construct_Q_and_loss()
    max_q_value, max_action, max_action_disp_features, max_q_feed_dict = construct_max_Q()
    vali_path = save_dir + '/DQN'+str(_k)+'u'+str(_noclick_weight)+'greedyQ/'
    saver = tf.train.Saver(max_to_keep=None)
    save_path = os.path.join(vali_path, 'best-reward')
    saver.restore(sess, save_path)
    print('Pre-trained DQN loaded..')

    #  (3) ===========
    #  TEST
    print('TESTING')
    test_data = TestData(num_user=cmd_args.num_test_user)
    feature_space = test_data.feature_space
    _users_to_test = test_data.test_user
    print('Test data loaded')

    # Load Behavior Policy
    print('Loading behavior Policy')
    behavior_pol = BehaviorPolicy(_users_to_test, feature_space)
    _ = repeated_test(0.0, cmd_args.num_repeat, behavior_pol)

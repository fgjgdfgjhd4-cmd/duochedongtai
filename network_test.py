from abc_for_map import ABC
from map import Map
from CNN_PPO import PPO
import os
import glob
import time
from datetime import datetime
from itertools import cycle, islice
import math
import cv2
import matplotlib.pyplot as plt
import random
from torch.distributions import Categorical

import torch
import numpy as np
import logging
import os


#################################### Testing ###################################
def test():
    print("============================================================================================")

    # 清理不必要进程
    torch.cuda.empty_cache()

    env_name = "Map"
    has_continuous_action_space = False  # 输出的动作为各个方案的概率

    max_ep_len = 200  # max timesteps in one episode

    action_std = 0.1  # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)

    total_test_episodes = 1  # total num of testing episodes
    K_epochs = 80  # update policy for K epochs

    eps_clip = 0.2  # clip param for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network

    print("testing environment name: " + env_name)

    # # 5
    # checkpoint_path = "/PPO_preTrained/Map/10_robots/0725-09-25PPO_Map_4500.pth"
    # env = Map(agent_num=10, render_mode="human", test_result_save=True)

    # 10
    checkpoint_path = "/PPO_preTrained/Map/10_robots/0725-09-25PPO_Map_4500.pth"
    env = Map(agent_num=10, render_mode="human", test_result_save=True)

    seed = 42
    torch.manual_seed(seed=seed)

    # 用来画图
    # episode_reward_storage = []
    # episode_timestep_storage = []
    # episodes = []
    # timestep_reward_storage = []

    # 根据agent数量设置lb和ub
    lower_bound = [5, -math.pi / 4]
    upper_bound = [20, math.pi / 4]
    repeated_iter = cycle(lower_bound)
    fn_lb = list(islice(repeated_iter, len(lower_bound) * env.agent_num))

    repeated_iter = cycle(upper_bound)
    fn_ub = list(islice(repeated_iter, len(upper_bound) * env.agent_num))

    # artificial bee colony
    abc_for_map = ABC(30, 29, env.objective, fn_lb=fn_lb, fn_ub=fn_ub, env=env)
    # abc_for_map = ABC(15, 14, env.objective, fn_lb=fn_lb, fn_ub=fn_ub, env=env)

    success_ep_num = 0
    # action space dimension
    action_dim = 3  # 三个概率

    print("============================================================================================")

    ################# testing procedure ################
    # initialize a PPO agent
    ppo_agent = PPO(action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT): ", start_time)

    print("============================================================================================")

    ppo_agent.load(checkpoint_path)

    test_running_reward = 0

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0

        obs, _ = env.reset(seed=seed)
        current_ep_reward = 0
        initial_rgb_array = env.render()  # 输出形状为（800， 800， 3）

        initial_rgb_array_transposed = np.transpose(initial_rgb_array, (2, 0, 1))

        initial_rgb_array_transposed = np.expand_dims(initial_rgb_array_transposed, axis=0)

        rgb_array_transposed = torch.from_numpy(initial_rgb_array_transposed).float()

        for t in range(1, max_ep_len + 1):
            # adjust the probability of three strategies
            actions = {}
            probabilities = ppo_agent.select_action(rgb_array_transposed)
            probabilities = probabilities.cpu().numpy()
            probabilities = [round(value, 3) for value in probabilities.flat]

            abc_for_map.set_probability(probabilities)
            temp = abc_for_map.optimize()

            for i in range(len(env.agents)):
                agent = env.agents[i]
                actions[agent] = np.array([temp[2 * i], temp[2 * i + 1]])

            _, reward, done, _, _, _ = env.step(actions)
            ep_reward += reward
            # timestep_reward_storage.append(reward)
            if done:
                env.render()
            else:
                rgb_array = env.render()

                rgb_array_transposed = np.transpose(rgb_array, (2, 0, 1))

                rgb_array_transposed = np.expand_dims(rgb_array_transposed, axis=0)

                rgb_array_transposed = torch.from_numpy(rgb_array_transposed).float()

            if done:
                success_ep_num += 1
                print("success")
                break
            elif all([env.collisions[agent] or env.terminations[agent] for agent in env.agents]):
                print("failed")
                break

        # clean buffer
        ppo_agent.buffer.clear()

        test_running_reward += ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()
    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")

if __name__ == '__main__':

    test()

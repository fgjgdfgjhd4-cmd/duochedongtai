import copy

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
import time
import torch
import numpy as np
import logging
import os
from Compare_Method_ABC.swarm import ABC_origin
from Compare_Method_PSO.PSO_map import PSO
from CPSO.CPSO_map import CPSO
from Compare_Method_ABCL.ABCL_MAP import ABCL
import pygame

"""
这个是考虑到需要保证不同算法的测试环境和条件应该相同
"""

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

    total_test_episodes = 15  # total num of testing episodes
    K_epochs = 80  # update policy for K epochs

    eps_clip = 0.2  # clip param for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network

    print("testing environment name: " + env_name)

    # 5
    # checkpoint_path = "./PPO_preTrained/Map/5_robots/PPO_Map_4500.pth"
    # env = Map(agent_num=5, render_mode="human", test_result_save=True)
    #
    # seed = 42
    # torch.manual_seed(seed=seed)
    #
    # map_index = 2
    # initial_starts_map_1_robots_5 = [np.array([265, 670]), np.array([265, 130]), np.array([265, 400]),
    #                                  np.array([130, 535]), np.array([130, 265])]
    # initial_targets_map_1_robots_5 = [np.array([670, 535]), np.array([670, 265]), np.array([535, 400]),
    #                                   np.array([535, 670]), np.array([535, 130])]
    #
    # initial_starts_map_2_robots_5 = [np.array([100, 100]), np.array([700, 700]), np.array([100, 700]),
    #                                  np.array([700, 100]), np.array([150, 150])]
    # initial_targets_map_2_robots_5 = [np.array([400, 520]), np.array([400, 280]), np.array([280, 400]),
    #                                   np.array([520, 400]), np.array([400, 400])]
    #
    # initial_starts_map_3_robots_5 = [np.array([285, 400]), np.array([65, 290]), np.array([65, 510]),
    #                                  np.array([285, 630]), np.array([285, 170])]
    # initial_targets_map_3_robots_5 = [np.array([750, 400]), np.array([525, 170]), np.array([525, 630]),
    #                                   np.array([750, 510]), np.array([750, 290])]

    # 10
    checkpoint_path = "./PPO_preTrained/Map/10_robots/0115-09-31PPO_Map_4500.pth"
    env = Map(agent_num=10, render_mode="human", test_result_save=True)

    seed = 42
    torch.manual_seed(seed=seed)

    map_index = 2

    initial_starts_map_1_robots_10 = [np.array([265, 670]), np.array([265, 620]),
                                      np.array([265, 450]), np.array([265, 350]),
                                      np.array([265, 180]), np.array([265, 130]),
                                      np.array([130, 585]), np.array([130, 485]),
                                      np.array([130, 315]), np.array([130, 215])]


    initial_targets_map_1_robots_10 = [np.array([670, 585]), np.array([670, 485]),
                                       np.array([535, 450]), np.array([535, 350]),
                                       np.array([670, 315]), np.array([670, 215]),
                                       np.array([535, 720]), np.array([535, 620]),
                                       np.array([535, 180]), np.array([535, 130])]

    initial_starts_map_2_robots_10 = [np.array([100, 100]), np.array([700, 700]), np.array([100, 700]),
                                      np.array([700, 100]), np.array([150, 650]), np.array([650, 150]),
                                      np.array([650, 650]), np.array([150, 100]), np.array([200, 400]),
                                      np.array([600, 400])]

    initial_targets_map_2_robots_10 = [np.array([400, 520]), np.array([400, 280]), np.array([280, 400]),
                                       np.array([520, 400]), np.array([200, 200]), np.array([200, 600]),
                                       np.array([600, 200]), np.array([600, 600]), np.array([450, 400]),
                                       np.array([350, 400])]

    # initial_starts_map_3_robots_10 = [np.array([65, 400]), np.array([65, 510]), np.array([65, 290]),
    #                                   np.array([150, 510]), np.array([150, 290]), np.array([275, 650]),
    #                                   np.array([275, 510]), np.array([275, 400]), np.array([275, 290]),
    #                                   np.array([275, 150])]
    #
    # initial_targets_map_3_robots_10 = [np.array([525, 400]), np.array([525, 510]), np.array([525, 290]),
    #                                    np.array([525, 650]), np.array([525, 150]), np.array([650, 510]),
    #                                    np.array([735, 510]), np.array([735, 400]), np.array([735, 290]),
    #                                    np.array([650, 290])]

    initial_starts_map_3_robots_10 = [np.array([644, 115]), np.array([566, 461]), np.array([407, 592]),
                                      np.array([57, 571]), np.array([738, 468]), np.array([301, 746]),
                                      np.array([540, 367]), np.array([263, 190]), np.array([581, 507]),
                                      np.array([594, 560])]

    initial_targets_map_3_robots_10 = [np.array([250, 440]), np.array([467, 312]), np.array([557, 166]),
                                       np.array([479, 466]), np.array([495, 68]), np.array([657, 157]),
                                       np.array([296, 168]), np.array([486, 535]), np.array([300, 700]),
                                       np.array([737, 307])]

    starts = {}
    targets = {}
    # for i in range(env.agent_num):
    #     if map_index == 0:
    #         starts[env.possible_agents[i]] = initial_starts_map_1_robots_5[i]
    #         targets[env.possible_agents[i]] = initial_targets_map_1_robots_5[i]
    #     elif map_index == 1:
    #         starts[env.possible_agents[i]] = initial_starts_map_2_robots_5[i]
    #         targets[env.possible_agents[i]] = initial_targets_map_2_robots_5[i]
    #     elif map_index == 2:
    #         starts[env.possible_agents[i]] = initial_starts_map_3_robots_5[i]
    #         targets[env.possible_agents[i]] = initial_targets_map_3_robots_5[i]

    for i in range(env.agent_num):
        if map_index == 0:
            starts[env.possible_agents[i]] = initial_starts_map_1_robots_10[i]
            targets[env.possible_agents[i]] = initial_targets_map_1_robots_10[i]
            # starts[env.possible_agents[i]] = initial_starts_map_1_robots_15[i]
            # targets[env.possible_agents[i]] = initial_targets_map_1_robots_15[i]
        elif map_index == 1:
            starts[env.possible_agents[i]] = initial_starts_map_2_robots_10[i]
            targets[env.possible_agents[i]] = initial_targets_map_2_robots_10[i]
        elif map_index == 2:
            starts[env.possible_agents[i]] = initial_starts_map_3_robots_10[i]
            targets[env.possible_agents[i]] = initial_targets_map_3_robots_10[i]

    total_path_dis = 0

    # 根据agent数量设置lb和ub
    lower_bound = [5, -math.pi / 4]
    upper_bound = [20, math.pi / 4]
    repeated_iter = cycle(lower_bound)
    fn_lb = list(islice(repeated_iter, len(lower_bound) * env.agent_num))

    repeated_iter = cycle(upper_bound)
    fn_ub = list(islice(repeated_iter, len(upper_bound) * env.agent_num))

    """
    PSO的size和ABC的n_population对应，dimension就是动作维度，ABC里面应该没有
    """

    # artificial bee colony
    abc_for_map = ABC(npopulation=15, nruns=15, fn_eval=env.objective, fn_lb=fn_lb, fn_ub=fn_ub, env=env)

    # origin ABC
    origin_abc_for_map = ABC_origin(n_population=15, n_runs=15, fn_eval=env.objective, fn_lb=fn_lb, fn_ub=fn_ub, env=env)

    # PSO
    pso_for_map = PSO(dimension=env.agent_num*2, time=15, size=15, fn_lb=fn_lb, fn_ub=fn_ub, v_low=-3, v_high=3, env=env)

    # CPSO
    cpso_for_map = CPSO(dimension=env.agent_num*2, generation=15, size=15, fn_lb=fn_lb, fn_ub=fn_ub, v_low=-3, v_high=3, env=env)

    # ABCL
    abcl_for_map = ABCL(npopulation=15, nruns=15, fn_eval=env.objective, fn_lb=lower_bound, fn_ub=upper_bound, env=env)

    # 记录每种方法的成功次数
    ppo_abc_success_ep_num = 0
    abc_success_ep_num = 0
    pso_success_ep_num = 0
    cpso_success_ep_num = 0
    abcl_success_ep_num = 0

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

    ppo_abc_test_running_reward = 0
    abc_test_running_reward = 0
    pso_test_running_reward = 0

    ep_reward = 0

    time_store = []
    dis_store = []
    time_per_step = []

    """PPO-ABC testing"""
    for ep in range(0, 1):
        obs, _ = env.reset(seed=seed, starting_points=starts, targets=targets, map_index=map_index)

        current_ep_reward = 0
        initial_rgb_array = env.render()  # 输出形状为（800， 800， 3）
        initial_rgb_array_transposed = np.transpose(initial_rgb_array, (2, 0, 1))
        initial_rgb_array_transposed = np.expand_dims(initial_rgb_array_transposed, axis=0)
        rgb_array_transposed = torch.from_numpy(initial_rgb_array_transposed).float()

        ppo_abc_start_time = time.time()

        for t in range(1, max_ep_len + 1):
            # adjust the probability of three strategies
            start_time_one_step = time.time()


            actions = {}
            probabilities, strategy_index, raw_action_probs = ppo_agent.select_strategy_profile(
                rgb_array_transposed,
                deterministic=True,
            )
            probabilities = [round(value, 3) for value in probabilities]

            abc_for_map.set_probability(probabilities)
            temp = abc_for_map.optimize()
            end_time_one_step = time.time()
            time_per_step.append(end_time_one_step-start_time_one_step)

            for i in range(len(env.agents)):
                agent = env.agents[i]
                actions[agent] = np.array([temp[2 * i], temp[2 * i + 1]])

            # print(actions)
            _, reward, done, _, _, path_dis = env.step(actions)
            ep_reward += reward
            total_path_dis += np.round(path_dis)
            # timestep_reward_storage.append(reward)
            if done:
                env.render()
            else:
                rgb_array = env.render()

                rgb_array_transposed = np.transpose(rgb_array, (2, 0, 1))

                rgb_array_transposed = np.expand_dims(rgb_array_transposed, axis=0)

                rgb_array_transposed = torch.from_numpy(rgb_array_transposed).float()

            if done:
                ppo_abc_end_time = time.time()
                ppo_abc_success_ep_num += 1
                ppo_abc_lasting_time = np.round(ppo_abc_end_time - ppo_abc_start_time, 2)
                time_store.append(ppo_abc_lasting_time)
                dis_store.append(total_path_dis)
                print("success")
                # pygame.image.save(env.surf, "./PPO_ABC_test_images/5_robots/map3/PPO_ABC_ep_{}_test_result.png".format(ep))
                # pygame.image.save(env.surf, "./PPO_ABC_test_images/10_robots/map1/PPO_ABC_ep_{}_test_result.png".format(ep))
                # pygame.image.save(env.surf, "./PPO_ABC_test_images/10_robots/map2/PPO_ABC_ep_{}_test_result.png".format(ep))
                pygame.image.save(env.surf, "./PPO_ABC_test_images/10_robots/map3/PPO_ABC_ep_{}_test_result.png".format(ep))

                break
            elif all([env.collisions[agent] or env.terminations[agent] or env.truncations[agent] for agent in env.agents]):
                ppo_abc_end_time = time.time()
                ppo_abc_lasting_time = np.round(ppo_abc_end_time - ppo_abc_start_time, 2)
                print("failed")
                break

        # clean buffer
        ppo_agent.buffer.clear()

        ppo_abc_test_running_reward += ep_reward
        print('Algorithm: {} \t\t Episode: {} \t\t Success: {} \t\t Reward: {} \t\t Lasting time: {} \t\t Total Path Distance: {}'.format("PPO-ABC", ep, done, round(ep_reward, 2), ppo_abc_lasting_time, total_path_dis))
        total_path_dis = 0
        ep_reward = 0


    if len(time_store) > 0:
        print("Max Time: {} \t\t Min Time: {} \t\t Mean Time: {} \t\t Max Dis: {} \t\t Min Dis: {} \t\t Mean: {}".format(max(time_store), min(time_store), np.mean(time_store), max(dis_store), min(dis_store), np.mean(dis_store)))
    if len(time_per_step) > 0:
        print("Average time per step: {} \t Max: {} \t Min: {}".format(np.mean(time_per_step), max(time_per_step), min(time_per_step)))
    time_store = []
    dis_store = []
    time_per_step = []


    """ABC testing"""
    for ep in range(0, 0):

        """调整初始位置和终点"""
        obs, _ = env.reset(seed=seed, starting_points=starts, targets=targets, map_index=map_index)

        env.render()

        abc_start_time = time.time()

        for t in range(0, max_ep_len):
            start_time_one_step = time.time()

            actions = {}
            temp = origin_abc_for_map.optimize()

            end_time_one_step = time.time()
            time_per_step.append(end_time_one_step-start_time_one_step)

            for i in range(len(env.agents)):
                agent = env.agents[i]
                actions[agent] = np.array([temp[2 * i], temp[2 * i + 1]])

            # print(actions)
            _, reward, done, _, _, path_dis = env.step(actions)
            total_path_dis += np.round(path_dis)
            ep_reward += reward

            env.render()

            if done:
                abc_end_time = time.time()
                abc_success_ep_num += 1
                abc_lasting_time = np.round(abc_end_time - abc_start_time, 2)
                time_store.append(abc_lasting_time)
                dis_store.append(total_path_dis)
                print("success")

                # pygame.image.save(env.surf, "./ABC_test_images/5_robots/ABC_ep_{}_test_result.png".format(ep))
                # pygame.image.save(env.surf, "./ABC_test_images/10_robots/map1/ABC_ep_{}_test_result.png".format(ep))
                # pygame.image.save(env.surf, "./ABC_test_images/10_robots/map2/ABC_ep_{}_test_result.png".format(ep))
                pygame.image.save(env.surf, "./ABC_test_images/10_robots/map3/ABC_ep_{}_test_result.png".format(ep))


                break
            elif all([env.collisions[agent] or env.terminations[agent] or env.truncations[agent] for agent in env.agents]):
                abc_end_time = time.time()
                abc_lasting_time = np.round(abc_end_time - abc_start_time, 2)
                print("failed")
                break
            else:
                abc_end_time = time.time()
                abc_lasting_time = np.round(abc_end_time - abc_start_time, 2)


        abc_test_running_reward += ep_reward
        print('Algorithm: {} \t\t Episode: {} \t\t Success: {} \t\t Reward: {} \t\t Lasting time: {} \t\t Total Path Distance: {}'.format("ABC", ep, done, round(ep_reward, 2), abc_lasting_time, total_path_dis))
        total_path_dis = 0
        ep_reward = 0

    if len(time_store) > 0:
        print("Max Time: {} \t\t Min Time: {} \t\t Mean Time: {} \t\t Max Dis: {} \t\t Min Dis: {} \t\t Mean: {}".format(
                max(time_store), min(time_store), np.mean(time_store), max(dis_store), min(dis_store),
                np.mean(dis_store)))
    if len(time_per_step) > 0:
        print("Average time per step: {} \t Max: {} \t Min: {}".format(np.mean(time_per_step), max(time_per_step), min(time_per_step)))
    time_store = []
    dis_store = []
    time_per_step = []


    """PSO testing"""
    for ep in range(0, 0):

        """调整初始位置和终点"""
        obs, _ = env.reset(seed=seed, starting_points=starts, targets=targets, map_index=map_index)

        env.render()

        pso_start_time = time.time()

        for t in range(0, max_ep_len):
            start_time_one_step = time.time()

            actions = {}
            temp = pso_for_map.optimize()

            end_time_one_step = time.time()
            time_per_step.append(end_time_one_step-start_time_one_step)

            for i in range(len(env.agents)):
                agent = env.agents[i]
                actions[agent] = np.array([temp[2 * i], temp[2 * i + 1]])

            # print(actions)
            _, reward, done, _, _, path_dis = env.step(actions)
            total_path_dis += np.round(path_dis)
            ep_reward += reward

            env.render()

            if done:
                pso_end_time = time.time()
                pso_lasting_time = np.round(pso_end_time - pso_start_time, 2)
                time_store.append(pso_lasting_time)
                dis_store.append(total_path_dis)

                pso_success_ep_num += 1
                print("success")

                # pygame.image.save(env.surf, "./PSO_test_images/10_robots/map1/PSO_ep_{}_test_result.png".format(ep))
                # pygame.image.save(env.surf, "./PSO_test_images/10_robots/map2/PSO_ep_{}_test_result.png".format(ep))
                pygame.image.save(env.surf, "./PSO_test_images/10_robots/map3/PSO_ep_{}_test_result.png".format(ep))


                break
            elif all([env.collisions[agent] or env.terminations[agent] for agent in env.agents]):
                pso_end_time = time.time()
                pso_lasting_time = np.round(pso_end_time - pso_start_time, 2)
                print("failed")
                break
            else:
                pso_end_time = time.time()
                pso_lasting_time = np.round(pso_end_time - pso_start_time, 2)

        pso_test_running_reward += ep_reward
        print('Algorithm: {} \t\t Episode: {} \t\t Success: {} \t\t Reward: {} \t\t Lasting time: {} \t\t Total Path Distance: {}'.format("PSO", ep, done, round(ep_reward, 2), pso_lasting_time, np.round(total_path_dis, 2)))
        total_path_dis = 0
        ep_reward = 0

    if len(time_store) > 0:
        print("Max Time: {} \t\t Min Time: {} \t\t Mean Time: {} \t\t Max Dis: {} \t\t Min Dis: {} \t\t Mean: {}".format(
                max(time_store), min(time_store), np.mean(time_store), max(dis_store), min(dis_store),
                np.mean(dis_store)))
    # time_store = []
    # dis_store = []
    if len(time_per_step) > 0:
        print("Average time per step: {} \t Max: {} \t Min: {}".format(np.mean(time_per_step), max(time_per_step), min(time_per_step)))


    """CPSO testing"""
    for ep in range(0, 0):
        """调整初始位置和终点"""
        obs, _ = env.reset(seed=seed, starting_points=starts, targets=targets, map_index=map_index)

        env.render()

        cpso_start_time = time.time()

        for t in range(0, max_ep_len):
            start_time_one_step = time.time()

            actions = {}
            temp = cpso_for_map.optimize()

            end_time_one_step = time.time()
            time_per_step.append(end_time_one_step-start_time_one_step)

            for i in range(len(env.agents)):
                agent = env.agents[i]
                actions[agent] = np.array([temp[2 * i], temp[2 * i + 1]])

            # print(actions)
            _, reward, done, _, _, path_dis = env.step(actions)
            total_path_dis += np.round(path_dis)
            ep_reward += reward

            env.render()

            if done:
                cpso_end_time = time.time()
                cpso_lasting_time = np.round(cpso_end_time - cpso_start_time, 2)
                time_store.append(cpso_lasting_time)
                dis_store.append(total_path_dis)

                cpso_success_ep_num += 1
                print("success")

                pygame.image.save(env.surf, "./CPSO_test_images/10_robots/map2/CPSO_ep_{}_test_result.png".format(ep))
                
                break
            elif all([env.collisions[agent] or env.terminations[agent] for agent in env.agents]):
                cpso_end_time = time.time()
                cpso_lasting_time = np.round(cpso_end_time - cpso_start_time, 2)
                print("failed")
                break
            else:
                cpso_end_time = time.time()
                cpso_lasting_time = np.round(cpso_end_time - cpso_start_time, 2)

        pso_test_running_reward += ep_reward
        print('Algorithm: {} \t\t Episode: {} \t\t Success: {} \t\t Reward: {} \t\t Lasting time: {} \t\t Total Path Distance: {}'.format("CPSO", ep, done, round(ep_reward, 2), cpso_lasting_time, np.round(total_path_dis, 2)))
        total_path_dis = 0
        ep_reward = 0

    if len(time_store) > 0:
        print("Max Time: {} \t\t Min Time: {} \t\t Mean Time: {} \t\t Max Dis: {} \t\t Min Dis: {} \t\t Mean: {}".format(
                max(time_store), min(time_store), np.mean(time_store), max(dis_store), min(dis_store),
                np.mean(dis_store)))
    # time_store = []
    # dis_store = []
    if len(time_per_step) > 0:
        print("Average time per step: {} \t Max: {} \t Min: {}".format(np.mean(time_per_step), max(time_per_step), min(time_per_step)))


    """ABCL testing"""
    for ep in range(0, 0):
        """调整初始位置和终点"""
        obs, _ = env.reset(seed=seed, starting_points=starts, targets=targets, map_index=map_index)

        env.render()

        cpso_start_time = time.time()

        for t in range(0, max_ep_len):
            start_time_one_step = time.time()

            actions = {}
            temp = abcl_for_map.optimize()

            end_time_one_step = time.time()
            time_per_step.append(end_time_one_step-start_time_one_step)

            for i in range(len(env.agents)):
                agent = env.agents[i]
                actions[agent] = np.array([temp[2 * i], temp[2 * i + 1]])

            # print(actions)
            _, reward, done, _, _, path_dis = env.step(actions)
            total_path_dis += np.round(path_dis)
            ep_reward += reward

            env.render()

            if done:
                cpso_end_time = time.time()
                cpso_lasting_time = np.round(cpso_end_time - cpso_start_time, 2)
                time_store.append(cpso_lasting_time)
                dis_store.append(total_path_dis)

                abcl_success_ep_num += 1
                print("success")

                pygame.image.save(env.surf, "./ABCL_test_images/10_robots/map2/ABCL_ep_{}_test_result.png".format(ep))
                break
            elif all([env.collisions[agent] or env.terminations[agent] for agent in env.agents]):
                cpso_end_time = time.time()
                cpso_lasting_time = np.round(cpso_end_time - cpso_start_time, 2)
                print("failed")
                break

        pso_test_running_reward += ep_reward
        print('Algorithm: {} \t\t Episode: {} \t\t Success: {} \t\t Reward: {} \t\t Lasting time: {} \t\t Total Path Distance: {}'.format("ABCL", ep, done, round(ep_reward, 2), cpso_lasting_time, np.round(total_path_dis, 2)))
        total_path_dis = 0
        ep_reward = 0

    if len(time_store) > 0:
        print("Max Time: {} \t\t Min Time: {} \t\t Mean Time: {} \t\t Max Dis: {} \t\t Min Dis: {} \t\t Mean: {}".format(
                max(time_store), min(time_store), np.mean(time_store), max(dis_store), min(dis_store),
                np.mean(dis_store)))
    # time_store = []
    # dis_store = []
    if len(time_per_step) > 0:
        print("Average time per step: {} \t Max: {} \t Min: {}".format(np.mean(time_per_step), max(time_per_step), min(time_per_step)))


    env.close()
    # print("============================================================================================")
    #
    # ppo_abc_avg_test_reward = np.round(ppo_abc_test_running_reward / total_test_episodes, 1)
    # print("average test reward : " + str(ppo_abc_avg_test_reward))
    #
    # print("============================================================================================")

if __name__ == '__main__':

    test()

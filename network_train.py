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


def train():
    print("============================================================================================")

    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # 清理不必要进程
    torch.cuda.empty_cache()

    # 记录日志
    now = datetime.now()
    timestamp = now.strftime("%m%d-%H-%M")
    logging.basicConfig(level=logging.INFO,
                        filename='training_record_{}.log'.format(timestamp),
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
    logger = logging.getLogger(__name__)

    ####### initialize environment hyperparameters ######
    env_name = "Map"
    has_continuous_action_space = False  # 输出的动作为各个方案的概率

    max_ep_len = 200  # max timesteps in one episode
    max_training_timesteps = int(6e3)  # break training loop if timesteps > max_training_timesteps

    print_freq = max_ep_len * 6  # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
    save_model_freq = int(7.5e2)  # save model frequency (in num timesteps)

    action_std = 0.6  # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(3e2)  # action_std decay frequency (in num timesteps)
    lr_decay_freq = int(3e2)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = 100  # update policy every n timesteps
    # update_timestep = 2
    K_epochs = 80  # update policy for K epochs in one PPO update

    eps_clip = 0.2  # clip param for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network

    random_seed = 0  # set random seed if required (0 = no random seed)
    #####################################################

    print("training environment name: " + env_name)

    env = Map(agent_num=5, render_mode="rgb_array")
    seed = 42
    torch.manual_seed(seed=seed)

    # 用来画图
    episode_reward_storage = []
    episode_timestep_storage = []
    episodes = []
    timestep_reward_storage = []

    # 根据agent数量设置lb和ub
    lower_bound = [5, -math.pi / 4]
    upper_bound = [20, math.pi / 4]
    repeated_iter = cycle(lower_bound)
    fn_lb = list(islice(repeated_iter, len(lower_bound) * env.agent_num))

    repeated_iter = cycle(upper_bound)
    fn_ub = list(islice(repeated_iter, len(upper_bound) * env.agent_num))

    # artificial bee colony
    abc_for_map = ABC(60, 59, env.objective, fn_lb=fn_lb, fn_ub=fn_ub, env=env)

    success_ep_num = 0

    """不需要state_dim，所以这里没有写"""

    # action space dimension
    action_dim = 3  # 三个概率

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + "/PPO_" + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 0  # change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # directory = directory + '/' + env_name + '/' + 'RGB' + '/'
    # if not os.path.exists(directory):
    #     os.makedirs(directory)

    directory = directory + '/' + env_name + '/' + 'HSV' + '/' + timestamp
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path: " + checkpoint_path)
    #####################################################

    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################
    # initialize a PPO agent
    ppo_agent = PPO(action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT): ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode, timestep, reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:

        _, _ = env.reset(seed=seed)
        current_ep_reward = 0
        initial_rgb_array = env.render()  # 输出形状为（800， 800， 3）

        # 对输出进行预处理
        # initial_rgb_array = initial_rgb_array.astype(np.float32) / 255.0
        # initial_rgb_array = cv2.resize(initial_rgb_array, (200, 200), interpolation=cv2.INTER_LINEAR)
        initial_rgb_array_transposed = np.transpose(initial_rgb_array, (2, 0, 1))

        initial_rgb_array_transposed = np.expand_dims(initial_rgb_array_transposed, axis=0)

        rgb_array_transposed = torch.from_numpy(initial_rgb_array_transposed).float()

        for t in range(1, max_ep_len + 1):
            # adjust the probability of three strategies
            actions = {}
            probabilities, strategy_index, raw_action_probs = ppo_agent.select_strategy_profile(rgb_array_transposed)
            probabilities = [round(value, 3) for value in probabilities]

            abc_for_map.set_probability(probabilities)
            temp = abc_for_map.optimize()

            # 如果可行方案没有，就随机生成概率重新生成方案，并修改buffer里面的内容
            # if temp is None:
            #     while True:
            #         print("Last Out Was None")
            #         prob1 = random.random()
            #         prob1 = np.round(prob1, 2)
            #         prob2 = random.random()
            #         prob2 = np.round(prob2, 2)
            #         prob3 = random.random()
            #         prob3 = np.round(prob3, 2)
            #         total = prob1 + prob2 + prob3

            #         probabilities_new = np.round([prob1 / total, prob2 / total, prob3 / total], 2)
            #         print(probabilities_new)
            #         prob_tensor = torch.tensor(probabilities_new)
            #         dist = Categorical(prob_tensor)
            #         action_new = dist.sample()
            #         action_logprob_new = dist.log_prob(action_new)
            #         ppo_agent.buffer.actions[-1] = action_new.detach()
            #         ppo_agent.buffer.logprobs[-1] = action_logprob_new.detach()

            #         abc_for_map.set_probability(probabilities_new)
            #         temp = abc_for_map.optimize()
            #         print(temp)
            #         if temp is not None:
            #             break

            for i in range(len(env.agents)):
                agent = env.agents[i]
                actions[agent] = np.array([temp[2 * i], temp[2 * i + 1]])

            _, reward, done, _, _, _ = env.step(actions)
            timestep_reward_storage.append(reward)
            rgb_array = env.render()

            # 处理RGB array
            # rgb_array = rgb_array.astype(np.float32) / 255.0
            # rgb_array = cv2.resize(rgb_array, (200, 200), interpolation=cv2.INTER_LINEAR)
            rgb_array_transposed = np.transpose(rgb_array, (2, 0, 1))

            rgb_array_transposed = np.expand_dims(rgb_array_transposed, axis=0)

            rgb_array_transposed = torch.from_numpy(rgb_array_transposed).float()

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            print("timestep: {}".format(time_step))
            print("Rewards: {}".format(env.rewards))

            current_ep_reward += reward
            logger.info("Map Index: {}".format(env.map_index))
            logger.info("Probabilities: {}".format(probabilities))
            logger.info("Strategy Index: {}".format(strategy_index))
            logger.info("Raw Action Probabilities: {}".format(raw_action_probs))
            logger.info("Actions: {}".format(actions))
            logger.info("Destinations: {}".format(env.destinations))
            logger.info("Current Positions: {}".format(env.agent_positions))
            logger.info(
                "Current Episode: {} \t\t Steps For This Episode: {} \t\t Current Episode Reward : {}".format(i_episode,
                                                                                                              t,
                                                                                                              current_ep_reward))
            logger.info("Terminations: {}".format(env.terminations))
            logger.info("Collisions: {}".format(env.collisions))
            logger.info("\n")

            # update PPO agent
            if time_step % update_timestep == 0:
                print("Model Update")
                ppo_agent.update()
                print("Model Update Finished")

            if time_step % lr_decay_freq == 0:
                ppo_agent.decay_learning_rate(time_step, max_training_timesteps)

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                        print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                checkpoint_path = directory + "PPO_{}_{}.pth".format(env_name, time_step)
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break: if the episode is over
            if done:
                success_ep_num += 1
                print("Terminations: {}".format(env.terminations))
                print("Current Episode : {} \t\t Steps For This Episode: {} \t\t Current Episode Reward : {}".format(
                    i_episode, t, current_ep_reward))
                episode_reward_storage.append(current_ep_reward)
                episode_timestep_storage.append(t)
                episodes.append(i_episode + 1)
                break
            elif all([env.collisions[agent] or env.terminations[agent] for agent in env.agents]):
                # 所有的车都处于termination或collision

                print("Terminations: {}".format(env.terminations))
                print("Collisions: {}".format(env.collisions))
                print("Current Episode : {} \t\t Steps For This Episode: {} \t\t Current Episode Reward : {}".format(
                    i_episode, t, current_ep_reward))

                episode_reward_storage.append(current_ep_reward)
                episode_timestep_storage.append(t)
                episodes.append(i_episode + 1)
                break

            if t == max_ep_len:
                # 如果达到max_ep_len还没完成，输出相关结果
                print("Terminations: {}".format(env.terminations))
                print("Collisions: {}".format(env.collisions))
                print("Current Episode : {} \t\t Steps For This Episode: {} \t\t Current Episode Reward : {}".format(
                    i_episode, t, current_ep_reward))
                episode_reward_storage.append(current_ep_reward)
                episode_timestep_storage.append(t)
                episodes.append(i_episode + 1)
                break

        # if done or len(ppo_agent.buffer.rewards) >= 100:
        #     print("Model Update")
        #     ppo_agent.update()
        #     print("Model Update Finished")

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1
        # print("episode: {}".format(i_episode))

    log_f.close()
    env.close()

    success_rate = np.round(success_ep_num / i_episode, 2)

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("Total Episodes: {} \t Success Rate: {}".format(i_episode, success_rate))
    print("============================================================================================")

    # 设置线条宽度
    plt.rcParams['lines.linewidth'] = 1

    # 设置线条颜色
    plt.rcParams['lines.color'] = 'blue'

    # 设置线条样式
    plt.rcParams['lines.linestyle'] = '-'

    # episode-reward graph
    now = datetime.now()
    plt.plot(episodes, episode_reward_storage, label='Reward per Episode')
    plt.legend(loc="upper right")
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.savefig('./graph/episode_reward_{}.png'.format(now))
    # plt.show()

    # timestep-reward graph
    steps = list(range(1, time_step + 1))
    plt.plot(steps, timestep_reward_storage, label='Reward per Timestep')
    plt.legend(loc="upper right")
    plt.xlabel('Timesteps')
    plt.ylabel('Rewards')
    plt.savefig('./graph/timestep_reward_{}.png'.format(now))

    # episode-timestep graph
    # plt.plot(episodes, episode_timestep_storage, label='Timestep per Episode')
    # plt.legend(loc="upper right")
    # plt.xlabel('Episodes')
    # plt.ylabel('Timesteps')
    # plt.savefig('./graph/episode_timestep.png')


if __name__ == '__main__':
    train()


from swarm import ABC_origin
from map import Map
from datetime import datetime
from itertools import cycle, islice
import math
import numpy as np

def compare_test():
    print("=========================================================================================")

    max_ep_len = 200

    total_test_episodes = 5

    env = Map(agent_num=10, render_mode="human", test_result_save=True)

    lower_bound = [5, -math.pi / 4]
    upper_bound = [20, math.pi / 4]
    repeated_iter = cycle(lower_bound)
    fn_lb = list(islice(repeated_iter, len(lower_bound) * env.agent_num))

    repeated_iter = cycle(upper_bound)
    fn_ub = list(islice(repeated_iter, len(upper_bound) * env.agent_num))

    compare_abc = ABC_origin(30, 30, env.objective, fn_lb=fn_lb, fn_ub=fn_ub, env=env)

    seed = 42

    success_ep_num = 0

    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT): ", start_time)

    print("=========================================================================================")

    test_running_reward = 0
    for ep in range(0, total_test_episodes):
        ep_reward = 0

        obs, _ = env.reset(seed=seed)

        env.render()

        for t in range(0, max_ep_len):
            actions = {}
            temp = compare_abc.optimize()

            for i in range(len(env.agents)):
                agent = env.agents[i]
                actions[agent] = np.array([temp[2 * i], temp[2 * i + 1]])

            _, reward, done, _, _, _ = env.step(actions)
            ep_reward += reward

            env.render()

            if done:
                success_ep_num += 1
                print("success")
                break
            elif all([env.collisions[agent] or env.terminations[agent] for agent in env.agents]):
                print("failed")
                break

        test_running_reward += ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()
    print("========================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("========================================================================================")


if __name__ == '__main__':
    compare_test()



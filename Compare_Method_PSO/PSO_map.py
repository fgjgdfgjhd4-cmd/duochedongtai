import math
import random
import copy
import numpy as np

class PSO:
    def __init__(self, dimension, time, size, fn_lb, fn_ub, v_low, v_high, env):
        self.dimension = dimension
        self.time = time
        self.size = size
        self.fn_lb = np.array(fn_lb)
        self.fn_ub = np.array(fn_ub)

        self.v_low = v_low
        self.v_high = v_high
        self.x = np.zeros((self.size, self.dimension))  # 这个指代初始化时的随机产生的动作
        self.v = np.zeros((self.size, self.dimension))  # 这个指代动作变化的速率
        self.p_best = np.zeros((self.size, self.dimension)) # 记录单个例子遍历过的最佳动作
        self.g_best = np.zeros((1, self.dimension))     # 记录全局最佳动作
        self.env = env

    def optimize(self):
        self.initialize()
        self.final_best = np.zeros(self.dimension)
        for gen in range(self.time):
            self.update(self.size)
            if self.fitness(self.g_best) > self.fitness(self.final_best):
                self.final_best = self.g_best.copy()


        actions = self.final_best
        positions = copy.deepcopy(self.env.agent_positions)
        for i in range(len(self.env.possible_agents)):
            agent = self.env.possible_agents[i]
            if self.env.terminations[agent] is True or self.env.collisions[agent] is True:
                continue

            orientation_new = self.env.orientation[agent] + actions[2 * i + 1]
            positions[agent][0] += np.cos(orientation_new) * actions[2 * i]
            positions[agent][1] += np.sin(orientation_new) * actions[2 * i]

            if any(positions[agent] <= 0 + self.env.safe_distance / 2) or \
                any(positions[agent] >= self.env.screen_width - self.env.safe_distance / 2):
                self.final_best[2 * i] = -1
                self.final_best[2 * i + 1] = np.random.uniform(-math.pi / 4, 0)
                self.env.obj_ratio[agent] += 0.1
                continue

            for obs_idx in range(len(self.env.obstacle_centers)):

                """由于把机器人看成圆，探测碰撞的范围会变大，因此在图中车和障碍物可能不会碰撞"""
                if np.linalg.norm(self.env.obstacle_centers[obs_idx] - positions[agent]) <= self.env.radius[obs_idx] + \
                        self.env.safe_distance / 2:
                    self.final_best[2 * i] = -1
                    self.final_best[2 * i + 1] = np.random.uniform(-math.pi/4, 0)
                    self.env.obj_ratio[agent] += 0.1
                    break

            for obs_idx in range(len(self.env.rec_center)):
                current_obs_center = self.env.rec_center[obs_idx]
                current_obs_size = self.env.rec_size[obs_idx]

                if abs(positions[agent][0] - current_obs_center[0]) <= current_obs_size[0] / 2 + self.env.safe_distance / 2 and \
                   abs(positions[agent][1] - current_obs_center[1]) <= current_obs_size[1] / 2 + self.env.safe_distance / 2:
                    self.final_best[2 * i] = -1
                    self.final_best[2 * i + 1] = np.random.uniform(-math.pi/4, 0)
                    self.env.obj_ratio[agent] += 0.1
                    break

        # 希望等所有的位置都更新完后再判断车之间是否撞
        for i in range(len(self.env.agents)):
            agent = self.env.agents[i]
            for j in range(i + 1, len(self.env.agents)):
                other_agent = self.env.agents[j]
                if np.linalg.norm(positions[agent] - positions[other_agent]) < self.env.safe_distance / 2:
                    self.final_best[2 * i] = -1
                    self.final_best[2 * i + 1] = np.random.uniform(-math.pi/4, 0)
                    self.env.obj_ratio[agent] += 0.1

                    self.final_best[2 * i] = -1
                    self.final_best[2 * i + 1] = np.random.uniform(-math.pi/4, 0)
                    self.env.obj_ratio[other_agent] += 0.1

        return self.final_best

    def initialize(self):
        temp = -1000000
        for i in range(self.size):
            for j in range(self.dimension):
                self.x[i][j] = random.uniform(self.fn_lb[j], self.fn_ub[j])
                self.v[i][j] = random.uniform(self.v_low, self.v_high)
            self.p_best[i] = self.x[i]
            fit = self.fitness(self.p_best[i])
            if fit > temp:
                self.g_best = self.p_best[i]
                temp = fit

    def fitness(self, particle):
        result = self.env.objective(particle)

        if result >= 0:
            fitness = 1 / (1 + result)
        else:
            fitness = 1 + abs(result)

        return fitness

    def update(self, size):
        c1 = 2.0    # 学习因子
        c2 = 2.0
        w = 0.8
        for i in range(size):
            # 更新速度
            self.v[i] = w * self.v[i] + c1 * random.uniform(0, 1) * (self.p_best[i] - self.x[i]) + \
                        c2 * random.uniform(0, 1) * (self.g_best - self.x[i])

            # 速度限制
            for j in range(self.dimension):
                self.v[i][j] = min(self.v_high, max(self.v_low, self.v[i][j]))

                # 更新位置以及位置限制
                self.x[i][j] += self.v[i][j]
                self.x[i][j] = min(self.fn_ub[j], max(self.x[i][j], self.fn_lb[j]))
                self.x[i][j] = np.round(self.x[i][j], 2)

            # 更新p_best和g_best
            if self.fitness(self.x[i]) > self.fitness(self.p_best[i]):
                self.p_best[i] = self.x[i]
            if self.fitness(self.x[i]) > self.fitness(self.g_best):
                self.g_best = self.x[i]


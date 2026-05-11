import math
import random
import copy
import numpy as np
from experiment_utils import repair_candidate_solution

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

        self.final_best = repair_candidate_solution(self.final_best, self.env)
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


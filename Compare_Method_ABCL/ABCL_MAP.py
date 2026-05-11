import copy
import random
import math
import numpy as np
import pprint as pp
import random as rand
from operator import attrgetter
from Compare_Method_ABCL.food_source import FoodSource
from experiment_utils import repair_candidate_solution

"""
注意这里的fn_lb和fn_ub只有两个维度，没有扩展到2*num_robots的维度数
"""

class ABCL(object):

    food_source = []

    def __init__(self, npopulation, nruns, fn_eval, *, trials_limit=50,
                 employed_bees_percentage=0.5, fn_lb, fn_ub, env):
        super(ABCL, self).__init__()
        self.npopulation = npopulation
        self.nruns = nruns
        self.fn_eval = fn_eval
        self.trials_limit = trials_limit
        self.fn_lb = np.array(fn_lb)
        self.fn_ub = np.array(fn_ub)

        self.employed_bees = round(npopulation * employed_bees_percentage)
        self.onlooker_bees = npopulation - self.employed_bees

        self.FES = 0
        self.FESmax = 3000

        self.env = env

    def optimize(self):
        self.initialize()

        for n_run in range(1, self.nruns + 1):
            self.employed_bees_stage()
            self.onlooker_bees_stage()
            self.scout_bees_stage()

        best_fs = self.best_source()

        self.FES += self.npopulation

        return np.round(best_fs.solution, 2)

    def initialize(self):
        self.food_sources = [self.create_foodsource() for i in range(self.employed_bees)]

    def employed_bees_stage(self):
        for i in range(self.employed_bees):
            food_source = self.food_sources[i]
            new_solution = self.generate_solution_for_employer(i)

            best_solution = self.best_solution(food_source.solution, new_solution)
            self.set_solution(food_source, best_solution)

    def onlooker_bees_stage(self):
        for i in range(self.employed_bees):
            food_source = self.food_sources[i]

            new_solution = self.generate_solution_for_onlooker(i)

            best_solution = self.best_solution(food_source.solution, new_solution)
            self.set_solution(food_source, best_solution)

    def scout_bees_stage(self):
        for i in range(self.employed_bees):
            food_source = self.food_sources[i]

            if food_source.trials > self.trials_limit:
                new_solution = self.generate_solution_for_scout(i)
                self.set_solution(food_source, new_solution)
                food_source.fitness = self.fitness_for_abc(food_source.solution)
                self.FES += 1
                food_source.trials = 0

    """以下为主要阶段的辅助函数"""

    def generate_solution_for_employer(self, index):
        phi_1 = round(np.random.uniform(-1, 1), 2)
        phi_2 = round(np.random.uniform(-1, 1), 2)

        idx_1 = self.random_solution_excluding([index])
        idx_2 = self.random_solution_excluding([index, idx_1])

        j = random.randint(0, 2 * self.env.agent_num - 1)

        gbest_idx = self.global_best_solution()

        new_solution = copy.deepcopy(self.food_sources[index].solution)
        new_solution[j] += phi_1 * (self.food_sources[gbest_idx].solution[j] - new_solution[j]) + \
                            phi_2 * (self.food_sources[idx_1].solution[j] - self.food_sources[idx_2].solution[j])

        return new_solution

    def generate_solution_for_onlooker(self, index):
        k = self.random_solution_excluding([index])

        j = random.randint(0, 2 * self.env.agent_num - 1)
        new_solution = copy.deepcopy(self.food_sources[index].solution)
        if self.food_sources[index].fitness > self.food_sources[k].fitness:
            new_solution[j] += random.uniform(0, 1) * (self.food_sources[index].solution[j] - \
                                                       self.food_sources[k].solution[j])
        else:
            new_solution[j] += random.uniform(0, 1) * (self.food_sources[k].solution[j] - \
                                                       self.food_sources[index].solution[j])

        return new_solution

    def generate_solution_for_scout(self, index):
        phi = round(np.random.uniform(-1, 1), 2)
        gbest_idx = self.global_best_solution()

        new_solution = copy.deepcopy(self.food_sources[index].solution)
        j = random.randint(0, 2 * self.env.agent_num - 1)

        new_solution[j] = self.food_sources[gbest_idx].solution[j] + phi * (self.fn_ub[j % 2] - \
                                                                            self.fn_lb[j % 2])

        return new_solution


    def create_foodsource(self):
        solution = self.candidate_solution()
        fitness = self.fitness_for_abc(solution)

        return FoodSource(solution, fitness)

    def candidate_solution(self):
        solution = []
        for _ in range(self.env.agent_num):
            solution.append(np.round(random.uniform(self.fn_lb[0], self.fn_ub[0])))
            solution.append(np.round(random.uniform(self.fn_lb[1], self.fn_ub[1])))
        return solution

    def fitness_for_abc(self, solution):
        result = self.env.objective(solution)

        if result >= 0:
            fitness = 1 / (1 + result)
        else:
            fitness = 1 + abs(result)

        return fitness

    def random_solution_excluding(self, excluded_index):
        available_indexes = set(range(self.employed_bees))
        exclude_set = set(excluded_index)
        diff = available_indexes - exclude_set
        selected = random.choice(list(diff))

        return selected

    def global_best_solution(self):
        return self.food_sources.index(max(self.food_sources, key=lambda x: x.fitness))

    def best_solution(self, current_solution, new_solution):
        if self.env.objective(new_solution) < self.env.objective(current_solution):
            return new_solution
        else:
            return current_solution

    def set_solution(self, food_source, new_solution):
        if np.array_equal(new_solution, food_source.solution):
            food_source.trials += 1
        else:
            food_source.solution = new_solution
            food_source.trials = 0

    def best_source(self):

        best = max(self.food_sources, key=attrgetter('fitness'))

        best.solution = repair_candidate_solution(best.solution, self.env)
        return best

        # 判断最佳方案里面是否会有导致车碰撞的动作，如果有，则将这个车的动作变成1


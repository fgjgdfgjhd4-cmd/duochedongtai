"""
ABC FOR MAP
"""
import numpy as np
from original_artificial_bee_colony.food_source import FoodSource

from map import Map
import math
import random
from operator import attrgetter
import copy
from itertools import cycle, islice
from experiment_utils import repair_candidate_solution


class ABC(object):
    """docstring for ABC"""

    food_sources = []

    def __init__(self, npopulation, nruns, fn_eval, *, trials_limit=50,
                 employed_bees_percentage=0.5, fn_lb, fn_ub, env):
        super(ABC, self).__init__()
        self.npopulation = npopulation
        self.nruns = nruns
        self.fn_eval = fn_eval
        self.trials_limit = trials_limit
        self.fn_lb = np.array(fn_lb)
        self.fn_ub = np.array(fn_ub)

        self.employed_bees = round(npopulation * employed_bees_percentage)
        self.onlooker_bees = npopulation - self.employed_bees

        # 这个参数是和选择三种不同的onlooker策略有关
        """
        在人工蜂群算法（Artificial Bee Colony Algorithm, ABC）中，FES 通常代表**“Function Evaluations”，即函数评估次数**。这个参数用于衡量算法在整个优化过程中的评估次数，它常常被用作算法的终止条件或性能指标之一。

        FES 在 ABC 算法中的作用
        终止条件：在ABC算法中，FES的最大值通常会被设置为一个固定的数值（如1,000或10,000次评估）。一旦算法运行达到这个最大FES，优化过程将会终止，从而控制算法的总运行时间和计算复杂度。
        
        性能评估指标：在实验中，FES被用来衡量算法在不同阶段的表现。例如，通过记录ABC算法达到某个目标值的FES，可以评估算法的收敛速度。
        
        资源消耗控制：在计算资源受限的情况下，FES能够帮助控制算法的资源消耗。设置一个合理的最大FES值，可以避免算法在复杂问题上运行过久。
        
        在ABC算法的迭代过程中，每次进行适应度计算时，都会增加FES的计数值。这样，FES可以很直观地反映出算法在优化过程中的资源使用情况。
        """
        self.FES = 0
        self.FESmax = 3000

        self.env = env

        """如果网络输出的是策略编号"""
        self.strategy_index = 0

        # 概率分界点
        self.p_1 = 1.0 / 3.0
        self.p_2 = 1.0 / 3.0

    def set_strategy_index(self, index):
        """人工设置选取的策略编号"""
        self.strategy_index = index

    def set_probability(self, probabilities):
        """根据probabilities设置三个策略的初始概率"""
        probabilities = np.array(probabilities, dtype=float)
        total = float(np.sum(probabilities))
        if total <= 0:
            probabilities = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
        else:
            probabilities = probabilities / total
        self.p_1 = float(probabilities[0])
        self.p_2 = float(probabilities[1])
        self.p_3 = float(probabilities[2])

    def optimize(self):

        self.initialize()
        self.strategy_counts = [0, 0, 0]
        # pp.pprint(self.food_sources)

        for n_run in range(1, self.nruns + 1):
            self.employed_bees_stage()
            self.onlooker_bees_stage()
            self.scout_bees_stage()

        # pp.pprint(self.food_sources)
        best_fs = self.best_source()

        # if best_fs == None:
        #     return None
        # else:
        self.FES += self.npopulation

        return np.round(best_fs.solution, 2)

    def initialize(self):
        """产生初始解"""
        self.food_sources = [self.create_foodsource() for i in range(self.employed_bees)]

    def employed_bees_stage(self):
        """雇佣蜜蜂阶段"""
        for i in range(self.employed_bees):
            food_source = self.food_sources[i]
            new_solution = self.generate_solution_for_employer()
            """比较初始解和新生成的解哪个好"""
            best_solution = self.best_solution(food_source.solution, new_solution)

            self.set_solution(food_source, best_solution)

    def onlooker_bees_stage(self):
        for i in range(self.onlooker_bees):
            probabilities = [self.probability(fs) for fs in self.food_sources]
            selected_index = self.selection(range(len(self.food_sources)), probabilities)
            selected_source = self.food_sources[selected_index]

            # 生成随机数用于确定采取哪种策略
            prob_for_onlooker = round(random.uniform(0, 1), 2)

            new_solution = self.generate_solution_for_onlooker(selected_index, prob_for_onlooker)
            best_solution = self.best_solution(selected_source.solution, new_solution)

            self.set_solution(selected_source, best_solution)

    def scout_bees_stage(self):
        for i in range(self.employed_bees):
            food_source = self.food_sources[i]

            if food_source.trials > self.trials_limit:
                self.food_sources[i] = self.create_foodsource()
                self.FES += 1

    def global_best_solution(self):
        fitnesses = [fs.fitness for fs in self.food_sources]
        max_fitness_idx = fitnesses.index(max(fitnesses))
        return self.food_sources[max_fitness_idx].solution

    def neighbor_best_solution(self, current_solution_index, d):
        """因为车数量增加后，维度太大，因此考虑在单个维度上找邻居最优"""
        neighbors = []
        for idx in range(len(self.food_sources)):
            if idx != current_solution_index:
                if abs(self.food_sources[idx].solution[d] - self.food_sources[current_solution_index].solution[d]) <= \
                        self.fn_ub[d] / 4:
                    neighbors.append(self.food_sources[idx])

        """如果没有邻居。则返回current solution"""
        if len(neighbors) == 0:
            return self.food_sources[current_solution_index].solution
        else:
            fitnesses = [fs.fitness for fs in neighbors]
            max_fitness_idx = fitnesses.index(max(fitnesses))
            return self.food_sources[max_fitness_idx].solution

    def generate_solution_for_employer(self):
        phi_1 = round(np.random.uniform(-1, 1), 2)
        phi_2 = round(np.random.uniform(-1, 1), 2)

        idx_1 = random.randint(0, len(self.food_sources) - 1)
        idx_2 = self.random_solution_excluding([idx_1])
        idx_3 = self.random_solution_excluding([idx_1, idx_2])

        new_solution = self.food_sources[idx_1].solution + phi_1 * (self.food_sources[idx_2].solution - \
                                                                    self.food_sources[idx_1].solution) + phi_2 * (
                                   self.food_sources[idx_3].solution - \
                                   self.food_sources[idx_1].solution)

        new_solution = np.round(new_solution, 2)

        for i in range(len(self.env.agents)):
            agent = self.env.agents[i]
            if self.env.terminations[agent] is True or self.env.collisions[agent] is True:
                new_solution[i * 2] = 0
                new_solution[i * 2 + 1] = 0
            else:
                new_solution[i * 2] = max(min(self.fn_ub[i * 2], new_solution[i * 2]), self.fn_lb[i * 2])
                new_solution[i * 2 + 1] = max(min(self.fn_ub[i * 2 + 1], new_solution[i * 2 + 1]),
                                              self.fn_lb[i * 2 + 1])

        return new_solution

    def generate_solution_for_onlooker(self, current_solution_index, prob_for_onlooker):
        solution = self.food_sources[current_solution_index].solution

        """这是原来的ABC的策略计算方法"""
        # k_source_index = self.random_solution_excluding([current_solution_index])
        # k_solution = self.food_sources[k_source_index].solution

        random_dimensions = []
        for i in range(len(self.env.agents)):
            agent = self.env.agents[i]
            if self.env.terminations[agent] is True or self.env.collisions[agent] is True:
                continue

            random_dimensions.append(random.randint(2 * i, 2 * i + 1))

        r = random.uniform(-1, 1)

        """这是原来的ABC的策略计算方法"""
        # new_solution[d] = solution[d] + r * (solution[d] - k_solution[d])

        new_solution = np.copy(solution)

        """根据概率选择"""
        if prob_for_onlooker <= self.p_1:
            self.strategy_counts[0] += 1
            # spiral approximation strategy
            global_best_solution = self.global_best_solution()
            b = 1 - self.FES / self.FESmax
            r = round(np.random.uniform(-1, 1), 2)

            for d in random_dimensions:
                new_solution[d] = global_best_solution[d] + abs(global_best_solution[d] - new_solution[d]) * \
                                  np.exp(b * r) * np.cos(2 * math.pi * r)
                new_solution[d] = np.round(new_solution[d], 2)


        elif prob_for_onlooker <= self.p_1 + self.p_2:
            self.strategy_counts[1] += 1
            # global and neighborhood best guide strategy
            global_best_solution = self.global_best_solution()

            r_1 = self.random_solution_excluding([current_solution_index])
            random_solution = self.food_sources[r_1].solution
            phi_1 = round(np.random.uniform(-1, 1), 2)
            phi_2 = round(np.random.uniform(-1, 1), 2)
            obj_current = self.env.objective(solution)
            obj_random = self.env.objective(random_solution)

            c_1 = (obj_current - obj_random) / (obj_current + obj_random + 0.001)  # 防止除以0
            c_2 = 2 * self.food_sources[current_solution_index].trials / self.trials_limit
            c_3 = 2 * (1 - self.food_sources[current_solution_index].trials / self.trials_limit)

            for d in random_dimensions:
                neighbor_best_solution = self.neighbor_best_solution(current_solution_index, d)

                new_solution[d] = solution[d] + c_1 * phi_1 * (random_solution[d] - solution[d]) + c_2 * phi_2 * \
                                  (global_best_solution[d] - solution[d]) + c_3 * (
                                              neighbor_best_solution[d] - solution[d])

                new_solution[d] = np.round(new_solution[d], 2)

        else:
            self.strategy_counts[2] += 1
            F = round(np.random.uniform(-1, 1), 2)
            idx_1 = self.random_solution_excluding([current_solution_index])
            idx_2 = self.random_solution_excluding([current_solution_index, idx_1])
            for d in random_dimensions:
                new_solution[d] = solution[d] + F * (self.food_sources[idx_1].solution[d] - \
                                                     self.food_sources[idx_2].solution[d])

                new_solution[d] = np.round(new_solution[d], 2)

        # 统一限制在上下限范围内
        for i in range(len(new_solution)):
            new_solution[i] = max(min(self.fn_ub[i], new_solution[i]), self.fn_lb[i])
        return np.round(new_solution, decimals=2)

        """这是原来的ABC的策略计算方法"""
        # # 限制在上下限范围内
        # new_solution[d] = max(min(self.fn_ub[d], new_solution[d]), self.fn_lb[d])
        # return np.around(new_solution, decimals=4)

    def create_foodsource(self):
        solution = self.candidate_solution(self.fn_lb, self.fn_ub)
        fitness = self.fitness_for_abc(solution)

        return FoodSource(solution, fitness)

    def candidate_solution(self, lb, ub):
        r = round(random.uniform(0, 1), 2)
        solution = lb + (ub - lb) * r

        return np.round(solution, decimals=2)

    def random_solution_excluding(self, excluded_index):
        available_indexes = set(range(self.employed_bees))
        exclude_set = set(excluded_index)
        diff = available_indexes - exclude_set
        selected = random.choice(list(diff))

        return selected

    def best_solution(self, current_solution, new_solution):
        if self.env.objective(new_solution) < self.env.objective(current_solution):
            return new_solution
        else:
            return current_solution

    def probability(self, food_source):
        fitness_sum = sum([fs.fitness for fs in self.food_sources])
        probability = food_source.fitness / fitness_sum

        return probability

    def fitness_for_abc(self, solution):
        result = self.env.objective(solution)

        if result >= 0:
            fitness = 1 / (1 + result)
        else:
            fitness = 1 + abs(result)

        return fitness

    def selection(self, solutions, weights):
        return random.choices(solutions, weights)[0]

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



def main():
    from CNN_PPO import PPO
    import torch

    env = Map(agent_num=10, render_mode="human")
    lower_bound = [200, -math.pi / 2]
    upper_bound = [400, math.pi / 2]
    repeated_iter = cycle(lower_bound)
    fn_lb = list(islice(repeated_iter, len(lower_bound) * env.agent_num))
    repeated_iter = cycle(upper_bound)
    fn_ub = list(islice(repeated_iter, len(upper_bound) * env.agent_num))

    abc_for_map = ABC(30, 29, env.objective, fn_lb=fn_lb, fn_ub=fn_ub, env=env)
    ppo_agent = PPO(3, 0.0003, 0.001, 0.99, 80, 0.2, False, 0.6)
    # cnn = CNN()
    # ppo_agent = PPO(10, 3, 0.0003, 0.001, 0.99, 80, 0.2, False, 0.6)
    # env.set_start_and_end("agent_0", 200, 370, 200, 30)

    _, _ = env.reset()
    initial_rgb_array = env.render()
    initial_rgb_array_transposed = np.transpose(initial_rgb_array, (2, 0, 1))
    initial_rgb_array_transposed = np.expand_dims(initial_rgb_array_transposed, axis=0)

    rgb_array_transposed = torch.from_numpy(initial_rgb_array_transposed).float()
    step = 0

    while abc_for_map.FES < abc_for_map.FESmax:
        actions = {}
        temp = None

        # prob1 = random.random()
        # prob1 = np.round(prob1, 2)
        # prob2 = random.random()
        # prob2 = np.round(prob2, 2)
        # prob3 = random.random()
        # prob3 = np.round(prob3, 2)
        # total = prob1 + prob2 + prob3
        #
        # probabilities = np.round([prob1 / total, prob2 / total, prob3 / total], 2)
        #
        # abc_for_map.set_probability(probabilities)

        """这里应该是根据网络输出更新策略概率"""

        temp = abc_for_map.optimize()
        # print(len(temp))
        for i in range(int(len(temp) / 2)):
            agent = env.agents[i]
            actions[agent] = np.array([temp[2 * i], temp[2 * i + 1]])

        _, reward, done, _, _, _ = env.step(actions)
        env.render()

        if done:
            print("Terminations: {}".format(env.terminations))
        elif any([env.collisions[agent] for agent in env.agents]):
            print("Collisions: {}".format(env.collisions))

    # while step < 150:

    #     actions = {}
    #     probabilities = ppo_agent.select_action(rgb_array_transposed)
    #     probabilities = probabilities.cpu().numpy()
    #     probabilities = [round(value, 2) for value in probabilities.flat]

    #     abc_for_map.set_probability(probabilities)

    #     """这里应该是根据网络输出更新策略概率"""

    #     temp = abc_for_map.optimize()
    #     # print(len(temp))
    #     for i in range(int(len(temp) / 2)):
    #         agent = env.agents[i]
    #         actions[agent] = np.array([temp[2*i], temp[2*i+1]])

    #     # print(actions)
    #     print("Probabilities: {}".format(probabilities))
    #     _, reward, done, _, _, _ = env.step(actions)
    #     rgb_array = env.render()
    #     rgb_array_transposed = np.transpose(rgb_array, (2, 0, 1))

    #     rgb_array_transposed = np.expand_dims(rgb_array_transposed, axis=0)

    #     rgb_array_transposed = torch.from_numpy(rgb_array_transposed).float()

    #     if done:
    #         print("Terminations: {}".format(env.terminations))
    #     elif any([env.collisions[agent] for agent in env.agents]):
    #         print("Collisions: {}".format(env.collisions))


if __name__ == "__main__":
    main()



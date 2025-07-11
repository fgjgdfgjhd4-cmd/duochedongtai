import copy
import random
import math
import numpy as np
from Genome import Genome

class GA:
    def __init__(self, n_population, genome_length, mutation_rate, crossover_rate,
                 generations, fn_lb, fn_ub, env):
        self.n_population = n_population
        self.genome_length = genome_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations

        self.fn_lb = fn_lb
        self.fn_ub = fn_ub
        self.env = env

        self.population = []

    def optimize(self):
        self.generate_population()

        population = []
        for _ in self.generations:
            # 选择下一代
            for i in range(self.n_population // 2):
                idx1 = self.select_parent()
                idx2 = self.select_parent()
                offspring1, offspring2 = self.crossover(idx1, idx2)
                self.population[i].solution = offspring1
                self.population[i].fitness = self.fitness(offspring1)
                self.population[i+1].solution = offspring2
                self.population[i+1].fitness = self.fitness(offspring2)

        best_idx = self.population.index(max(self.population, key=lambda x: x.fitness))

        # 返回最优解
        return self.population[best_idx].solution

    def generate_population(self):
        for _ in range(self.n_population):
            temp = []

            # 生成初始解
            for i in range(self.env.agent_num):
                temp.append(random.uniform(self.fn_lb[0], self.fn_ub[0]))
                temp.append(random.uniform(self.fn_lb[1], self.fn_ub[1]))

            solution = np.array(temp)

            # 计算适应度
            fitness = self.fitness(solution)

            # 这两个被封装成一个对象
            self.population.append(Genome(solution, fitness))

    def fitness(self, solution):
        result = self.env.objective(solution)

        if result >= 0:
            fitness = 1 / (1 + result)
        else:
            fitness = 1 + abs(result)

        return fitness

    # 选择父代（轮盘赌）
    def select_parent(self):
        total_fitness = sum([self.population[i].fitness for i in range(self.n_population)])
        pick = random.uniform(0, total_fitness)
        current = 0
        for i in range(self.n_population):
            current += self.population[i].fitness
            # 获取父类的索引
            if current > pick:
                return i

    # 交叉选择（单点交叉）
    def crossover(self, idx1, idx2):
        parent1, parent2 = self.population[idx1], self.population[idx2]
        if random.random() < self.crossover_rate:
            point = random.randint(1, self.genome_length - 1)
            return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
        else:
            return parent1, parent2

    # 变异操作
    def mutate(self, solution):
        for i in range(len(solution)):
            if random.random() < self.mutation_rate:
                solution[i] = random.uniform(self.fn_lb[i % 2], self.fn_ub[i % 2])
        return solution



import copy
import random
import math
import numpy as np
import pprint as pp
import random as rand
from operator import attrgetter
from Compare_Method_ABC.food_source import FoodSource
from experiment_utils import repair_candidate_solution


class ABC_origin(object):

    food_source = []

    def __init__(self, n_population, n_runs, fn_eval, *, trials_limit=50, employed_bees_percentage=0.5,
                 fn_lb=[-5, -5], fn_ub=[5,5], env):
        super(ABC_origin, self).__init__()
        self.n_population = n_population
        self.n_runs = n_runs
        self.fn_eval = fn_eval
        self.trials_limit = trials_limit
        self.fn_lb = np.array(fn_lb)
        self.fn_ub = np.array(fn_ub)

        self.employed_bees = round(n_population * employed_bees_percentage)
        self.onlooker_bees = n_population - self.employed_bees

        self.env = env

    def optimize(self):
        self.initialize()

        for i in range(0, self.n_runs):
            self.employed_bees_stage()
            self.onlooker_bees_stage()
            self.scout_bees_stage()

        best_fs = self.best_source()

        return np.round(best_fs.solution, 2)

    def initialize(self):
        self.food_sources = [self.create_foodsource() for i in range(self.employed_bees)]

    def employed_bees_stage(self):
        for i in range(self.employed_bees):
            food_source = self.food_sources[i]
            new_solution = self.generate_solution(i)
            best_solution = self.best_solution(food_source.solution, new_solution)

            self.set_solution(food_source, best_solution)

    def onlooker_bees_stage(self):
        for i in range(self.onlooker_bees):
            probabilities = [self.probability(fs) for fs in self.food_sources]
            selected_index = self.selection(range(len(self.food_sources)), probabilities)
            selected_source = self.food_sources[selected_index]
            new_solution = self.generate_solution(selected_index)
            best_solution = self.best_solution(selected_source.solution, new_solution)

            self.set_solution(selected_source, best_solution)

    def scout_bees_stage(self):
        for i in range(self.employed_bees):
            food_source = self.food_sources[i]

            if food_source.trials > self.trials_limit:
                self.food_sources[i] = self.create_foodsource()

    def generate_solution(self, current_solution_index):
        solution = self.food_sources[current_solution_index].solution
        k_source_index = self.random_solution_excluding([current_solution_index])
        k_solution = self.food_sources[k_source_index].solution
        d = rand.randint(0, len(self.fn_lb) - 1)
        r = rand.uniform(-1, 1)

        new_solution = np.copy(solution)
        new_solution[d] = solution[d] + r * (solution[d] - k_solution[d])

        new_solution[d] = min(self.fn_ub[d], max(self.fn_lb[d], new_solution[d]))

        return np.around(new_solution, decimals=4)

    def create_foodsource(self):
        solution = self.candidate_solution(self.fn_lb, self.fn_ub)
        fitness = self.fitness_for_abc(solution)

        return FoodSource(solution, fitness)

    def candidate_solution(self, lb, ub):
        #
        r = round(random.uniform(0, 1), 2)
        solution = lb + (ub - lb) * r
        for j in range(len(solution)):
            solution[j] = min(self.fn_ub[j], max(self.fn_lb[j], solution[j]))

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

    def fitness_for_abc(self, solution):
        result = self.env.objective(solution)

        if result >= 0:
            fitness = 1 / (1 + result)
        else:
            fitness = 1 + abs(result)

        return fitness

    def selection(self, solutions, weights):
        return rand.choices(solutions, weights)[0]

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



    def probability(self, solution_fitness):
        fitness_sum = sum([fs.fitness for fs in self.food_sources])
        probability = solution_fitness.fitness / fitness_sum

        return probability




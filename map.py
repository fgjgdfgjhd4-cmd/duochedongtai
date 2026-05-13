import functools
import random
from datetime import datetime
try:
    import cv2
except ImportError:
    cv2 = None
import gym
import numpy as np
import copy
from gym.error import DependencyNotInstalled
import math
from gymnasium.spaces import Box
from original_artificial_bee_colony.lib import target_toward_distance, Repulsive, intersect_ray_with_rectangle

try:
    from pettingzoo import ParallelEnv
except ImportError:
    class ParallelEnv:
        pass


class Map(ParallelEnv):
    """The metadata holds environment constants.

    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        "name": "test_map",
        "render_modes": ["human"],
        "render_fps": 30,
    }

    def __init__(self, agent_num=2, render_mode=None, test_result_save=False):
        self.agent_num = agent_num
        self.possible_agents = ["agent_" + str(r) for r in range(self.agent_num)]

        self.screen_width = 800
        self.screen_height = 800
        self.screen = None
        self.clock = None
        self.isopen = None
        self.terminations = {}
        self.truncations = {}
        self.test_result_save = test_result_save

        # self.agent_name_mapping = dict(
        #     zip(self.possible_agents, list(range(len(self.possible_agents))))
        # )

        self.timestep = 0
        self.render_mode = render_mode

        # 在goal range内即可视为到达终点
        self.goal_range = 5

        self.map_index = None

        # 速度和角度
        self._action_spaces = {
            agent: Box(low=np.array([0.0, -math.pi / 2]), high=np.array([5.0, math.pi / 2]), dtype=np.float32)
            for agent in self.possible_agents
        }

        # 坐标和朝向角
        self._observation_space = {
            agent: Box(low=np.array([5.0, 5.0, -math.pi / 2]), high=np.array([395.0, 395.0, math.pi / 2]),
                       dtype=np.float32)
            for agent in self.possible_agents
        }

        self.agent_positions = {agent: np.zeros(2) for agent in self.possible_agents}
        self.destinations = {agent: np.zeros(2) for agent in self.possible_agents}
        self.orientation = {agent: 0 for agent in self.possible_agents}
        self.collisions = {agent: False for agent in self.possible_agents}

        """在reset里面确定用哪一种地图"""
        self.obstacle_centers = []

        self.radius = []
        # self.obstacle_sizes = [np.array([200, 50]), np.array([200, 50]), np.array([50, 120]), np.array([50, 120])]

        # 机器人视为边长为20的机器人，但计算损失的时候视为10*sqrt(2)的圆
        self.vehicle_size = 10

        # safe_distance是机器人对角线长度2倍
        self.safe_distance = self.vehicle_size * 2 * np.sqrt(2)

        # self.agent_colors = {agent: np.random.randint(low=20, high=230, size=3) for agent in self.possible_agents}

        """默认五个车, 0：紫罗兰，1：皇家蓝，2：海洋绿，3：深卡其布，4：棕色, 5: 青绿色，6：暖灰色, 7: 巧克力色, 8: 淡紫色，9：沙棕色"""
        # self.agent_colors = {"agent_0": np.array([199, 21, 133]), "agent_1": np.array([65, 105, 225]),
        #                      "agent_2": np.array([46, 139, 87]), "agent_3": np.array([189, 183, 107]),
        #                      "agent_4": np.array([165, 42, 42])}

        self.agent_colors = {"agent_0": np.array([199, 21, 133]), "agent_1": np.array([65, 105, 225]),
                             "agent_2": np.array([46, 139, 87]), "agent_3": np.array([189, 183, 107]),
                             "agent_4": np.array([165, 42, 42]), "agent_5": np.array([64, 224, 208]),
                             "agent_6": np.array([128, 128, 105]), "agent_7": np.array([210, 105, 30]),
                             "agent_8": np.array([218, 112, 214]), "agent_9": np.array([244, 164, 96]),
                             "agent_10": np.array([210, 180, 140]), "agent_11": np.array([127, 255, 212]),
                             "agent_12": np.array([138, 43, 226]), "agent_13": np.array([72, 61, 139]),
                             "agent_14": np.array([233, 150, 122])}

    def check_within_threshold(self, dict, target_key, threshold):
        # 防止起点之间或终点之间距离太近
        for key, value in dict.items():
            if key != target_key:
                if sum(abs(dict[key] - dict[target_key])) <= threshold:
                    return True  # 只要有小于阈值的元素就返回

        return False

    def set_start_and_end(self, agent, x_start, y_start, x_end, y_end):
        self.agent_positions[agent] = np.array([x_start, y_start])
        self.destinations[agent] = np.array([x_end, y_end])

    def reset(self, seed=None, options=None, map_index=None, starting_points=None, targets=None):
        self.agents = copy.deepcopy(self.possible_agents)
        self.orientation = {agent: 0 for agent in self.agents}

        self.obj_ratio = {agent: 1 for agent in self.agents}

        self.vehicle_vertices = {agent: [] for agent in self.agents}

        self.rewards = {agent: 0 for agent in self.agents}

        self.terminations = {agent: False for agent in self.possible_agents}
        self.collisions = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}

        if self.test_result_save:
            self.past_positions = {agent: [] for agent in self.agents}

        infos = {agent: {} for agent in self.agents}

        if map_index is not None:
            self.map_index = map_index
        else:
            self.map_index = random.randint(0, 2)

        # self.map_index = 2
        if self.map_index == 0:
            self.obstacle_centers = [np.array([400, 400]), np.array([400, 670]), np.array([400, 130]),
                                     np.array([130, 400]),
                                     np.array([670, 400]), np.array([130, 130]), np.array([130, 670]),
                                     np.array([670, 130]),
                                     np.array([670, 670]), np.array([265, 265]), np.array([535, 535]),
                                     np.array([265, 535]), np.array([535, 265])]

            # self.radius = [70, 70, 70, 70, 70, 70, 70, 70, 70, 40, 40, 40, 40]
            self.radius = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]

            self.rec_center = []
            self.rec_size = []

            # self.obstacle_centers = [np.array([100, 650]), np.array([100, 700]), np.array([150, 700]),
            #                          np.array([650, 300]), np.array([650, 250]), np.array([600, 250]),
            #                          np.array([180, 250])]
            #
            # self.radius = [50, 50, 50, 40, 40, 40, 60]
            #
            # self.rec_center = [np.array([650, 600]), np.array([525, 500]), np.array([525, 450]), np.array([500, 50]),
            #                    np.array([350, 75]), np.array([400, 750]), np.array([225, 450])]
            # self.rec_size = [(300, 50), (50, 200), (150, 50), (30, 100), (300, 30), (40, 100), (100, 100)]

        elif self.map_index == 1:
            self.obstacle_centers = [np.array([280, 280]), np.array([520, 520]), np.array([280, 520]),
                                     np.array([520, 280])]

            self.radius = [45, 45, 45, 45]

            self.rec_center = [np.array([90, 400]), np.array([710, 400]), np.array([400, 710]), np.array([400, 90])]

            self.rec_size = [(80, 250), (80, 250), (300, 80), (300, 80)]

            initial_positions = [np.array([100, 100]), np.array([100, 150]), np.array([150, 100]), np.array([150, 150]),
                                 np.array([100, 700]), np.array([100, 650]), np.array([150, 650]), np.array([150, 700]),
                                 np.array([700, 700]), np.array([650, 700]), np.array([650, 650]), np.array([700, 650]),
                                 np.array([700, 100]), np.array([700, 150]), np.array([650, 150]), np.array([650, 100])
                                ]

        elif self.map_index == 2:
            # self.obstacle_centers = [np.array([400, 400]), np.array([400, 630]), np.array([400, 170]),
            #                          np.array([170, 400]), np.array([630, 400]), np.array([150, 150]),
            #                          np.array([150, 650]), np.array([650, 650]), np.array([650, 150])]
            #
            # self.radius = [55, 55, 55, 55, 55, 85, 85, 85, 85]

            # self.rec_center = [np.array([400, 150]), np.array([200, 500]), np.array([550, 575]), np.array([630, 625]),
            #                    np.array([450, 375])]
            #
            # self.rec_size = [(600, 100), (100, 200), (60, 150), (100, 50), (120, 120)]

            self.rec_center = []
            self.rec_size = []

            self.obstacle_centers = [np.array([100, 650]), np.array([100, 700]), np.array([150, 700]),
                                     np.array([650, 300]), np.array([650, 250]), np.array([600, 250]),
                                     np.array([110, 250]), np.array([180, 320]), np.array([250, 320]),
                                     np.array([650, 650]), np.array([700, 600]), np.array([600, 700]),
                                     np.array([540, 700]), np.array([480, 700]), np.array([250, 520]),
                                     np.array([280, 550]), np.array([310, 520]), np.array([430, 130]),
                                     np.array([370, 130]), np.array([400, 100])]

            self.radius = [50, 50, 50, 40, 40, 40, 70, 70, 70, 60, 60, 60, 60, 60, 40, 40, 40, 40, 40, 40]

        radius_for_reset = copy.deepcopy(self.radius)
        for idx in range(len(radius_for_reset)):
            radius_for_reset[idx] += 15

        if starting_points is not None and targets is not None:
            self.agent_positions = copy.deepcopy(starting_points)
            self.destinations = copy.deepcopy(targets)
        else:

            # 初始化起点和终点
            for agent in self.agents:

                """初始化起点"""
                while True:
                    if self.map_index == 1:

                        choice = random.randint(0, len(initial_positions)-1)
                        self.agent_positions[agent] = initial_positions[choice]
                        initial_positions.pop(choice)
                        break
                    #     # print(self.agent_positions[agent])
                    else:
                        self.agent_positions[agent] = np.random.randint(low=50, high=751, size=2)
                    should_reset = False

                    # # 计算顶点
                    # vertices_for_checking = []
                    # vehicle_pos = np.array([self.agent_positions[agent][0], self.agent_positions[agent][1]])
                    #
                    # for i in range(4):
                    #     angle = math.radians(i * 90 - 45) + self.orientation[agent]  # 旋转角度
                    #     # 用三角函数计算顶点坐标
                    #     x = round(vehicle_pos[0] + self.vehicle_size / 2 * np.sqrt(2) * math.cos(angle), 2)
                    #     y = round(vehicle_pos[1] + self.vehicle_size / 2 * np.sqrt(2) * math.sin(angle), 2)
                    #     vertices_for_checking.append((x, y))

                    """判断是否和圆形障碍物过近"""
                    for i in range(len(self.obstacle_centers)):
                        if np.linalg.norm(self.agent_positions[agent] - self.obstacle_centers[i]) <= radius_for_reset[i] + \
                                self.vehicle_size * np.sqrt(2):
                            should_reset = True
                            break

                    """判断是否和长方形障碍物过近"""
                    for i in range(len(self.rec_center)):
                        current_obs_center = self.rec_center[i]
                        current_obs_size = self.rec_size[i]
                        if abs(self.agent_positions[agent][0] - current_obs_center[0]) <= current_obs_size[0] + \
                                self.vehicle_size * np.sqrt(2) and \
                           abs(self.agent_positions[agent][1] - current_obs_center[1]) <= current_obs_size[1] + \
                                self.vehicle_size * np.sqrt(2):
                            should_reset = True
                            break

                    # """判断是否和三角形障碍物过近"""
                    # for i in range(len(self.triangle_points)):
                    #
                    #     if is_collision(vertices_for_checking, self.triangle_points[i]):
                    #         should_reset = True
                    #         break
                    #
                    #     """判断点是否可能在三角形内部，因为上面的方法可能会漏判"""
                    #     for point in vertices_for_checking:
                    #         if point_in_triangle(point, self.triangle_points[i]):
                    #             should_reset = True
                    #             break

                    # 和障碍物撞，reset
                    if should_reset is True:
                        continue

                    # 和机器人起点距离近，reset
                    if self.check_within_threshold(self.agent_positions, agent, 50):
                        continue

                    # 没有以上问题, break
                    break

                """初始化终点"""
                while True:
                    self.destinations[agent] = np.random.randint(low=50, high=751, size=2)
                    should_reset = False

                    """判断是否和圆形障碍物过近"""
                    for i in range(len(self.obstacle_centers)):
                        if np.linalg.norm(self.destinations[agent] - self.obstacle_centers[i]) <= radius_for_reset[i] + \
                                self.vehicle_size + self.goal_range:
                            should_reset = True
                            break

                    """判断是否和长方形障碍物过近"""
                    for i in range(len(self.rec_center)):
                        current_obs_center = self.rec_center[i]
                        current_obs_size = self.rec_size[i]
                        if abs(self.destinations[agent][0] - current_obs_center[0]) <= current_obs_size[0] + \
                                self.vehicle_size * np.sqrt(2) and \
                            abs(self.destinations[agent][1] - current_obs_center[1]) <= current_obs_size[1] + \
                                self.vehicle_size * np.sqrt(2):
                            should_reset = True
                            break

                    # 和障碍物撞，reset
                    if should_reset is True:
                        continue

                    # 和机器人起点距离近，reset
                    if self.check_within_threshold(self.destinations, agent, 50):
                        continue

                    # 没有以上问题，break
                    break


        # 观察值为智能体位置和朝向的角度
        observations = {}

        # 初始化observations以及past_positions
        for agent in self.agents:
            value = self.agent_positions[agent]

            observations[agent] = np.append(value, np.array([self.orientation[agent]]))

            vehicle_pos = np.array([self.agent_positions[agent][0], self.agent_positions[agent][1]])

            for i in range(4):
                angle = math.radians(i * 90 - 45) + self.orientation[agent]  # 旋转角度
                # 用三角函数计算顶点坐标
                x = vehicle_pos[0] + (self.vehicle_size * np.sqrt(2) / 2) * math.cos(angle)
                y = vehicle_pos[1] + (self.vehicle_size * np.sqrt(2) / 2) * math.sin(angle)
                self.vehicle_vertices[agent].append((x, y))

            if self.test_result_save:
                self.past_positions[agent].append(np.append(value, np.array([self.orientation[agent]])))

        return observations, infos


    def repulsive_between_vehicles(self, positions):
        f_2 = 0

        """暂时还是先考虑一个车"""
        # 斥力是双向的
        for agent in self.agents:
            if self.terminations[agent] is True or self.collisions[agent] is True:
                continue

            for other_agent in self.agents:
                if agent != other_agent:

                    Frep_x, Frep_y = Repulsive(positions[agent][0], positions[agent][1], positions[other_agent][0],
                                               positions[other_agent][1], self.destinations[agent][0],
                                               self.destinations[agent][1],
                                               1, self.safe_distance, False)

                    """判断正负号是否相同"""
                    if Frep_x != 0:
                        if (positions[agent][0] - self.agent_positions[agent][0]) * Frep_x < 0:
                            f_2 += 100 * self.obj_ratio[agent]
                        else:
                            f_2 += 10 * self.obj_ratio[agent]

                    if Frep_y != 0:
                        if (positions[agent][1] - self.agent_positions[agent][1]) * Frep_y < 0:
                            f_2 += 100 * self.obj_ratio[agent]
                        else:
                            f_2 += 10 * self.obj_ratio[agent]

        return f_2

    def obs_repulsive_against_vehicle(self, positions, orientation, epsilon=1):
        f_3 = 0

        """暂时还是先考虑一个车"""
        for agent in self.agents:
            if self.terminations[agent] is True or self.collisions[agent] is True:
                continue

            # 圆形障碍物
            if len(self.obstacle_centers) > 0:
                for i in range(len(self.obstacle_centers)):

                    Frep_x, Frep_y = Repulsive(positions[agent][0], positions[agent][1], self.obstacle_centers[i][0],
                                               self.obstacle_centers[i][1], self.destinations[agent][0],
                                               self.destinations[agent][1],
                                               1, self.radius[i] + self.safe_distance, False)

                    """使用异或操作判断正负号是否相同"""
                    if Frep_x != 0:
                        if (positions[agent][0] - self.agent_positions[agent][0]) * Frep_x < 0:
                            f_3 += 100 * self.obj_ratio[agent]
                        else:
                            f_3 += 10 * self.obj_ratio[agent]

                    if Frep_y != 0:
                        if (positions[agent][1] - self.agent_positions[agent][1]) * Frep_y < 0:
                            f_3 += 100 * self.obj_ratio[agent]
                        else:
                            f_3 += 10 * self.obj_ratio[agent]


            # 针对矩形障碍物
            if len(self.rec_center) > 0:
                for i in range(len(self.rec_center)):

                    current_rec_center = self.rec_center[i]
                    current_rec_size = self.rec_size[i]
                    ori_vector = [np.cos(orientation[agent]), np.sin(orientation[agent])]
                    dis_to_nearest_rec = intersect_ray_with_rectangle(positions[agent], ori_vector, current_rec_center,
                                                                      current_rec_size[0], current_rec_size[1])

                    if dis_to_nearest_rec is not None and dis_to_nearest_rec < self.safe_distance:
                        f_3 += 1000 * self.obj_ratio[agent]

        return f_3

    def distance_to_circle_and_rec(self, positions):
        f_3 = 0
        for agent in self.agents:
            if self.terminations[agent] is True or self.collisions[agent] is True:
                continue

            if len(self.obstacle_centers) > 0:
                min_cir_dis = 10e5
                for i in range(len(self.obstacle_centers)):
                    min_cir_dis = min(min_cir_dis, np.linalg.norm(positions[agent] - self.obstacle_centers[i]) - \
                                      self.vehicle_size * np.sqrt(2) / 2)

            if min_cir_dis <= 20:
                f_3 += 5000 / min_cir_dis

            if len(self.rec_center) > 0:
                min_rec_dis = 10e5
                for i in range(len(self.rec_center)):
                    vec = positions[agent] - self.rec_center[i]
                    dist_x = abs(vec[0]) - (self.rec_size[i][0] / 2 + self.vehicle_size * np.sqrt(2) / 2)
                    dist_y = abs(vec[1]) - (self.rec_size[i][1] / 2 + self.vehicle_size * np.sqrt(2) / 2)
                    if dist_x <= 0 and dist_y <= 0:
                        return 10e4
                    min_rec_dis = min(min_rec_dis, min(abs(dist_x), abs(dist_y)))

            if min_rec_dis <= 20:
                f_3 += 5000 / min_rec_dis

        return f_3



    def objective(self, actions, epsilon=1):
        # objective越小，适应度值越大,因此要增大碰撞方案的objective

        result = 0
        """考虑多个车"""
        positions = copy.deepcopy(self.agent_positions)
        orientation_new = copy.deepcopy(self.orientation)
        for i in range(len(self.agents)):
            agent = self.agents[i]
            if self.terminations[agent] is True:
                continue

            orientation_new[agent] = self.orientation[agent] + actions[2 * i + 1]
            positions[agent][0] += np.cos(orientation_new[agent]) * actions[2 * i]
            positions[agent][1] += np.sin(orientation_new[agent]) * actions[2 * i]

        f_1 = target_toward_distance(positions, self.agent_positions, self.destinations, self.terminations,
                                     self.collisions)
        f_2 = self.repulsive_between_vehicles(positions)
        f_3 = self.obs_repulsive_against_vehicle(positions, orientation_new)

        # f_3 = self.distance_to_circle_and_rec(positions)
        result += f_1 + f_2 + f_3

        return result

    # 利用AABB法判断三角形障碍物和无人车是否相撞


    def objective_batch_torch(self, candidate_solutions, device=None):
        """GPU-friendly batch objective helper for tensorized ABC variants."""
        from gpu_objective import objective_batch_torch

        return objective_batch_torch(candidate_solutions, self, device=device)

    def step(self, actions):
        self.timestep += 1

        time = 0.5

        # 记录未执行动作前当前位置指向目标的向量方向，为后面计算奖励用
        original_vector_to_target = {}
        delta_x = {}
        delta_y = {}
        observations = {}
        self.rewards = {agent: 0 for agent in self.agents}
        path_dis = 0

        # 执行动作
        for agent in self.agents:
            # 如果已经到终点或者撞了就跳过
            if self.terminations[agent] is True or self.collisions[agent] is True:
                continue

            agent_action = actions[agent]
            original_vector_to_target[agent] = self.destinations[agent] - self.agent_positions[agent]

            # 根据动力学模型，时间默认为1
            orientation_new = self.orientation[agent] + agent_action[1]
            x_new = round(self.agent_positions[agent][0] + np.cos(orientation_new) * agent_action[0] * time, 2)
            x_new = min(795, max(5, x_new))
            delta_x[agent] = round(np.cos(orientation_new) * agent_action[0] * time, 2)

            y_new = round(self.agent_positions[agent][1] + np.sin(orientation_new) * agent_action[0] * time, 2)
            y_new = min(795, max(5, y_new))
            delta_y[agent] = round(np.sin(orientation_new) * agent_action[0] * time, 2)

            self.agent_positions[agent] = np.array([x_new, y_new])
            self.orientation[agent] = orientation_new
            path_dis += np.sqrt(delta_x[agent]**2 + delta_y[agent]**2)


        # 计算奖励
        for agent in self.agents:

            vehicle_pos = copy.deepcopy(self.agent_positions[agent])

            for i in range(4):
                angle = math.radians(i * 90 - 45) + self.orientation[agent]  # 旋转角度
                # 用三角函数计算顶点坐标
                x = round(vehicle_pos[0] + (self.vehicle_size * np.sqrt(2) / 2) * math.cos(angle), 2)
                y = round(vehicle_pos[1] + (self.vehicle_size * np.sqrt(2) / 2) * math.sin(angle), 2)
                self.vehicle_vertices[agent].append((x, y))

            if self.terminations[agent] is True:
                self.rewards[agent] = 0
                continue
            elif self.collisions[agent] is True:
                self.rewards[agent] = -10
                continue

            # 判断是否到终点
            if np.linalg.norm(self.agent_positions[agent] - self.destinations[agent]) <= self.goal_range:
                self.terminations[agent] = True
                self.rewards[agent] = 100
                self.obj_ratio[agent] = 0
                continue

            # 判断机器人之间是否撞
            if self.agent_num > 1:
                for other_agent in self.agents:
                    if agent != other_agent:
                        if np.linalg.norm(self.agent_positions[agent] - \
                                          self.agent_positions[other_agent]) < self.safe_distance / 2:
                            self.collisions[agent] = True
                            self.rewards[agent] = -1000
                            self.collisions[other_agent] = True
                            self.rewards[other_agent] = -1000
                            print("Collision type: robots")
                            print("Robots: {}, {}".format(agent, other_agent))
                            print("Distance: {}".format(
                                np.linalg.norm(self.agent_positions[agent] - self.agent_positions[other_agent])))
                            print("Positions: {}, {}".format(self.agent_positions[agent],
                                                             self.agent_positions[other_agent]))
                            break

            if self.collisions[agent] is True:
                continue

            # 判断是否撞到边界
            if any(self.agent_positions[agent] <= 0 + self.vehicle_size / 2) or \
                    any(self.agent_positions[agent] >= self.screen_width - self.vehicle_size / 2):
                self.collisions[agent] = True
                self.rewards[agent] = -1000
                print("Collision type: border")
                print("Robot: {}".format(agent))
                print("Position: {}".format(self.agent_positions[agent]))
                continue

            # 判断是否撞到障碍物
            # 圆形障碍物
            if len(self.obstacle_centers) > 0:
                for i in range(len(self.obstacle_centers)):

                    """由于把机器人看成圆，探测碰撞的范围会变大，因此在图中车和障碍物可能不会碰撞"""
                    if np.linalg.norm(self.obstacle_centers[i] - self.agent_positions[agent]) <= self.radius[
                        i] + self.vehicle_size / 2:
                        self.collisions[agent] = True
                        self.rewards[agent] = -1000
                        print("Collision type: circle obstacle")
                        print("Robot: {}".format(agent))
                        print("Distance: {} \t Obstacle center: {} \t Position: {} \t Radius: {}".format(
                            np.linalg.norm(self.obstacle_centers[i] - self.agent_positions[agent]),
                            self.obstacle_centers[i], self.agent_positions[agent], self.radius[i]))
                        break

            # 长方形障碍物
            if len(self.rec_center) > 0:
                for i in range(len(self.rec_center)):
                    current_obs_center = self.rec_center[i]
                    current_obs_size = self.rec_size[i]

                    if abs(self.agent_positions[agent][0] - current_obs_center[0]) <= current_obs_size[0] / 2 + self.vehicle_size / 2 and \
                        abs(self.agent_positions[agent][1] - current_obs_center[1]) <= current_obs_size[1] / 2 + self.vehicle_size / 2:
                        self.collisions[agent] = True
                        self.rewards[agent] -= 1000
                        print("Collision type: rectangle obstacle")
                        print("Robot: {}".format(agent))
                        print("Obstacle center: {} \t Position: {} \t Size: {}".format(current_obs_center,
                                self.agent_positions[agent], current_obs_size))
                        break

            # # 三角形障碍物
            # if len(self.triangle_points) > 0:
            #     # 计算无人车四个顶点
            #     for i in range(len(self.triangle_points)):
            #         if is_collision(self.vehicle_vertices[agent], self.triangle_points[i]):
            #             self.collisions[agent] = True
            #             self.rewards[agent] = -1000
            #             print("Collision type: triangle obstacle")
            #             print("Robot: {}".format(agent))
            #             break
            #
            #         """判断点是否可能在三角形内部，因为上面的方法可能会漏判"""
            #         for point in self.vehicle_vertices[agent]:
            #             if point_in_triangle(point, self.triangle_points[i]):
            #                 self.collisions[agent] = True
            #                 self.rewards[agent] = -1000
            #                 print("Collision type: triangle obstacle")
            #                 print("Robot: {}".format(agent))
            #                 break

            # 坐标增量是否按照朝向目标
            if original_vector_to_target[agent][0] * delta_x[agent] > 0:
                self.rewards[agent] += 2

            if original_vector_to_target[agent][1] * delta_y[agent] > 0:
                self.rewards[agent] += 2

            # 时间惩罚
            self.rewards[agent] -= 10

            # 返回状态值
            value = self.agent_positions[agent]
            observations[agent] = np.append(value, np.array([self.orientation[agent]]))

            if self.test_result_save:
                self.past_positions[agent].append(np.append(value, np.array([self.orientation[agent]])))

        """这个返回值形式针对PPO"""
        return observations, sum(self.rewards.values()), all(
            self.terminations.values()), self.collisions, self.truncations, path_dis

        """这个返回值形式针对MARL"""
        # return observations, self.rewards, self.terminations, self.collisions, self.truncations, {}

    def seed(self, seed):
        pass

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        scale = self.screen_width / self.screen_width

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        xs = np.linspace(0, self.screen_width - 1, self.screen_width)
        ys = np.linspace(0, self.screen_height - 1, self.screen_height)
        xys = list(zip((xs - 0) * scale, ys * scale))

        # obstacles
        # 圆形
        for i in range(len(self.obstacle_centers)):
            obstacle_center = np.array([self.obstacle_centers[i][0], self.obstacle_centers[i][1]])
            pygame.draw.circle(self.surf, (0, 0, 0), obstacle_center, self.radius[i])

        # 长方形
        for i in range(len(self.rec_center)):
            # 计算长方形左上角的点的位置
            rec_left_up_point = copy.deepcopy(self.rec_center[i])
            rec_left_up_point[0] -= self.rec_size[i][0] / 2
            rec_left_up_point[1] -= self.rec_size[i][1] / 2

            pygame.draw.rect(self.surf, (0, 0, 0), (rec_left_up_point, self.rec_size[i]))

        # # 三角形
        # if len(self.triangle_points) > 0:
        #     for i in range(len(self.triangle_points)):
        #         pygame.draw.polygon(self.surf, (0, 0, 0), self.triangle_points[i])



        # position是车辆中心的位置
        for agent in self.agents:
            # 绘制旋转的正方形
            pygame.draw.polygon(self.surf, self.agent_colors[agent], self.vehicle_vertices[agent])

            # 在尾端画两个小轮，判断哪边是车尾
            pygame.draw.circle(self.surf, self.agent_colors[agent], self.vehicle_vertices[agent][2], 2)
            pygame.draw.circle(self.surf, self.agent_colors[agent], self.vehicle_vertices[agent][3], 2)

            # 画目的地以及goal range
            vehicle_des = np.array([self.destinations[agent][0], self.destinations[agent][1]])
            # pygame.draw.circle(self.surf, self.agent_colors[agent], vehicle_des, self.vehicle_size / 2)
            pygame.draw.circle(self.surf, self.agent_colors[agent], vehicle_des,
                               self.vehicle_size / 2 + self.goal_range,
                               width=2)

            self.vehicle_vertices[agent] = []

        self.screen.blit(self.surf, (0, 0))

        enhance_channel_with_positions = np.zeros((self.screen_width, self.screen_height, 1))

        # 存个开始的地图
        # if self.test_result_save and self.timestep == 0:
        #     pygame.image.save(self.surf, "initial_map_{}.png".format(datetime.now().strftime("%m%d-%H-%M-%S")))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

            if all([self.terminations[agent] for agent in self.agents]) and self.test_result_save:
                for agent in self.agents:
                    for idx in range(len(self.past_positions[agent]) - 1):
                        current_pos = self.past_positions[agent][idx]
                        vehicle_pos = np.array([current_pos[0], current_pos[1]])
                        orientation = current_pos[2]


                        next_pos = self.past_positions[agent][idx + 1]
                        next_vehicle_pos = np.array([next_pos[0], next_pos[1]])

                        points = []
                        for i in range(4):
                            angle = math.radians(i * 90 - 45) + orientation  # 旋转角度
                            # 用三角函数计算顶点坐标
                            x = vehicle_pos[0] + (self.vehicle_size * np.sqrt(2) / 2) * math.cos(angle)
                            y = vehicle_pos[1] + (self.vehicle_size * np.sqrt(2) / 2) * math.sin(angle)
                            points.append((x, y))

                        # # 画最后时刻的车辆
                        # if idx == 0 or idx == len(self.past_positions[agent]) - 2:
                        #     points = []
                        #     if idx == 0:
                        #         for i in range(4):
                        #             angle = math.radians(i * 90 - 45) + orientation  # 旋转角度
                        #             # 用三角函数计算顶点坐标
                        #             x = vehicle_pos[0] + (self.vehicle_size * np.sqrt(2) / 2) * math.cos(angle)
                        #             y = vehicle_pos[1] + (self.vehicle_size * np.sqrt(2) / 2) * math.sin(angle)
                        #             points.append((x, y))
                        #     if idx == len(self.past_positions[agent]) - 2:
                        #         for i in range(4):
                        #             angle = math.radians(i * 90 - 45) + orientation  # 旋转角度
                        #             # 用三角函数计算顶点坐标
                        #             x = self.agent_positions[agent][0] + (self.vehicle_size * np.sqrt(2) / 2) * math.cos(angle)
                        #             y = self.agent_positions[agent][1] + (self.vehicle_size * np.sqrt(2) / 2) * math.sin(angle)
                        #             points.append((x, y))

                            # 绘制旋转的正方形
                        if idx == 0:
                            pygame.draw.polygon(self.surf, self.agent_colors[agent], points)
                        else:
                            pygame.draw.polygon(self.surf, self.agent_colors[agent], points, 2)


                        # 绘制两个时刻的位置之间的连线
                        pygame.draw.line(self.surf, self.agent_colors[agent], vehicle_pos, next_vehicle_pos, 2)

                    pygame.draw.line(self.surf, self.agent_colors[agent], next_vehicle_pos, self.agent_positions[agent], 2)


            else:
                # hsv
                # 三个维度按顺序为x, y, channel
                rgb_array_for_return = pygame.surfarray.pixels3d(self.screen)
                if cv2 is None:
                    hsv_array = rgb_array_for_return
                else:
                    hsv_array = cv2.cvtColor(rgb_array_for_return, cv2.COLOR_RGB2HSV)

                for agent in self.agents:
                    x = round(self.agent_positions[agent][0])
                    y = round(self.agent_positions[agent][1])
                    for y_ in range(max(0, y - 5), min(799, y + 5)):
                        enhance_channel_with_positions[max(0, x - 5):min(799, x + 5), y_, 0] = 0.5 * 255.0

                hsv_array_for_return = np.concatenate((hsv_array, enhance_channel_with_positions), axis=2)

                return hsv_array_for_return

        elif self.render_mode == "rgb_array":
            """0：width，1：height，2：channel"""

            # hsv
            # 三个维度按顺序为x, y, channel
            rgb_array_for_return = pygame.surfarray.pixels3d(self.screen)
            if cv2 is None:
                hsv_array = rgb_array_for_return
            else:
                hsv_array = cv2.cvtColor(rgb_array_for_return, cv2.COLOR_RGB2HSV)

            for agent in self.agents:
                x = int(self.agent_positions[agent][0])
                y = int(self.agent_positions[agent][1])
                for y_ in range(max(0, int(y - self.vehicle_size / 2)), min(799, int(y + self.vehicle_size / 2))):
                    enhance_channel_with_positions[
                    max(0, int(x - self.vehicle_size / 2)):min(799, int(x + self.vehicle_size / 2)), y_,
                    0] = 0.5 * 255.0

            hsv_array_for_return = np.concatenate((hsv_array, enhance_channel_with_positions), axis=2)

            return hsv_array_for_return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=np.array([5.0, 5.0, -math.pi / 2]),
                   high=np.array([795.0, 795.0, math.pi / 2]), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Box(low=np.array([1.0, -math.pi / 2]),
                   high=np.array([20.0, math.pi / 2]), dtype=np.float32)


if __name__ == "__main__":
    from pettingzoo.test import parallel_api_test

    env = Map(agent_num=10, render_mode="human", test_result_save=True)
    # env.reset()
    # parallel_api_test(env, num_cycles=10000)

    seed = 42
    import torch

    torch.manual_seed(seed)
    obs, _ = env.reset(seed=seed, map_index=2)
    # print(list(env.agent_positions.values()))

    # env.set_start_and_end("agent_0", 20, 150, 100, 10)

    while True:
        env.render()

        action = {"agent_0": env.action_space("agent_0").sample()}
        action = {"agent_0": np.array([5, 0])}

        obs, rews, terms, collisions, truncs, _ = env.step(action)


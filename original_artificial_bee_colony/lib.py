import math

import numpy as np


def target_toward_distance(positions, old_positions, destinations, terminations, collisions):
    cost = 0.0
    for agent, position in positions.items():
        if terminations.get(agent, False) or collisions.get(agent, False):
            continue
        old_distance = np.linalg.norm(old_positions[agent] - destinations[agent])
        new_distance = np.linalg.norm(position - destinations[agent])
        cost += new_distance
        if new_distance > old_distance:
            cost += 20.0
    return cost


def Repulsive(x, y, obs_x, obs_y, goal_x, goal_y, eta, influence_radius, _use_goal=True):
    delta = np.array([x - obs_x, y - obs_y], dtype=float)
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-6 or distance >= influence_radius:
        return 0.0, 0.0

    direction = delta / distance
    scale = eta * (1.0 / distance - 1.0 / influence_radius) / (distance * distance)
    force = scale * direction
    return float(force[0]), float(force[1])


def intersect_ray_with_rectangle(origin, direction, rect_center, width, height):
    ox, oy = float(origin[0]), float(origin[1])
    dx, dy = float(direction[0]), float(direction[1])
    if abs(dx) <= 1e-9 and abs(dy) <= 1e-9:
        return None

    min_x = float(rect_center[0]) - width / 2
    max_x = float(rect_center[0]) + width / 2
    min_y = float(rect_center[1]) - height / 2
    max_y = float(rect_center[1]) + height / 2

    t_min = -math.inf
    t_max = math.inf
    for origin_axis, direction_axis, low, high in (
        (ox, dx, min_x, max_x),
        (oy, dy, min_y, max_y),
    ):
        if abs(direction_axis) <= 1e-9:
            if origin_axis < low or origin_axis > high:
                return None
            continue
        t1 = (low - origin_axis) / direction_axis
        t2 = (high - origin_axis) / direction_axis
        t_near = min(t1, t2)
        t_far = max(t1, t2)
        t_min = max(t_min, t_near)
        t_max = min(t_max, t_far)

    if t_max < 0 or t_min > t_max:
        return None
    return max(0.0, t_min)

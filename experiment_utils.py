import copy
import csv
import math
import os
import random
import time
from dataclasses import dataclass, field

import numpy as np


def expand_bounds(agent_num, lower_bound, upper_bound):
    fn_lb = []
    fn_ub = []
    for _ in range(agent_num):
        fn_lb.extend(lower_bound)
        fn_ub.extend(upper_bound)
    return fn_lb, fn_ub


def solution_to_actions(solution, agents):
    return {
        agent: np.array([solution[2 * idx], solution[2 * idx + 1]], dtype=float)
        for idx, agent in enumerate(agents)
    }


def safe_stop_turn_action(env, agent, turn_low=-math.pi / 4, turn_high=math.pi / 4):
    """Return a legal fallback action when the candidate move is unsafe."""
    return 0.0, float(np.random.uniform(turn_low, turn_high))


def apply_fallback_to_solution(solution, idx, env, agent):
    speed, turn = safe_stop_turn_action(env, agent)
    solution[2 * idx] = speed
    solution[2 * idx + 1] = turn
    if hasattr(env, "obj_ratio"):
        env.obj_ratio[agent] += 0.1


def repair_candidate_solution(solution, env):
    """Clamp unsafe candidate actions to a legal stop-and-turn fallback.

    The old code used speed = -1 for unsafe actions. That creates a hidden
    reverse maneuver outside the configured action bounds, so this repair keeps
    every returned action within the experimental action space.
    """
    repaired = solution
    positions = copy.deepcopy(env.agent_positions)

    for idx, agent in enumerate(env.possible_agents):
        if env.terminations[agent] is True or env.collisions[agent] is True:
            continue

        orientation_new = env.orientation[agent] + repaired[2 * idx + 1]
        positions[agent][0] += np.cos(orientation_new) * repaired[2 * idx]
        positions[agent][1] += np.sin(orientation_new) * repaired[2 * idx]

        if any(positions[agent] <= env.safe_distance / 2) or any(
            positions[agent] >= env.screen_width - env.safe_distance / 2
        ):
            apply_fallback_to_solution(repaired, idx, env, agent)
            continue

        for obs_idx in range(len(env.obstacle_centers)):
            if np.linalg.norm(env.obstacle_centers[obs_idx] - positions[agent]) <= (
                env.radius[obs_idx] + env.safe_distance / 2
            ):
                apply_fallback_to_solution(repaired, idx, env, agent)
                break

        for obs_idx in range(len(env.rec_center)):
            current_obs_center = env.rec_center[obs_idx]
            current_obs_size = env.rec_size[obs_idx]

            if (
                abs(positions[agent][0] - current_obs_center[0])
                <= current_obs_size[0] / 2 + env.safe_distance / 2
                and abs(positions[agent][1] - current_obs_center[1])
                <= current_obs_size[1] / 2 + env.safe_distance / 2
            ):
                apply_fallback_to_solution(repaired, idx, env, agent)
                break

    for idx, agent in enumerate(env.agents):
        for other_idx in range(idx + 1, len(env.agents)):
            other_agent = env.agents[other_idx]
            if np.linalg.norm(positions[agent] - positions[other_agent]) < env.safe_distance / 2:
                apply_fallback_to_solution(repaired, idx, env, agent)
                apply_fallback_to_solution(repaired, other_idx, env, other_agent)

    return repaired


def _collides_with_obstacles(env, point):
    for idx, center in enumerate(env.obstacle_centers):
        if np.linalg.norm(point - center) <= env.radius[idx] + env.vehicle_size + env.goal_range:
            return True

    for idx, center in enumerate(env.rec_center):
        size = env.rec_size[idx]
        if (
            abs(point[0] - center[0]) <= size[0] / 2 + env.vehicle_size + env.goal_range
            and abs(point[1] - center[1]) <= size[1] / 2 + env.vehicle_size + env.goal_range
        ):
            return True

    return False


def sample_start_targets(env, seed=None, min_distance=120, pair_threshold=50, max_attempts=5000):
    rng = np.random.default_rng(seed)
    starts = {}
    targets = {}

    for agent in env.possible_agents:
        for _ in range(max_attempts):
            point = rng.integers(50, 751, size=2).astype(float)
            if _collides_with_obstacles(env, point):
                continue
            if any(np.linalg.norm(point - other) < pair_threshold for other in starts.values()):
                continue
            starts[agent] = point
            break
        else:
            raise RuntimeError(f"Could not sample a valid start for {agent}")

        for _ in range(max_attempts):
            point = rng.integers(50, 751, size=2).astype(float)
            if _collides_with_obstacles(env, point):
                continue
            if np.linalg.norm(point - starts[agent]) < min_distance:
                continue
            if any(np.linalg.norm(point - other) < pair_threshold for other in targets.values()):
                continue
            targets[agent] = point
            break
        else:
            raise RuntimeError(f"Could not sample a valid target for {agent}")

    return starts, targets


def manual_probabilities(env):
    active_agents = [
        agent
        for agent in env.agents
        if not env.terminations[agent] and not env.collisions[agent]
    ]
    if not active_agents:
        return [1 / 3, 1 / 3, 1 / 3]

    distances_to_goal = [
        np.linalg.norm(env.agent_positions[agent] - env.destinations[agent])
        for agent in active_agents
    ]
    min_pair_distance = float("inf")
    for idx, agent in enumerate(active_agents):
        for other in active_agents[idx + 1:]:
            min_pair_distance = min(
                min_pair_distance,
                np.linalg.norm(env.agent_positions[agent] - env.agent_positions[other]),
            )

    if min_pair_distance < env.safe_distance:
        return [0.15, 0.65, 0.20]
    if float(np.mean(distances_to_goal)) > 250:
        return [0.55, 0.25, 0.20]
    return [0.25, 0.25, 0.50]


@dataclass
class EpisodeResult:
    algorithm: str
    map_index: int
    agent_num: int
    episode: int
    success: bool
    collision: bool
    timeout: bool
    steps: int
    reward: float
    path_distance: float
    elapsed_time: float
    avg_step_time: float
    probability_trace: list = field(default_factory=list)


def summarize_results(results):
    if not results:
        return {}

    rewards = np.array([r.reward for r in results], dtype=float)
    steps = np.array([r.steps for r in results], dtype=float)
    distances = np.array([r.path_distance for r in results], dtype=float)
    elapsed = np.array([r.elapsed_time for r in results], dtype=float)
    probability_rows = [
        probabilities
        for result in results
        for probabilities in result.probability_trace
        if probabilities is not None
    ]
    if probability_rows:
        mean_probabilities = np.mean(np.array(probability_rows, dtype=float), axis=0)
    else:
        mean_probabilities = [np.nan, np.nan, np.nan]
    return {
        "episodes": len(results),
        "success_rate": float(np.mean([r.success for r in results])),
        "collision_rate": float(np.mean([r.collision for r in results])),
        "timeout_rate": float(np.mean([r.timeout for r in results])),
        "mean_reward": float(np.mean(rewards)),
        "mean_steps": float(np.mean(steps)),
        "mean_path_distance": float(np.mean(distances)),
        "mean_elapsed_time": float(np.mean(elapsed)),
        "mean_strategy_1": float(mean_probabilities[0]),
        "mean_strategy_2": float(mean_probabilities[1]),
        "mean_strategy_3": float(mean_probabilities[2]),
    }


def write_episode_results(path, results):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algorithm",
                "map_index",
                "agent_num",
                "episode",
                "success",
                "collision",
                "timeout",
                "steps",
                "reward",
                "path_distance",
                "elapsed_time",
                "avg_step_time",
            ],
        )
        writer.writeheader()
        for result in results:
            row = result.__dict__.copy()
            row.pop("probability_trace", None)
            writer.writerow(row)


def write_summary(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "algorithm",
            "map_index",
            "agent_num",
            "episodes",
            "success_rate",
            "collision_rate",
            "timeout_rate",
            "mean_reward",
            "mean_steps",
            "mean_path_distance",
            "mean_elapsed_time",
            "mean_strategy_1",
            "mean_strategy_2",
            "mean_strategy_3",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

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


def predict_positions_from_solution(solution, env, time_scale=1.0):
    positions = copy.deepcopy(env.agent_positions)
    orientations = copy.deepcopy(env.orientation)
    for idx, agent in enumerate(env.agents):
        if env.terminations[agent] is True or env.collisions[agent] is True:
            continue
        speed = solution[2 * idx]
        turn = solution[2 * idx + 1]
        orientations[agent] = env.orientation[agent] + turn
        positions[agent][0] += np.cos(orientations[agent]) * speed * time_scale
        positions[agent][1] += np.sin(orientations[agent]) * speed * time_scale
    return positions, orientations


def compute_agent_priority(env, agent):
    if env.terminations[agent] or env.collisions[agent]:
        return -float("inf")
    distance_to_goal = np.linalg.norm(env.agent_positions[agent] - env.destinations[agent])
    progress_bonus = 1.0 / (distance_to_goal + 1.0)
    return progress_bonus - 0.001 * distance_to_goal


def detect_conflicts(env, solution=None, threshold=None):
    """Detect predicted pair conflicts for the current or candidate state."""
    if threshold is None:
        threshold = env.safe_distance
    if solution is None:
        positions = env.agent_positions
    else:
        positions, _ = predict_positions_from_solution(solution, env)

    conflicts = []
    active_agents = [
        agent
        for agent in env.agents
        if not env.terminations[agent] and not env.collisions[agent]
    ]
    for idx, agent in enumerate(active_agents):
        for other_agent in active_agents[idx + 1:]:
            distance = np.linalg.norm(positions[agent] - positions[other_agent])
            if distance < threshold:
                conflicts.append((agent, other_agent, float(distance)))
    return conflicts


def conflict_avoidance_turn(env, agent, other_agent):
    delta = env.agent_positions[agent] - env.agent_positions[other_agent]
    target_delta = env.destinations[agent] - env.agent_positions[agent]
    cross = delta[0] * target_delta[1] - delta[1] * target_delta[0]
    return -1.0 if cross >= 0 else 1.0


def conflict_aware_repair(solution, env):
    """Repair predicted robot-robot conflicts with priority-aware yielding."""
    repaired = np.copy(solution)
    conflicts = detect_conflicts(env, repaired, threshold=env.safe_distance)
    if not conflicts:
        return repaired

    for agent, other_agent, _ in conflicts:
        agent_idx = env.agents.index(agent)
        other_idx = env.agents.index(other_agent)
        agent_priority = compute_agent_priority(env, agent)
        other_priority = compute_agent_priority(env, other_agent)
        yield_agent, keep_agent = (
            (agent, other_agent)
            if agent_priority < other_priority
            else (other_agent, agent)
        )
        yield_idx = env.agents.index(yield_agent)
        keep_idx = env.agents.index(keep_agent)

        repaired[2 * yield_idx] = max(0.0, repaired[2 * yield_idx] * 0.35)
        repaired[2 * yield_idx + 1] += conflict_avoidance_turn(env, yield_agent, keep_agent) * math.pi / 8
        repaired[2 * keep_idx] = max(repaired[2 * keep_idx], solution[2 * keep_idx])

    return repaired


def conflict_aware_search_guidance(solution, env, strength=0.25):
    """Nudge candidate actions away from predicted pair conflicts."""
    guided = np.copy(solution)
    conflicts = detect_conflicts(env, guided, threshold=env.safe_distance * 1.25)
    if not conflicts:
        return guided

    for agent, other_agent, distance in conflicts:
        agent_idx = env.agents.index(agent)
        other_idx = env.agents.index(other_agent)
        severity = max(0.0, (env.safe_distance * 1.25 - distance) / (env.safe_distance * 1.25))
        turn_step = strength * severity * math.pi / 4

        if compute_agent_priority(env, agent) < compute_agent_priority(env, other_agent):
            guided[2 * agent_idx] *= max(0.2, 1.0 - strength * severity)
            guided[2 * agent_idx + 1] += conflict_avoidance_turn(env, agent, other_agent) * turn_step
        else:
            guided[2 * other_idx] *= max(0.2, 1.0 - strength * severity)
            guided[2 * other_idx + 1] += conflict_avoidance_turn(env, other_agent, agent) * turn_step

    return guided


def repair_candidate_solution(solution, env, use_conflict_aware=False):
    """Clamp unsafe candidate actions to a legal stop-and-turn fallback.

    The old code used speed = -1 for unsafe actions. That creates a hidden
    reverse maneuver outside the configured action bounds, so this repair keeps
    every returned action within the experimental action space.
    """
    repaired = np.copy(solution)
    if use_conflict_aware:
        repaired = conflict_aware_repair(repaired, env)
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


def nearest_obstacle_distance(env, point):
    distances = []
    for idx, center in enumerate(env.obstacle_centers):
        distances.append(np.linalg.norm(point - center) - env.radius[idx])

    for idx, center in enumerate(env.rec_center):
        size = env.rec_size[idx]
        dx = max(abs(point[0] - center[0]) - size[0] / 2, 0.0)
        dy = max(abs(point[1] - center[1]) - size[1] / 2, 0.0)
        distances.append(math.hypot(dx, dy))

    return min(distances) if distances else float("inf")


def agent_manual_probabilities(env):
    """Return a strategy profile per agent from local navigation context."""
    profiles = {}
    active_agents = [
        agent
        for agent in env.agents
        if not env.terminations[agent] and not env.collisions[agent]
    ]

    for agent in env.agents:
        if agent not in active_agents:
            profiles[agent] = [1 / 3, 1 / 3, 1 / 3]
            continue

        position = env.agent_positions[agent]
        distance_to_goal = np.linalg.norm(position - env.destinations[agent])
        obstacle_distance = nearest_obstacle_distance(env, position)
        robot_distance = float("inf")
        for other_agent in active_agents:
            if other_agent == agent:
                continue
            robot_distance = min(
                robot_distance,
                np.linalg.norm(position - env.agent_positions[other_agent]),
            )

        if robot_distance < env.safe_distance:
            profiles[agent] = [0.15, 0.65, 0.20]
        elif obstacle_distance < env.safe_distance:
            profiles[agent] = [0.20, 0.30, 0.50]
        elif distance_to_goal > 250:
            profiles[agent] = [0.60, 0.25, 0.15]
        else:
            profiles[agent] = [0.25, 0.25, 0.50]

    return profiles


def flatten_probability_trace(probabilities):
    """Convert global or agent-wise profiles to one mean profile row."""
    if probabilities is None:
        return None
    if isinstance(probabilities, dict):
        rows = list(probabilities.values())
        if not rows:
            return None
        return np.mean(np.array(rows, dtype=float), axis=0)
    return np.array(probabilities, dtype=float)


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
        row
        for result in results
        for probabilities in result.probability_trace
        for row in [flatten_probability_trace(probabilities)]
        if row is not None
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

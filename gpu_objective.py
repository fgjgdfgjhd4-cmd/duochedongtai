"""Batch-parallel candidate evaluation helpers for multi-robot planning.

The functions here keep the environment loop on CPU but move the high-frequency
geometry used by candidate scoring to torch tensors. They are intentionally
standalone so the legacy NumPy objective can remain the default reference.
"""

import numpy as np
import torch


def _device(device=None):
    if device is not None:
        return torch.device(device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _stack_agent_values(env, values, device):
    return torch.tensor(
        np.array([values[agent] for agent in env.agents]),
        dtype=torch.float32,
        device=device,
    )


def _active_mask(env, device):
    return torch.tensor(
        [
            not env.terminations[agent] and not env.collisions[agent]
            for agent in env.agents
        ],
        dtype=torch.bool,
        device=device,
    )


def objective_batch_torch(candidate_solutions, env, device=None):
    """Evaluate a batch of action candidates with torch tensor geometry.

    This is a GPU-friendly approximation of ``Map.objective`` designed for
    population-level pre-scoring. It returns one objective value per candidate;
    lower is better, matching the existing ABC fitness convention.
    """
    device = _device(device)
    candidates = torch.as_tensor(candidate_solutions, dtype=torch.float32, device=device)
    if candidates.ndim == 1:
        candidates = candidates.unsqueeze(0)

    batch_size = candidates.shape[0]
    agent_num = len(env.agents)
    actions = candidates.reshape(batch_size, agent_num, 2)
    speeds = actions[:, :, 0]
    turns = actions[:, :, 1]

    positions = _stack_agent_values(env, env.agent_positions, device)
    destinations = _stack_agent_values(env, env.destinations, device)
    orientations = torch.tensor(
        [env.orientation[agent] for agent in env.agents],
        dtype=torch.float32,
        device=device,
    )
    active = _active_mask(env, device)

    next_orientations = orientations.unsqueeze(0) + turns
    next_positions = positions.unsqueeze(0).repeat(batch_size, 1, 1)
    next_positions[:, :, 0] += torch.cos(next_orientations) * speeds
    next_positions[:, :, 1] += torch.sin(next_orientations) * speeds

    old_dist = torch.linalg.norm(positions - destinations, dim=-1)
    new_dist = torch.linalg.norm(next_positions - destinations.unsqueeze(0), dim=-1)
    progress_penalty = torch.where(
        new_dist > old_dist.unsqueeze(0),
        torch.full_like(new_dist, 20.0),
        torch.zeros_like(new_dist),
    )
    active_float = active.float().unsqueeze(0)
    target_cost = ((new_dist + progress_penalty) * active_float).sum(dim=1)

    pair_cost = torch.zeros(batch_size, dtype=torch.float32, device=device)
    if agent_num > 1:
        pair_delta = next_positions[:, :, None, :] - next_positions[:, None, :, :]
        pair_dist = torch.linalg.norm(pair_delta, dim=-1)
        eye = torch.eye(agent_num, dtype=torch.bool, device=device).unsqueeze(0)
        active_pairs = active[None, :, None] & active[None, None, :] & ~eye
        pair_risk = torch.clamp(env.safe_distance - pair_dist, min=0.0)
        pair_cost = (pair_risk * active_pairs.float()).sum(dim=(1, 2)) * 10.0

    obstacle_cost = torch.zeros(batch_size, dtype=torch.float32, device=device)
    if len(env.obstacle_centers) > 0:
        centers = torch.tensor(np.array(env.obstacle_centers), dtype=torch.float32, device=device)
        radii = torch.tensor(env.radius, dtype=torch.float32, device=device)
        delta = next_positions[:, :, None, :] - centers[None, None, :, :]
        dist = torch.linalg.norm(delta, dim=-1) - radii[None, None, :]
        obstacle_risk = torch.clamp(env.safe_distance - dist, min=0.0)
        obstacle_cost = obstacle_cost + (obstacle_risk * active_float.unsqueeze(-1)).sum(dim=(1, 2)) * 10.0

    if len(env.rec_center) > 0:
        centers = torch.tensor(np.array(env.rec_center), dtype=torch.float32, device=device)
        sizes = torch.tensor(np.array(env.rec_size), dtype=torch.float32, device=device)
        delta = torch.abs(next_positions[:, :, None, :] - centers[None, None, :, :])
        clearance = delta - sizes[None, None, :, :] / 2
        outside = torch.clamp(clearance, min=0.0)
        dist = torch.linalg.norm(outside, dim=-1)
        rect_risk = torch.clamp(env.safe_distance - dist, min=0.0)
        obstacle_cost = obstacle_cost + (rect_risk * active_float.unsqueeze(-1)).sum(dim=(1, 2)) * 10.0

    return target_cost + pair_cost + obstacle_cost

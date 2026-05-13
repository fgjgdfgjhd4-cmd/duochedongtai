"""Structured-state MLP scheduler for agent-wise IABC strategy profiles.

This module is a lightweight bridge between hand-written agent-aware rules and
future graph encoders. It uses explicit local robot features instead of rendered
pixels, runs on CUDA when available, and outputs one 3-strategy profile per
agent.
"""

import math

import numpy as np
import torch
import torch.nn as nn

from experiment_utils import nearest_obstacle_distance


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def agent_feature_matrix(env):
    rows = []
    active_agents = [
        agent
        for agent in env.agents
        if not env.terminations[agent] and not env.collisions[agent]
    ]
    diagonal = math.sqrt(env.screen_width ** 2 + env.screen_height ** 2)

    for agent in env.agents:
        position = env.agent_positions[agent]
        target = env.destinations[agent]
        target_vec = target - position
        distance_to_goal = np.linalg.norm(target_vec)
        target_angle = math.atan2(target_vec[1], target_vec[0])
        heading_error = math.atan2(
            math.sin(target_angle - env.orientation[agent]),
            math.cos(target_angle - env.orientation[agent]),
        )

        nearest_robot = diagonal
        for other_agent in active_agents:
            if other_agent == agent:
                continue
            nearest_robot = min(
                nearest_robot,
                np.linalg.norm(position - env.agent_positions[other_agent]),
            )

        obstacle_distance = nearest_obstacle_distance(env, position)
        if not np.isfinite(obstacle_distance):
            obstacle_distance = diagonal

        rows.append(
            [
                position[0] / env.screen_width,
                position[1] / env.screen_height,
                target[0] / env.screen_width,
                target[1] / env.screen_height,
                distance_to_goal / diagonal,
                nearest_robot / diagonal,
                obstacle_distance / diagonal,
                heading_error / math.pi,
                float(env.terminations[agent]),
                float(env.collisions[agent]),
            ]
        )

    return np.array(rows, dtype=np.float32)


class StructuredMLPScheduler(nn.Module):
    def __init__(self, feature_dim=10, hidden_dim=64, action_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, features):
        return torch.softmax(self.net(features), dim=-1)


class StructuredMLPPPO:
    """Inference wrapper compatible with the ablation scheduler interface."""

    def __init__(self, checkpoint="", hidden_dim=64):
        self.policy = StructuredMLPScheduler(hidden_dim=hidden_dim).to(device)
        if checkpoint:
            self.load(checkpoint)
        self.policy.eval()

    def load(self, checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.policy.load_state_dict(state)

    def select_agent_strategy_profiles(self, env, deterministic=True):
        features = torch.from_numpy(agent_feature_matrix(env)).float().to(device)
        with torch.no_grad():
            probabilities = self.policy(features).detach().cpu().numpy()
        return {
            agent: probabilities[idx].tolist()
            for idx, agent in enumerate(env.agents)
        }

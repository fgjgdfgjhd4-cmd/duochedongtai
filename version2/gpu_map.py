import math
from dataclasses import dataclass

import torch

from gpu_config import DTYPE, get_device


@dataclass
class StepResult:
    reward: float
    done: bool
    collision: bool
    path_distance: float


class GpuMapEnv:
    """GPU-first map environment for batched candidate-action evaluation.

    State tensors stay on `device`. Candidate actions use shape [B, N, 2],
    where the last dimension is [speed, delta_heading].
    """

    def __init__(self, agent_num=10, map_index=2, device=None):
        self.agent_num = agent_num
        self.device = device or get_device()
        self.map_index = map_index
        self.screen_width = 800.0
        self.goal_range = 5.0
        self.vehicle_size = 10.0
        self.safe_distance = self.vehicle_size * 2.0 * math.sqrt(2.0)
        self.time_delta = 0.5
        self.agents = [f"agent_{idx}" for idx in range(agent_num)]
        self._load_map(map_index)
        self.reset()

    def _tensor(self, value):
        return torch.as_tensor(value, dtype=DTYPE, device=self.device)

    def _load_map(self, map_index):
        if map_index == 0:
            circles = [
                [400, 400], [400, 670], [400, 130], [130, 400], [670, 400],
                [130, 130], [130, 670], [670, 130], [670, 670], [265, 265],
                [535, 535], [265, 535], [535, 265],
            ]
            radii = [50] * 13
            rects = []
        elif map_index == 1:
            circles = [[280, 280], [520, 520], [280, 520], [520, 280]]
            radii = [45] * 4
            rects = [
                [90, 400, 80, 250], [710, 400, 80, 250],
                [400, 710, 300, 80], [400, 90, 300, 80],
            ]
        else:
            circles = [
                [100, 650], [100, 700], [150, 700], [650, 300], [650, 250],
                [600, 250], [110, 250], [180, 320], [250, 320], [650, 650],
                [700, 600], [600, 700], [540, 700], [480, 700], [250, 520],
                [280, 550], [310, 520], [430, 130], [370, 130], [400, 100],
            ]
            radii = [50, 50, 50, 40, 40, 40, 70, 70, 70, 60, 60, 60, 60, 60, 40, 40, 40, 40, 40, 40]
            rects = []

        self.circle_centers = self._tensor(circles).reshape(-1, 2)
        self.circle_radii = self._tensor(radii).reshape(-1)
        self.rects = self._tensor(rects).reshape(-1, 4) if rects else self._tensor([]).reshape(0, 4)

    def reset(self, seed=42, starts=None, targets=None):
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        self.positions = torch.empty((self.agent_num, 2), device=self.device, dtype=DTYPE)
        self.destinations = torch.empty((self.agent_num, 2), device=self.device, dtype=DTYPE)
        self.orientation = torch.zeros(self.agent_num, device=self.device, dtype=DTYPE)
        self.terminated = torch.zeros(self.agent_num, device=self.device, dtype=torch.bool)
        self.collided = torch.zeros(self.agent_num, device=self.device, dtype=torch.bool)

        if starts is not None and targets is not None:
            self.positions = self._tensor(starts).reshape(self.agent_num, 2)
            self.destinations = self._tensor(targets).reshape(self.agent_num, 2)
            return self.state()

        self.positions = self._sample_points(generator)
        self.destinations = self._sample_points(generator)
        return self.state()

    def _sample_points(self, generator):
        points = []
        attempts = 0
        while len(points) < self.agent_num:
            attempts += 1
            if attempts > 20000:
                raise RuntimeError("Could not sample non-colliding points.")
            point = torch.randint(50, 751, (2,), generator=generator, device=self.device, dtype=torch.int64).to(DTYPE)
            if self._point_hits_obstacle(point):
                continue
            if points:
                existing = torch.stack(points)
                if torch.linalg.norm(existing - point, dim=1).min() < 50:
                    continue
            points.append(point)
        return torch.stack(points)

    def _point_hits_obstacle(self, point):
        if self.circle_centers.numel() > 0:
            circle_dist = torch.linalg.norm(self.circle_centers - point, dim=1)
            if torch.any(circle_dist <= self.circle_radii + self.vehicle_size + self.goal_range):
                return True
        if self.rects.numel() > 0:
            centers = self.rects[:, :2]
            half = self.rects[:, 2:] / 2.0
            inside = torch.all(torch.abs(centers - point) <= half + self.vehicle_size + self.goal_range, dim=1)
            if torch.any(inside):
                return True
        return False

    def state(self):
        return torch.cat([self.positions, self.orientation[:, None]], dim=1)

    def _next_state_from_actions(self, actions):
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        speeds = actions[..., 0].clamp_min(0.0)
        turns = actions[..., 1]
        orientation = self.orientation[None, :] + turns
        delta = torch.stack([torch.cos(orientation), torch.sin(orientation)], dim=-1) * speeds[..., None] * self.time_delta
        next_positions = (self.positions[None, :, :] + delta).clamp(5.0, self.screen_width - 5.0)
        return next_positions, orientation, delta

    def evaluate_batch(self, actions):
        next_positions, next_orientation, _ = self._next_state_from_actions(actions)
        active = (~self.terminated & ~self.collided).to(DTYPE)

        old_goal_dist = torch.linalg.norm(self.destinations - self.positions, dim=1)
        new_goal_dist = torch.linalg.norm(self.destinations[None, :, :] - next_positions, dim=2)
        target_cost = ((new_goal_dist - old_goal_dist[None, :]) * active[None, :]).sum(dim=1)

        pairwise = torch.cdist(next_positions, next_positions)
        eye = torch.eye(self.agent_num, dtype=torch.bool, device=self.device)[None, :, :]
        pairwise = pairwise.masked_fill(eye, float("inf"))
        robot_clearance = pairwise.min(dim=2).values
        robot_penalty = torch.relu(self.safe_distance - robot_clearance).sum(dim=1) * 50.0

        obstacle_penalty = torch.zeros(next_positions.shape[0], device=self.device, dtype=DTYPE)
        if self.circle_centers.numel() > 0:
            circle_dist = torch.cdist(next_positions, self.circle_centers[None, :, :].expand(next_positions.shape[0], -1, -1))
            clearance = circle_dist - self.circle_radii[None, None, :] - self.vehicle_size / 2.0
            obstacle_penalty = obstacle_penalty + torch.relu(self.safe_distance - clearance).sum(dim=(1, 2)) * 20.0
            obstacle_penalty = obstacle_penalty + (clearance <= 0).to(DTYPE).sum(dim=(1, 2)) * 1000.0

        if self.rects.numel() > 0:
            centers = self.rects[:, :2]
            half = self.rects[:, 2:] / 2.0 + self.vehicle_size / 2.0
            diff = torch.abs(next_positions[:, :, None, :] - centers[None, None, :, :]) - half[None, None, :, :]
            outside = torch.relu(diff)
            outside_dist = torch.linalg.norm(outside, dim=-1)
            inside = torch.all(diff <= 0, dim=-1)
            obstacle_penalty = obstacle_penalty + torch.relu(self.safe_distance - outside_dist).sum(dim=(1, 2)) * 20.0
            obstacle_penalty = obstacle_penalty + inside.to(DTYPE).sum(dim=(1, 2)) * 1000.0

        border_hit = torch.any((next_positions <= self.vehicle_size / 2.0) | (next_positions >= self.screen_width - self.vehicle_size / 2.0), dim=2)
        border_penalty = border_hit.to(DTYPE).sum(dim=1) * 1000.0

        return target_cost + robot_penalty + obstacle_penalty + border_penalty

    def step(self, actions):
        actions = actions.to(self.device, dtype=DTYPE)
        next_positions, next_orientation, delta = self._next_state_from_actions(actions)
        next_positions = next_positions.squeeze(0)
        next_orientation = next_orientation.squeeze(0)
        delta = delta.squeeze(0)

        active = ~self.terminated & ~self.collided
        old_vector = self.destinations - self.positions
        self.positions = torch.where(active[:, None], next_positions, self.positions)
        self.orientation = torch.where(active, next_orientation, self.orientation)

        reward = torch.zeros(self.agent_num, device=self.device, dtype=DTYPE)
        reached = torch.linalg.norm(self.positions - self.destinations, dim=1) <= self.goal_range
        self.terminated |= reached
        reward += reached.to(DTYPE) * 100.0

        if self.agent_num > 1:
            pairwise = torch.cdist(self.positions[None, :, :], self.positions[None, :, :]).squeeze(0)
            pairwise = pairwise.masked_fill(torch.eye(self.agent_num, dtype=torch.bool, device=self.device), float("inf"))
            robot_hit = pairwise.min(dim=1).values < self.safe_distance / 2.0
            self.collided |= robot_hit

        border_hit = torch.any((self.positions <= self.vehicle_size / 2.0) | (self.positions >= self.screen_width - self.vehicle_size / 2.0), dim=1)
        self.collided |= border_hit

        if self.circle_centers.numel() > 0:
            circle_dist = torch.cdist(self.positions[None, :, :], self.circle_centers[None, :, :]).squeeze(0)
            circle_hit = torch.any(circle_dist <= self.circle_radii[None, :] + self.vehicle_size / 2.0, dim=1)
            self.collided |= circle_hit

        if self.rects.numel() > 0:
            centers = self.rects[:, :2]
            half = self.rects[:, 2:] / 2.0 + self.vehicle_size / 2.0
            rect_hit = torch.all(torch.abs(self.positions[:, None, :] - centers[None, :, :]) <= half[None, :, :], dim=2).any(dim=1)
            self.collided |= rect_hit

        reward += self.collided.to(DTYPE) * -1000.0
        toward = (old_vector * delta).gt(0).to(DTYPE).sum(dim=1) * 2.0
        reward += torch.where(active, toward - 10.0, torch.zeros_like(reward))

        done = bool(torch.all(self.terminated).detach().cpu())
        collision = bool(torch.any(self.collided).detach().cpu())
        path_distance = float(torch.linalg.norm(delta, dim=1).sum().detach().cpu())
        return StepResult(float(reward.sum().detach().cpu()), done, collision, path_distance)

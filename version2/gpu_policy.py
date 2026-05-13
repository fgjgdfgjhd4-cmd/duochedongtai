import os
import sys

import torch

VERSION2_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(VERSION2_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class GpuPpoStrategyScheduler:
    """Loads the existing PPO policy and feeds it a GPU-generated 4-channel map."""

    def __init__(self, checkpoint_path, device):
        from CNN_PPO import PPO

        self.device = device
        self.ppo = PPO(
            action_dim=3,
            lr_actor=0.0003,
            lr_critic=0.001,
            gamma=0.99,
            K_epochs=80,
            eps_clip=0.2,
            has_continuous_action_space=False,
        )
        if checkpoint_path:
            self.ppo.load(checkpoint_path)
        self.ppo.policy_old.to(device)
        self.ppo.policy.to(device)
        self.ppo.policy_old.eval()

    def probabilities(self, env, deterministic=True):
        state = rasterize_env_state(env).to(self.device)
        probabilities, action_index, raw_probs = self.ppo.select_strategy_profile(state, deterministic=deterministic)
        self.ppo.buffer.clear()
        return probabilities, action_index, raw_probs


def rasterize_env_state(env):
    """Create a [1, 4, 800, 800] tensor state without pygame/cv2.

    Channels are obstacle, target, agent-position, and collision/termination mask.
    It is functionally compatible with the PPO CNN input shape, but not pixel-identical
    to the original pygame HSV render.
    """

    height = width = int(env.screen_width)
    state = torch.zeros((1, 4, height, width), dtype=torch.float32, device=env.device)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=env.device),
        torch.arange(width, device=env.device),
        indexing="ij",
    )
    grid = torch.stack([xx, yy], dim=-1).to(torch.float32)

    if env.circle_centers.numel() > 0:
        for center, radius in zip(env.circle_centers, env.circle_radii):
            mask = torch.linalg.norm(grid - center, dim=-1) <= radius
            state[0, 0, mask] = 255.0

    if env.rects.numel() > 0:
        for rect in env.rects:
            center = rect[:2]
            half = rect[2:] / 2.0
            mask = torch.all(torch.abs(grid - center) <= half, dim=-1)
            state[0, 0, mask] = 255.0

    for point in env.destinations:
        x = int(torch.clamp(point[0], 0, width - 1).detach().cpu())
        y = int(torch.clamp(point[1], 0, height - 1).detach().cpu())
        state[0, 1, max(0, y - 6): min(height, y + 7), max(0, x - 6): min(width, x + 7)] = 255.0

    for idx, point in enumerate(env.positions):
        x = int(torch.clamp(point[0], 0, width - 1).detach().cpu())
        y = int(torch.clamp(point[1], 0, height - 1).detach().cpu())
        state[0, 2, max(0, y - 5): min(height, y + 6), max(0, x - 5): min(width, x + 6)] = 127.5
        if bool((env.collided[idx] | env.terminated[idx]).detach().cpu()):
            state[0, 3, max(0, y - 5): min(height, y + 6), max(0, x - 5): min(width, x + 6)] = 255.0

    return state

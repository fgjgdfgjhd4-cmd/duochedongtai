import argparse
import csv
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

VERSION2_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(VERSION2_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from CNN_PPO import PPO
from map import Map
from gpu_config import get_device, set_seed
from gpu_map import GpuMapEnv
from gpu_policy import rasterize_env_state


@dataclass
class FidelityRow:
    map_index: int
    agent_num: int
    episode: int
    legacy_action: int
    gpu_action: int
    action_match: bool
    legacy_prob_0: float
    legacy_prob_1: float
    legacy_prob_2: float
    gpu_prob_0: float
    gpu_prob_1: float
    gpu_prob_2: float
    probability_l1: float
    probability_l2: float
    probability_kl_legacy_to_gpu: float
    pixel_mae: float
    pixel_rmse: float
    pixel_cosine: float


def parse_args():
    parser = argparse.ArgumentParser(description="Compare legacy PPO render state with version2 GPU raster state.")
    parser.add_argument("--agent-nums", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--map-indices", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="state_fidelity_results")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def scenario_seed(base_seed, map_index, agent_num, episode):
    return base_seed + map_index * 10000 + agent_num * 100 + episode


def build_ppo(checkpoint, device):
    ppo = PPO(
        action_dim=3,
        lr_actor=0.0003,
        lr_critic=0.001,
        gamma=0.99,
        K_epochs=80,
        eps_clip=0.2,
        has_continuous_action_space=False,
    )
    ppo.load(checkpoint)
    ppo.policy_old.to(device)
    ppo.policy.to(device)
    ppo.policy_old.eval()
    return ppo


def legacy_state_from_env(env, device):
    frame = env.render()
    state = np.transpose(frame, (2, 0, 1))
    state = np.expand_dims(state, axis=0)
    return torch.from_numpy(state).float().to(device)


def probabilities_for_state(ppo, state):
    with torch.no_grad():
        probs = ppo.policy_old.actor(state)
    probs = probs.detach().cpu().reshape(-1).float()
    action = int(torch.argmax(probs).item())
    return probs, action


def tensor_stats(legacy_state, gpu_state):
    legacy = legacy_state.detach().float().cpu()
    gpu = gpu_state.detach().float().cpu()
    diff = legacy - gpu
    mae = float(diff.abs().mean().item())
    rmse = float(torch.sqrt(torch.mean(diff * diff)).item())
    cosine = float(F.cosine_similarity(legacy.reshape(1, -1), gpu.reshape(1, -1)).item())
    return mae, rmse, cosine


def make_legacy_points(starts, targets, agent_num):
    starts_cpu = starts.detach().cpu().numpy()
    targets_cpu = targets.detach().cpu().numpy()
    start_dict = {}
    target_dict = {}
    for idx in range(agent_num):
        agent = f"agent_{idx}"
        start_dict[agent] = starts_cpu[idx].copy()
        target_dict[agent] = targets_cpu[idx].copy()
    return start_dict, target_dict


def validate_one(args, ppo, device, map_index, agent_num, episode):
    seed = scenario_seed(args.seed, map_index, agent_num, episode)
    gpu_env = GpuMapEnv(agent_num=agent_num, map_index=map_index, device=device)
    gpu_env.reset(seed=seed)

    starts = gpu_env.positions.detach().clone()
    targets = gpu_env.destinations.detach().clone()
    start_dict, target_dict = make_legacy_points(starts, targets, agent_num)

    legacy_env = Map(agent_num=agent_num, render_mode="rgb_array", test_result_save=False)
    legacy_env.reset(seed=seed, map_index=map_index, starting_points=start_dict, targets=target_dict)

    legacy_state = legacy_state_from_env(legacy_env, device)
    gpu_state = rasterize_env_state(gpu_env).to(device)
    legacy_probs, legacy_action = probabilities_for_state(ppo, legacy_state)
    gpu_probs, gpu_action = probabilities_for_state(ppo, gpu_state)
    ppo.buffer.clear()

    eps = 1e-8
    probability_l1 = float(torch.sum(torch.abs(legacy_probs - gpu_probs)).item())
    probability_l2 = float(torch.linalg.norm(legacy_probs - gpu_probs).item())
    probability_kl = float(torch.sum(legacy_probs * torch.log((legacy_probs + eps) / (gpu_probs + eps))).item())
    pixel_mae, pixel_rmse, pixel_cosine = tensor_stats(legacy_state, gpu_state)

    if hasattr(legacy_env, "close"):
        legacy_env.close()

    return FidelityRow(
        map_index=map_index,
        agent_num=agent_num,
        episode=episode,
        legacy_action=legacy_action,
        gpu_action=gpu_action,
        action_match=legacy_action == gpu_action,
        legacy_prob_0=float(legacy_probs[0].item()),
        legacy_prob_1=float(legacy_probs[1].item()),
        legacy_prob_2=float(legacy_probs[2].item()),
        gpu_prob_0=float(gpu_probs[0].item()),
        gpu_prob_1=float(gpu_probs[1].item()),
        gpu_prob_2=float(gpu_probs[2].item()),
        probability_l1=probability_l1,
        probability_l2=probability_l2,
        probability_kl_legacy_to_gpu=probability_kl,
        pixel_mae=pixel_mae,
        pixel_rmse=pixel_rmse,
        pixel_cosine=pixel_cosine,
    )


def write_rows(output_dir, rows):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "state_fidelity.csv")
    fieldnames = list(FidelityRow.__dataclass_fields__.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return path


def print_summary(rows):
    if not rows:
        print("No rows produced.")
        return
    action_match_rate = sum(row.action_match for row in rows) / len(rows)
    mean_l1 = sum(row.probability_l1 for row in rows) / len(rows)
    mean_l2 = sum(row.probability_l2 for row in rows) / len(rows)
    mean_kl = sum(row.probability_kl_legacy_to_gpu for row in rows) / len(rows)
    mean_mae = sum(row.pixel_mae for row in rows) / len(rows)
    mean_cosine = sum(row.pixel_cosine for row in rows) / len(rows)
    print(f"rows={len(rows)}")
    print(f"action_match_rate={action_match_rate:.3f}")
    print(f"mean_probability_l1={mean_l1:.6f}")
    print(f"mean_probability_l2={mean_l2:.6f}")
    print(f"mean_probability_kl_legacy_to_gpu={mean_kl:.6f}")
    print(f"mean_pixel_mae={mean_mae:.3f}")
    print(f"mean_pixel_cosine={mean_cosine:.6f}")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(prefer_cuda=not args.cpu)
    ppo = build_ppo(args.checkpoint, device)
    output_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    rows = []

    print(f"device={device}", flush=True)
    for agent_num in args.agent_nums:
        for map_index in args.map_indices:
            for episode in range(args.episodes):
                row = validate_one(args, ppo, device, map_index, agent_num, episode)
                rows.append(row)
                print(
                    f"map={map_index} agents={agent_num} episode={episode} "
                    f"legacy_action={row.legacy_action} gpu_action={row.gpu_action} "
                    f"l1={row.probability_l1:.6f} pixel_mae={row.pixel_mae:.3f}",
                    flush=True,
                )

    csv_path = write_rows(output_dir, rows)
    print_summary(rows)
    print(f"State fidelity results saved to: {csv_path}")


if __name__ == "__main__":
    main()

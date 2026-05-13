import argparse
import csv
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime

import torch

from gpu_config import get_device, set_seed
from gpu_map import GpuMapEnv
from gpu_optimizers import GpuABCL, GpuCPSO, GpuIABC, GpuOriginABC, GpuPSO
from gpu_policy import GpuPpoStrategyScheduler


COMPARE_ALGORITHMS = ("ppo_abc", "abc", "pso", "cpso", "abcl")
ABLATION_ALGORITHMS = (
    "origin_abc",
    "fixed_iabc",
    "manual_iabc",
    "ppo_iabc",
    "no_spiral_iabc",
    "no_guided_iabc",
    "no_diff_iabc",
)
ALL_ALGORITHMS = COMPARE_ALGORITHMS + ABLATION_ALGORITHMS
OPTIMIZERS = {
    "ppo_abc": GpuIABC,
    "abc": GpuIABC,
    "pso": GpuPSO,
    "cpso": GpuCPSO,
    "abcl": GpuABCL,
    "origin_abc": GpuOriginABC,
    "fixed_iabc": GpuIABC,
    "manual_iabc": GpuIABC,
    "ppo_iabc": GpuIABC,
    "no_spiral_iabc": GpuIABC,
    "no_guided_iabc": GpuIABC,
    "no_diff_iabc": GpuIABC,
}


@dataclass
class EpisodeRow:
    suite: str
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
    device: str


def parse_args():
    parser = argparse.ArgumentParser(description="Run all GPU-refactor supported experiments.")
    parser.add_argument("--suite", choices=["compare", "ablation", "all"], default="all")
    parser.add_argument("--algorithms", nargs="+", choices=ALL_ALGORITHMS, default=None)
    parser.add_argument("--agent-nums", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--map-indices", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-ep-len", type=int, default=200)
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output-dir", default="version2_results")
    parser.add_argument("--resume-dir", default="")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def scenario_seed(base_seed, map_index, agent_num, episode):
    return base_seed + map_index * 10000 + agent_num * 100 + episode


def default_algorithms(suite):
    if suite == "compare":
        return list(COMPARE_ALGORITHMS)
    if suite == "ablation":
        return list(ABLATION_ALGORITHMS)
    return list(ALL_ALGORITHMS)


def manual_probabilities(env):
    active = ~(env.terminated | env.collided)
    if not bool(torch.any(active).detach().cpu()):
        return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

    active_positions = env.positions[active]
    active_targets = env.destinations[active]
    distances = torch.linalg.norm(active_positions - active_targets, dim=1)
    min_pair_distance = torch.tensor(float("inf"), device=env.device)
    if active_positions.shape[0] > 1:
        pairwise = torch.cdist(active_positions[None, :, :], active_positions[None, :, :]).squeeze(0)
        pairwise = pairwise.masked_fill(torch.eye(active_positions.shape[0], dtype=torch.bool, device=env.device), float("inf"))
        min_pair_distance = pairwise.min()

    if float(min_pair_distance.detach().cpu()) < env.safe_distance:
        return [0.15, 0.65, 0.20]
    if float(distances.mean().detach().cpu()) > 250.0:
        return [0.55, 0.25, 0.20]
    return [0.25, 0.25, 0.50]


def strategy_probabilities(algorithm, env, ppo_scheduler):
    if algorithm == "fixed_iabc":
        return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    if algorithm == "no_spiral_iabc":
        return [0.0, 0.5, 0.5]
    if algorithm == "no_guided_iabc":
        return [0.5, 0.0, 0.5]
    if algorithm == "no_diff_iabc":
        return [0.5, 0.5, 0.0]
    if algorithm == "manual_iabc":
        return manual_probabilities(env)
    if algorithm in ("ppo_abc", "ppo_iabc"):
        if ppo_scheduler is None:
            raise ValueError(f"{algorithm} requires --checkpoint")
        probabilities, _, _ = ppo_scheduler.probabilities(env, deterministic=(algorithm == "ppo_abc"))
        return probabilities
    return None


def suite_for_algorithm(algorithm):
    return "ablation" if algorithm in ABLATION_ALGORITHMS else "compare"


def run_episode(args, algorithm, map_index, agent_num, episode, starts, targets, device, ppo_scheduler):
    env = GpuMapEnv(agent_num=agent_num, map_index=map_index, device=device)
    env.reset(seed=scenario_seed(args.seed, map_index, agent_num, episode), starts=starts, targets=targets)
    optimizer_cls = OPTIMIZERS[algorithm]

    total_reward = 0.0
    total_path_distance = 0.0
    step_times = []
    timeout = True
    start_time = time.perf_counter()

    for step in range(1, args.max_ep_len + 1):
        step_start = time.perf_counter()
        probabilities = strategy_probabilities(algorithm, env, ppo_scheduler)
        if probabilities is None:
            optimizer = optimizer_cls(env, population=args.population, iterations=args.iterations)
        else:
            optimizer = optimizer_cls(env, population=args.population, iterations=args.iterations, probabilities=probabilities)
        actions = optimizer.optimize()
        result = env.step(actions)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - step_start)
        total_reward += result.reward
        total_path_distance += result.path_distance

        if result.done:
            timeout = False
            break
        if result.collision:
            timeout = False
            break

    success = bool(torch.all(env.terminated).detach().cpu())
    collision = bool(torch.any(env.collided).detach().cpu())
    elapsed = time.perf_counter() - start_time
    return EpisodeRow(
        suite=suite_for_algorithm(algorithm),
        algorithm=algorithm,
        map_index=map_index,
        agent_num=agent_num,
        episode=episode,
        success=success,
        collision=collision,
        timeout=timeout,
        steps=step,
        reward=total_reward,
        path_distance=total_path_distance,
        elapsed_time=elapsed,
        avg_step_time=sum(step_times) / len(step_times),
        device=str(device),
    )


def make_scenarios(args, device):
    scenarios = []
    for agent_num in args.agent_nums:
        for map_index in args.map_indices:
            for episode in range(args.episodes):
                env = GpuMapEnv(agent_num=agent_num, map_index=map_index, device=device)
                env.reset(seed=scenario_seed(args.seed, map_index, agent_num, episode))
                scenarios.append(
                    (
                        agent_num,
                        map_index,
                        episode,
                        env.positions.detach().clone(),
                        env.destinations.detach().clone(),
                    )
                )
    return scenarios


def load_existing(output_dir):
    path = os.path.join(output_dir, "episodes.csv")
    rows = []
    completed = set()
    if not os.path.exists(path):
        return rows, completed
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row_obj = EpisodeRow(
                suite=row["suite"],
                algorithm=row["algorithm"],
                map_index=int(row["map_index"]),
                agent_num=int(row["agent_num"]),
                episode=int(row["episode"]),
                success=row["success"] == "True",
                collision=row["collision"] == "True",
                timeout=row["timeout"] == "True",
                steps=int(row["steps"]),
                reward=float(row["reward"]),
                path_distance=float(row["path_distance"]),
                elapsed_time=float(row["elapsed_time"]),
                avg_step_time=float(row["avg_step_time"]),
                device=row["device"],
            )
            rows.append(row_obj)
            completed.add((row_obj.suite, row_obj.algorithm, row_obj.map_index, row_obj.agent_num, row_obj.episode))
    return rows, completed


def write_outputs(output_dir, rows):
    os.makedirs(output_dir, exist_ok=True)
    episodes_path = os.path.join(output_dir, "episodes.csv")
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(EpisodeRow.__dataclass_fields__.keys())
    with open(episodes_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.suite, row.algorithm, row.map_index, row.agent_num)].append(row)

    summary_path = os.path.join(output_dir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "suite",
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
            "mean_step_time",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (suite, algorithm, map_index, agent_num), group in sorted(grouped.items()):
            n = len(group)
            writer.writerow(
                {
                    "suite": suite,
                    "algorithm": algorithm,
                    "map_index": map_index,
                    "agent_num": agent_num,
                    "episodes": n,
                    "success_rate": sum(row.success for row in group) / n,
                    "collision_rate": sum(row.collision for row in group) / n,
                    "timeout_rate": sum(row.timeout for row in group) / n,
                    "mean_reward": sum(row.reward for row in group) / n,
                    "mean_steps": sum(row.steps for row in group) / n,
                    "mean_path_distance": sum(row.path_distance for row in group) / n,
                    "mean_elapsed_time": sum(row.elapsed_time for row in group) / n,
                    "mean_step_time": sum(row.avg_step_time for row in group) / n,
                }
            )
    return episodes_path, summary_path


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(prefer_cuda=not args.cpu)
    algorithms = args.algorithms or default_algorithms(args.suite)
    needs_ppo = any(algorithm in ("ppo_abc", "ppo_iabc") for algorithm in algorithms)
    if needs_ppo and not args.checkpoint:
        raise ValueError("--checkpoint is required for ppo_abc or ppo_iabc")
    ppo_scheduler = GpuPpoStrategyScheduler(args.checkpoint, device) if needs_ppo else None
    output_dir = args.resume_dir or os.path.join(args.output_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    rows, completed = load_existing(output_dir)
    scenarios = make_scenarios(args, device)

    print(f"device={device}", flush=True)
    print(f"algorithms={' '.join(algorithms)}", flush=True)

    for algorithm in algorithms:
        for agent_num, map_index, episode, starts, targets in scenarios:
            key = (suite_for_algorithm(algorithm), algorithm, map_index, agent_num, episode)
            if key in completed:
                print(f"SKIP {algorithm} map={map_index} agents={agent_num} episode={episode}", flush=True)
                continue
            print(f"RUN  {algorithm} map={map_index} agents={agent_num} episode={episode}", flush=True)
            row = run_episode(args, algorithm, map_index, agent_num, episode, starts, targets, device, ppo_scheduler)
            rows.append(row)
            status = "SUCCESS" if row.success else "COLLISION" if row.collision else "TIMEOUT"
            print(
                f"  {status:9s} steps={row.steps:3d} reward={row.reward:9.2f} time={row.elapsed_time:7.3f}s",
                flush=True,
            )
            write_outputs(output_dir, rows)

    episodes_path, summary_path = write_outputs(output_dir, rows)
    print(f"Episode results saved to: {episodes_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

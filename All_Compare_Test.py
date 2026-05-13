#!/usr/bin/env python3
"""Algorithm comparison test: PPO-ABC vs ABC vs PSO vs CPSO vs ABCL.

Refactored to support CLI arguments, multiple maps, multiple episodes,
and CSV output using experiment_utils.
"""

import argparse
import csv
import math
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from itertools import cycle, islice

import numpy as np
import torch

from abc_for_map import ABC
from CNN_PPO import PPO
from Compare_Method_ABC.swarm import ABC_origin
from Compare_Method_PSO.PSO_map import PSO as PSOAlgo
from CPSO.CPSO_map import CPSO
from Compare_Method_ABCL.ABCL_MAP import ABCL
from experiment_utils import (
    EpisodeResult,
    expand_bounds,
    sample_start_targets,
    solution_to_actions,
    summarize_results,
    write_episode_results,
    write_summary,
)
from map import Map


ALGORITHMS = ["ppo_abc", "abc", "pso", "cpso", "abcl"]


def build_optimizer(name, env, fn_lb, fn_ub, population, runs):
    """Create the appropriate optimizer for each algorithm."""
    if name == "abc":
        return ABC_origin(
            n_population=population,
            n_runs=runs,
            fn_eval=env.objective,
            fn_lb=fn_lb,
            fn_ub=fn_ub,
            env=env,
        )
    elif name == "pso":
        return PSOAlgo(
            dimension=env.agent_num * 2,
            time=runs,
            size=population,
            fn_lb=fn_lb,
            fn_ub=fn_ub,
            v_low=-3,
            v_high=3,
            env=env,
        )
    elif name == "cpso":
        return CPSO(
            dimension=env.agent_num * 2,
            generation=runs,
            size=population,
            fn_lb=fn_lb,
            fn_ub=fn_ub,
            v_low=-3,
            v_high=3,
            env=env,
        )
    elif name == "abcl":
        return ABCL(
            npopulation=population,
            nruns=runs,
            fn_eval=env.objective,
            fn_lb=[5, -math.pi / 4],
            fn_ub=[20, math.pi / 4],
            env=env,
        )
    else:
        return ABC(
            npopulation=population,
            nruns=runs,
            fn_eval=env.objective,
            fn_lb=fn_lb,
            fn_ub=fn_ub,
            env=env,
        )


def render_state(env):
    frame = env.render()
    state = np.transpose(frame, (2, 0, 1))
    state = np.expand_dims(state, axis=0)
    return torch.from_numpy(state).float()


def run_episode(args, algorithm, episode_index, map_index, agent_num, starts, targets, ppo_agent):
    """Run a single episode for a given algorithm."""
    env = Map(
        agent_num=agent_num,
        render_mode=args.render_mode,
        test_result_save=args.save_images,
    )
    _, _ = env.reset(
        seed=args.seed + episode_index,
        map_index=map_index,
        starting_points=starts,
        targets=targets,
    )

    fn_lb, fn_ub = expand_bounds(agent_num, args.lower_bound, args.upper_bound)
    optimizer = build_optimizer(algorithm, env, fn_lb, fn_ub, args.population, args.runs)

    state = render_state(env) if algorithm == "ppo_abc" else None
    total_reward = 0.0
    total_path_distance = 0.0
    step_times = []
    success = False
    timeout = True
    start_time = time.time()

    for step in range(1, args.max_ep_len + 1):
        step_start = time.time()

        if algorithm == "ppo_abc" and state is not None:
            probabilities, _, _ = ppo_agent.select_strategy_profile(state, deterministic=True)
            optimizer.set_probability(probabilities)

        solution = optimizer.optimize()
        actions = solution_to_actions(solution, env.agents)
        _, reward, done, _, _, path_distance = env.step(actions)

        total_reward += float(reward)
        total_path_distance += float(path_distance)
        step_times.append(time.time() - step_start)

        if algorithm == "ppo_abc":
            if done:
                ppo_agent.buffer.rewards.append(float(reward))
                ppo_agent.buffer.is_terminals.append(done)
            else:
                state = render_state(env)
                ppo_agent.buffer.rewards.append(float(reward))
                ppo_agent.buffer.is_terminals.append(done)

        if done:
            success = True
            timeout = False
            break

        all_finished_or_failed = all(
            env.collisions[agent] or env.terminations[agent] or env.truncations[agent]
            for agent in env.agents
        )
        if all_finished_or_failed:
            timeout = False
            break

    if algorithm == "ppo_abc" and ppo_agent is not None:
        ppo_agent.buffer.clear()

    collision = any(env.collisions.values())
    elapsed = time.time() - start_time
    env.close()

    return EpisodeResult(
        algorithm=algorithm,
        map_index=map_index,
        agent_num=agent_num,
        episode=episode_index,
        success=success,
        collision=collision,
        timeout=timeout,
        steps=step,
        reward=total_reward,
        path_distance=total_path_distance,
        elapsed_time=elapsed,
        avg_step_time=float(np.mean(step_times)) if step_times else 0.0,
        probability_trace=[],
    )


def build_ppo(args):
    ppo = PPO(
        action_dim=3,
        lr_actor=0.0003,
        lr_critic=0.001,
        gamma=0.99,
        K_epochs=80,
        eps_clip=0.2,
        has_continuous_action_space=False,
    )
    if args.checkpoint:
        ppo.load(args.checkpoint)
    return ppo


def write_current_results(output_dir, results, grouped):
    episode_path = os.path.join(output_dir, "episodes.csv")
    write_episode_results(episode_path, results)

    summary_rows = []
    for (algorithm, map_index, agent_num), group_results in grouped.items():
        row = summarize_results(group_results)
        row.update({"algorithm": algorithm, "map_index": map_index, "agent_num": agent_num})
        summary_rows.append(row)

    summary_path = os.path.join(output_dir, "summary.csv")
    write_summary(summary_path, summary_rows)
    return episode_path, summary_path


def load_existing_results(output_dir):
    episode_path = os.path.join(output_dir, "episodes.csv")
    results = []
    grouped = defaultdict(list)
    completed = set()
    if not os.path.exists(episode_path):
        return results, grouped, completed

    with open(episode_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result = EpisodeResult(
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
                probability_trace=[],
            )
            results.append(result)
            key = (result.algorithm, result.map_index, result.agent_num)
            grouped[key].append(result)
            completed.add((result.algorithm, result.map_index, result.agent_num, result.episode))
    return results, grouped, completed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare PPO-ABC, ABC, PSO, CPSO, ABCL algorithms."
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=ALGORITHMS,
        choices=ALGORITHMS,
        help="Algorithms to test (default: all)",
    )
    parser.add_argument(
        "--agent-num",
        type=int,
        default=10,
        help="Number of agents/robots (default: 10)",
    )
    parser.add_argument(
        "--map-indices",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Map indices to test (default: 0 1 2)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes per algorithm per map (default: 10)",
    )
    parser.add_argument("--max-ep-len", type=int, default=200)
    parser.add_argument("--population", type=int, default=15)
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument(
        "--lower-bound",
        nargs=2,
        type=float,
        default=[5.0, -math.pi / 4],
    )
    parser.add_argument(
        "--upper-bound",
        nargs=2,
        type=float,
        default=[20.0, math.pi / 4],
    )
    parser.add_argument(
        "--render-mode",
        default="rgb_array",
        choices=["rgb_array", "human"],
        help="Render mode (default: rgb_array for headless)",
    )
    parser.add_argument(
        "--checkpoint",
        default="PPO_preTrained/Map/10_robots/0511-15-16PPO_Map_6000.pth",
        help="Path to PPO checkpoint",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="compare_results",
        help="Directory for CSV output",
    )
    parser.add_argument(
        "--resume-dir",
        default="",
        help="Existing timestamped output directory to resume from",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        default=False,
        help="Save test result images (requires pygame)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if "ppo_abc" in args.algorithms and not os.path.exists(args.checkpoint):
        print(f"WARNING: checkpoint not found: {args.checkpoint}")
        print("PPO-ABC will be skipped. Use --checkpoint to specify a valid path.")
        args.algorithms = [a for a in args.algorithms if a != "ppo_abc"]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = args.resume_dir or os.path.join(
        args.output_dir,
        datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    os.makedirs(output_dir, exist_ok=True)

    # Build PPO agent once if needed
    ppo_agent = build_ppo(args) if "ppo_abc" in args.algorithms else None

    results, grouped, completed = load_existing_results(output_dir)
    if completed:
        print(f"Resuming from {output_dir}: {len(completed)} episodes already complete", flush=True)

    for map_index in args.map_indices:
        print(f"\n{'='*60}", flush=True)
        print(f"Map {map_index}", flush=True)
        print(f"{'='*60}", flush=True)

        # Pre-sample start/target positions for reproducibility
        seed_env = Map(agent_num=args.agent_num, render_mode=None)
        seed_env.reset(seed=args.seed, map_index=map_index)

        scenarios = []
        for ep in range(args.episodes):
            starts, targets = sample_start_targets(
                seed_env,
                seed=args.seed + map_index * 10000 + ep,
            )
            scenarios.append((ep, starts, targets))
        seed_env.close()

        for algorithm in args.algorithms:
            print(f"\n--- {algorithm.upper()} ---", flush=True)
            algo_results = []

            for ep, starts, targets in scenarios:
                key = (algorithm, map_index, args.agent_num, ep)
                if key in completed:
                    print(f"  Episode {ep:3d}: SKIP existing result", flush=True)
                    continue
                result = run_episode(
                    args, algorithm, ep, map_index, args.agent_num,
                    starts, targets, ppo_agent,
                )
                results.append(result)
                algo_results.append(result)
                grouped[(algorithm, map_index, args.agent_num)].append(result)

                status = "SUCCESS" if result.success else (
                    "COLLISION" if result.collision else "TIMEOUT" if result.timeout else "FAIL"
                )
                print(
                    f"  Episode {ep:3d}: {status:9s}  "
                    f"steps={result.steps:3d}  "
                    f"reward={result.reward:8.2f}  "
                    f"time={result.elapsed_time:6.1f}s",
                    flush=True,
                )
                write_current_results(output_dir, results, grouped)

            # Print per-algorithm summary for this map
            if algo_results:
                summary = summarize_results(algo_results)
                print(
                    f"  SUMMARY: success_rate={summary['success_rate']:.2f}  "
                    f"mean_reward={summary['mean_reward']:.1f}  "
                    f"mean_steps={summary['mean_steps']:.1f}  "
                    f"mean_time={summary['mean_elapsed_time']:.1f}s",
                    flush=True,
                )

    # Print overall comparison
    print(f"\n{'='*60}")
    print("OVERALL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Algorithm':<12} {'Success':>8} {'Collision':>10} {'Timeout':>9} "
          f"{'MeanReward':>12} {'MeanSteps':>11} {'MeanTime':>10}")
    print("-" * 72)

    for algorithm in args.algorithms:
        algo_group = [r for r in results if r.algorithm == algorithm]
        if algo_group:
            s = summarize_results(algo_group)
            print(
                f"{algorithm:<12} {s['success_rate']:8.2f} {s['collision_rate']:10.2f} "
                f"{s['timeout_rate']:9.2f} {s['mean_reward']:12.1f} "
                f"{s['mean_steps']:11.1f} {s['mean_elapsed_time']:10.1f}"
            )

    # Write CSV output
    episode_path, summary_path = write_current_results(output_dir, results, grouped)
    print(f"\nEpisode results saved to: {episode_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

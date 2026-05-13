import argparse
import csv
import os
import random
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch

from abc_for_map import ABC
from CNN_PPO import PPO
from Compare_Method_ABC.swarm import ABC_origin
from experiment_utils import (
    EpisodeResult,
    expand_bounds,
    manual_probabilities,
    sample_start_targets,
    solution_to_actions,
    summarize_results,
    write_episode_results,
    write_summary,
)
from map import Map


ALGORITHMS = ("origin_abc", "fixed_iabc", "manual_iabc", "ppo_iabc")


def build_optimizer(algorithm, env, population, runs, fn_lb, fn_ub):
    if algorithm == "origin_abc":
        return ABC_origin(
            n_population=population,
            n_runs=runs,
            fn_eval=env.objective,
            fn_lb=fn_lb,
            fn_ub=fn_ub,
            env=env,
        )
    return ABC(
        npopulation=population,
        nruns=runs,
        fn_eval=env.objective,
        fn_lb=fn_lb,
        fn_ub=fn_ub,
        env=env,
    )


def build_ppo(args):
    ppo = PPO(
        args.action_dim,
        args.lr_actor,
        args.lr_critic,
        args.gamma,
        args.k_epochs,
        args.eps_clip,
        False,
        args.action_std,
    )
    if args.checkpoint:
        ppo.load(args.checkpoint)
    return ppo


def render_state(env):
    frame = env.render()
    state = np.transpose(frame, (2, 0, 1))
    state = np.expand_dims(state, axis=0)
    return torch.from_numpy(state).float()


def select_probabilities(algorithm, optimizer, ppo, state, env):
    if algorithm == "fixed_iabc":
        probabilities = [1 / 3, 1 / 3, 1 / 3]
    elif algorithm == "manual_iabc":
        probabilities = manual_probabilities(env)
    elif algorithm == "ppo_iabc":
        probabilities, _, _ = ppo.select_strategy_profile(state, deterministic=False)
    else:
        return None

    optimizer.set_probability(probabilities)
    return probabilities


def run_episode(args, algorithm, episode_index, map_index, agent_num, starts, targets):
    env = Map(agent_num=agent_num, render_mode=args.render_mode, test_result_save=False)
    _, _ = env.reset(
        seed=args.seed + episode_index,
        map_index=map_index,
        starting_points=starts,
        targets=targets,
    )

    fn_lb, fn_ub = expand_bounds(agent_num, args.lower_bound, args.upper_bound)
    optimizer = build_optimizer(algorithm, env, args.population, args.runs, fn_lb, fn_ub)
    ppo = build_ppo(args) if algorithm == "ppo_iabc" else None

    state = render_state(env) if algorithm == "ppo_iabc" else None
    total_reward = 0.0
    total_path_distance = 0.0
    step_times = []
    probability_trace = []
    success = False
    timeout = True
    start_time = time.time()

    for step in range(1, args.max_ep_len + 1):
        step_start = time.time()
        probabilities = select_probabilities(algorithm, optimizer, ppo, state, env)
        if probabilities is not None:
            probability_trace.append(probabilities)

        solution = optimizer.optimize()
        actions = solution_to_actions(solution, env.agents)
        _, reward, done, collisions, truncations, path_distance = env.step(actions)

        total_reward += float(reward)
        total_path_distance += float(path_distance)
        step_times.append(time.time() - step_start)

        if algorithm == "ppo_iabc":
            if done:
                ppo.buffer.rewards.append(float(reward))
                ppo.buffer.is_terminals.append(done)
            else:
                state = render_state(env)
                ppo.buffer.rewards.append(float(reward))
                ppo.buffer.is_terminals.append(done)

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

    if ppo is not None:
        ppo.buffer.clear()

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
        probability_trace=probability_trace,
    )


def make_scenarios(args):
    scenarios = []
    for agent_num in args.agent_nums:
        for map_index in args.map_indices:
            seed_env = Map(agent_num=agent_num, render_mode=None)
            seed_env.reset(seed=args.seed, map_index=map_index)
            for episode in range(args.episodes):
                starts, targets = sample_start_targets(
                    seed_env,
                    seed=args.seed + map_index * 10000 + agent_num * 100 + episode,
                )
                scenarios.append((agent_num, map_index, episode, starts, targets))
            seed_env.close()
    return scenarios


def parse_args():
    parser = argparse.ArgumentParser(description="Run standardized PPO-ABC ablation experiments.")
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS), choices=ALGORITHMS)
    parser.add_argument("--agent-nums", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--map-indices", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-ep-len", type=int, default=200)
    parser.add_argument("--population", type=int, default=15)
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--lower-bound", nargs=2, type=float, default=[0.0, -np.pi / 4])
    parser.add_argument("--upper-bound", nargs=2, type=float, default=[20.0, np.pi / 4])
    parser.add_argument("--render-mode", default="rgb_array", choices=["rgb_array", "human"])
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="ablation_results")
    parser.add_argument("--resume-dir", default="")

    parser.add_argument("--action-dim", type=int, default=3)
    parser.add_argument("--lr-actor", type=float, default=0.0003)
    parser.add_argument("--lr-critic", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--k-epochs", type=int, default=80)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--action-std", type=float, default=0.1)
    return parser.parse_args()


def write_current_results(output_dir, results, grouped):
    episode_path = os.path.join(output_dir, "episodes.csv")
    write_episode_results(episode_path, results)

    summary_rows = []
    for (algorithm, map_index, agent_num), group_results in grouped.items():
        row = summarize_results(group_results)
        row.update(
            {
                "algorithm": algorithm,
                "map_index": map_index,
                "agent_num": agent_num,
            }
        )
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


def main():
    args = parse_args()
    if "ppo_iabc" in args.algorithms and not args.checkpoint:
        raise ValueError("--checkpoint is required when running ppo_iabc.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = args.resume_dir or os.path.join(
        args.output_dir,
        datetime.now().strftime("%Y%m%d-%H%M%S"),
    )

    scenarios = make_scenarios(args)
    results, grouped, completed = load_existing_results(output_dir)
    if completed:
        print(f"Resuming from {output_dir}: {len(completed)} episodes already complete", flush=True)

    for algorithm in args.algorithms:
        for agent_num, map_index, episode, starts, targets in scenarios:
            key = (algorithm, map_index, agent_num, episode)
            if key in completed:
                print(
                    f"Skipping {algorithm}: map={map_index}, agents={agent_num}, episode={episode}",
                    flush=True,
                )
                continue
            print(
                f"Running {algorithm}: map={map_index}, agents={agent_num}, episode={episode}",
                flush=True,
            )
            result = run_episode(
                args,
                algorithm,
                episode,
                map_index,
                agent_num,
                starts,
                targets,
            )
            results.append(result)
            grouped[(algorithm, map_index, agent_num)].append(result)
            write_current_results(output_dir, results, grouped)

    episode_path, summary_path = write_current_results(output_dir, results, grouped)
    print(f"Episode results saved to: {episode_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

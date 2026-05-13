import argparse
import csv
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from experiment_utils import agent_manual_probabilities, sample_start_targets
from map import Map
from structured_mlp_ppo import StructuredMLPScheduler, agent_feature_matrix, device


def profile_matrix(env):
    profiles = agent_manual_probabilities(env)
    return np.array([profiles[agent] for agent in env.agents], dtype=np.float32)


def collect_distillation_batch(args, rng):
    feature_rows = []
    label_rows = []

    for _ in range(args.batch_scenarios):
        agent_num = int(rng.choice(args.agent_nums))
        map_index = int(rng.choice(args.map_indices))
        env = Map(agent_num=agent_num, render_mode=None)
        env.reset(seed=int(rng.integers(0, 1_000_000)), map_index=map_index)
        starts, targets = sample_start_targets(
            env,
            seed=int(rng.integers(0, 1_000_000)),
        )
        env.reset(
            seed=int(rng.integers(0, 1_000_000)),
            map_index=map_index,
            starting_points=starts,
            targets=targets,
        )

        feature_rows.append(agent_feature_matrix(env))
        label_rows.append(profile_matrix(env))
        env.close()

    features = np.concatenate(feature_rows, axis=0)
    labels = np.concatenate(label_rows, axis=0)
    return (
        torch.from_numpy(features).float().to(device),
        torch.from_numpy(labels).float().to(device),
    )


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    model = StructuredMLPScheduler(
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        action_dim=3,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train_log.csv")
    checkpoint_path = os.path.join(output_dir, "structured_mlp.pth")

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss", "kl", "mse"])
        writer.writeheader()

        model.train()
        for step in range(1, args.steps + 1):
            features, labels = collect_distillation_batch(args, rng)
            predictions = model(features)
            kl = F.kl_div(torch.log(predictions + 1e-8), labels, reduction="batchmean")
            mse = F.mse_loss(predictions, labels)
            loss = kl + args.mse_weight * mse

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            if step % args.log_freq == 0 or step == 1:
                row = {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "kl": float(kl.detach().cpu()),
                    "mse": float(mse.detach().cpu()),
                }
                writer.writerow(row)
                f.flush()
                print(
                    f"step={step} loss={row['loss']:.6f} "
                    f"kl={row['kl']:.6f} mse={row['mse']:.6f}"
                )

            if step % args.save_freq == 0 or step == args.steps:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "args": vars(args),
                        "step": step,
                    },
                    checkpoint_path,
                )

    print(f"Structured-MLP checkpoint saved to: {checkpoint_path}")
    print(f"Training log saved to: {log_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distill manual agent-aware IABC profiles into a Structured-MLP scheduler."
    )
    parser.add_argument("--agent-nums", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--map-indices", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-scenarios", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--feature-dim", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--mse-weight", type=float, default=0.25)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-freq", type=int, default=25)
    parser.add_argument("--save-freq", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="structured_mlp_checkpoints")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

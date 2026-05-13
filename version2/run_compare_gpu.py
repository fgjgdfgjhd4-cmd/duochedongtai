import argparse
import time

import torch

from gpu_config import get_device, set_seed
from gpu_map import GpuMapEnv
from gpu_optimizers import GpuABCL, GpuCPSO, GpuIABC, GpuPSO


OPTIMIZERS = {
    "abc": GpuIABC,
    "abcl": GpuABCL,
    "cpso": GpuCPSO,
    "pso": GpuPSO,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run GPU-first multi-robot optimization.")
    parser.add_argument("--algorithm", choices=sorted(OPTIMIZERS), default="pso")
    parser.add_argument("--agent-num", type=int, default=10)
    parser.add_argument("--map-index", type=int, default=2)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU for debugging.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(prefer_cuda=not args.cpu)
    env = GpuMapEnv(agent_num=args.agent_num, map_index=args.map_index, device=device)
    env.reset(seed=args.seed)
    optimizer_cls = OPTIMIZERS[args.algorithm]

    total_reward = 0.0
    total_distance = 0.0
    step_times = []
    start = time.perf_counter()

    for step in range(1, args.steps + 1):
        step_start = time.perf_counter()
        optimizer = optimizer_cls(env, population=args.population, iterations=args.iterations)
        actions = optimizer.optimize()
        result = env.step(actions)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - step_start)
        total_reward += result.reward
        total_distance += result.path_distance
        if result.done or result.collision:
            break

    elapsed = time.perf_counter() - start
    mean_step_time = sum(step_times) / len(step_times)
    print(f"device={device}")
    print(f"algorithm={args.algorithm}")
    print(f"agent_num={args.agent_num}")
    print(f"steps={step}")
    print(f"reward={total_reward:.2f}")
    print(f"path_distance={total_distance:.2f}")
    print(f"collision={bool(torch.any(env.collided).detach().cpu())}")
    print(f"success={bool(torch.all(env.terminated).detach().cpu())}")
    print(f"elapsed_sec={elapsed:.4f}")
    print(f"mean_step_sec={mean_step_time:.4f}")


if __name__ == "__main__":
    main()

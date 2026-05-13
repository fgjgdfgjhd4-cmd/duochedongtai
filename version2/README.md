# Version2: GPU-first Refactor

这个目录是独立重构版本，不修改上层已有代码。目标是把原项目中最耗时的候选解评估、碰撞检测、群智能优化迭代从 Python/NumPy 循环迁移到 GPU 上的 PyTorch 批量张量计算。

## 当前 CPU 瓶颈

- `map.py::objective()` 被 ABC、PSO、CPSO、ABCL 高频调用，每次只评估一个候选动作，并在 Python 层循环 agent、圆形障碍物、矩形障碍物。
- `Compare_Method_PSO/PSO_map.py`、`CPSO/CPSO_map.py`、`abc_for_map.py` 在粒子/食物源维度逐个调用 `env.objective()`，无法利用 GPU 并行。
- `map.py::step()` 也逐 agent 做运动学更新、终点判断、机器人碰撞、障碍物碰撞和奖励。
- `CNN_PPO.py` 的模型会使用 CUDA，但策略网络输出后的优化器和环境评估仍主要在 CPU。
- `render()` 依赖 pygame/cv2，适合保留在 CPU，只在需要生成图像状态或展示时调用。

## 重构原则

1. 保持原目录只读，不改任何旧文件。
2. 用 `torch.Tensor` 表示 agent 位置、目标、朝向、障碍物和候选动作。
3. 优化器使用 `evaluate_batch(actions_batch)`，一次评估形状为 `[batch, agent_num, 2]` 的候选动作。
4. 只有渲染、CSV、日志和少量控制逻辑留在 CPU。
5. 优先重构热点路径，不先动 pygame 渲染和旧实验脚本。

## 文件说明

- `gpu_config.py`: 统一设备、dtype、随机种子。
- `gpu_map.py`: GPU 版多机器人地图环境和批量 objective。
- `gpu_optimizers.py`: GPU 版 PSO/CPSO/ABC 风格优化器，核心评估全走 batch tensor。
- `run_compare_gpu.py`: 最小可运行入口，用 GPU 版优化器跑 episode 并输出统计。
- `MIGRATION_PLAN.md`: 分阶段迁移计划、验证指标和风险控制。

## 快速运行

```bash
python version2/run_compare_gpu.py --algorithm pso --agent-num 10 --steps 50 --population 256 --iterations 30
python version2/run_compare_gpu.py --algorithm cpso --agent-num 10 --steps 50 --population 256 --iterations 30
python version2/run_compare_gpu.py --algorithm abc --agent-num 10 --steps 50 --population 256 --iterations 30
```

如果机器没有 CUDA，会自动回退到 CPU，但这个版本的收益主要来自 CUDA。

## 跑完整重构版实验

原项目完整实验分两类：

- 算法对比：`ppo_abc`、`abc`、`pso`、`cpso`、`abcl`。
- 消融实验：`origin_abc`、`fixed_iabc`、`manual_iabc`、`ppo_iabc`。

`version2` 已经提供 GPU 版入口覆盖这些算法名：

- 对比实验：`ppo_abc`、`abc`、`pso`、`cpso`、`abcl`
- 消融实验：`origin_abc`、`fixed_iabc`、`manual_iabc`、`ppo_iabc`
- 去策略消融：`no_spiral_iabc`、`no_guided_iabc`、`no_diff_iabc`

完整跑重构版当前支持的实验用：

```bash
python version2/run_full_gpu_experiments.py \
  --suite all \
  --agent-nums 5 10 \
  --map-indices 0 1 2 \
  --episodes 10 \
  --max-ep-len 200 \
  --population 256 \
  --iterations 30 \
  --checkpoint PPO_preTrained/Map/10_robots/0511-15-16PPO_Map_6000.pth \
  --output-dir version2_results
```

结果会写到：

- `version2_results/<timestamp>/episodes.csv`
- `version2_results/<timestamp>/summary.csv`

断点续跑：

```bash
python version2/run_full_gpu_experiments.py \
  --resume-dir version2_results/<timestamp>
```

只跑去策略消融：

```bash
python version2/run_full_gpu_experiments.py \
  --suite ablation \
  --algorithms no_spiral_iabc no_guided_iabc no_diff_iabc \
  --agent-nums 5 10 \
  --map-indices 0 1 2 \
  --episodes 30 \
  --max-ep-len 200 \
  --population 256 \
  --iterations 30 \
  --output-dir version2_results
```

注意：`ppo_abc` 和 `ppo_iabc` 现在使用 GPU 生成的 4 通道状态图接入原 PPO checkpoint，输入形状与原 CNN 一致，但不是 pygame + cv2 HSV 渲染的逐像素复刻。因此它是可运行的 GPU 迁移版本；如果要做严格论文级复现实验，还需要额外做 PPO 状态图一致性校准。

状态一致性校准可以先用验证脚本量化同一批起终点下 legacy 渲染状态和 GPU raster 状态导致的 PPO 概率差异：

```bash
python version2/validate_ppo_state_fidelity.py \
  --agent-nums 5 10 \
  --map-indices 0 1 2 \
  --episodes 10 \
  --checkpoint PPO_preTrained/Map/10_robots/0511-15-16PPO_Map_6000.pth \
  --output-dir state_fidelity_results
```

脚本会输出 `state_fidelity.csv`，其中 `action_match`、`probability_l1`、`probability_kl_legacy_to_gpu` 用来判断 GPU 状态是否足够接近旧 PPO checkpoint 的输入分布。

## 迁移目标

第一阶段先把候选解评估和 PSO/CPSO/ABC 优化放到 GPU；第二阶段让 PPO 调度只传 tensor，不经过 NumPy；第三阶段再做多地图、多 episode 的批量并行。这样可以先拿到最大加速，同时避免一次性重写全部实验逻辑。

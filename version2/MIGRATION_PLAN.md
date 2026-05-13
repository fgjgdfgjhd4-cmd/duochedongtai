# GPU 重构计划

## 1. 热点重构范围

优先迁移下面三类计算：

- 候选动作 objective：目标距离、机器人间斥力、障碍物斥力、碰撞惩罚。
- 群智能优化器：PSO/CPSO 的粒子更新、ABC 的食物源生成和适应度选择。
- episode step：位置更新、终点/碰撞判断、奖励计算。

暂时不迁移：

- pygame/cv2 渲染。
- CSV 写入、日志、命令行解析。
- 已训练 PPO checkpoint 格式。

## 2. 新架构

```text
run_compare_gpu.py
    |
    +-- GpuMapEnv
    |       +-- reset()
    |       +-- evaluate_batch([B, N, 2]) -> [B]
    |       +-- step([N, 2])
    |
    +-- GpuPSO / GpuCPSO / GpuABC
            +-- optimize() -> [N, 2]
```

`B` 是候选解数量，`N` 是机器人数量。所有候选解共享当前环境状态，因此可以一次性评估。

## 3. 分阶段实施

### 阶段 A: 批量 objective

- 把 `agent_positions`、`destinations`、`orientation` 转为 `[N, 2]` / `[N]` tensor。
- 把候选动作 `[B, N, 2]` 转为候选下一位置 `[B, N, 2]`。
- 用 broadcasting 计算：
  - 目标距离改变量。
  - agent-agent 距离矩阵 `[B, N, N]`。
  - circle obstacle 距离 `[B, N, C]`。
  - rectangle obstacle AABB 距离 `[B, N, R]`。
- 输出 batch cost `[B]`，fitness 由优化器统一转换。

完成状态：已在 `gpu_map.py` 实现。

### 阶段 B: GPU 优化器

- PSO/CPSO 粒子位置和速度保持在 GPU。
- 每代只调用一次 `env.evaluate_batch(self.x)`。
- ABC 食物源保持在 GPU，以批量随机扰动生成新候选。
- 最终动作只在返回给 `step()` 时作为 tensor 使用。

完成状态：已在 `gpu_optimizers.py` 实现初版。

### 阶段 C: PPO 接入

- 保留 `CNN_PPO.py` 的网络结构，但新建 GPU 版入口，让 `select_strategy_profile()` 输入直接来自 tensor 状态。
- 对 PPO-ABC，仅让 PPO 输出策略概率；ABC 的候选生成和适应度评估继续走 `GpuABC`。
- 渲染状态只在需要图像策略时调用；否则优先使用低维 tensor 状态。

完成状态：已实现可运行版本。当前通过 GPU 生成 4 通道状态图喂给原 PPO checkpoint，不依赖 pygame/cv2；后续需要做状态图与原 HSV 渲染的一致性校准。

### 阶段 C.5: 原实验全量对齐

- `All_Compare_Test.py` 对齐项：
  - 已支持：`ppo_abc`、`abc`、`pso`、`cpso`、`abcl`。
- `ablation_experiment.py` 对齐项：
  - 已支持：`origin_abc`、`fixed_iabc`、`manual_iabc`、`ppo_iabc`。
- 当前可用入口：`run_full_gpu_experiments.py`，会跑完 version2 已支持的全量网格并输出 `episodes.csv` / `summary.csv`。

### 阶段 D: 多 episode 并行

- 把环境状态扩展为 `[E, N, ...]`，其中 `E` 是 episode/map batch。
- 优化器评估扩展为 `[E, B, N, 2]`。
- 适合一次跑多种 seed 或多张地图。

完成状态：计划中。

## 4. 正确性验证

建议按顺序验证：

1. 固定 seed，比较原 `Map.objective(single_solution)` 与 `GpuMapEnv.evaluate_batch(single_solution)` 的 cost 排名是否一致。
2. 对随机 1000 个候选解比较 top-k 候选是否高度重合。
3. 比较单 episode 的 success/collision/timeout 分布。
4. 比较 10/30/100 agent 下每步耗时和 GPU 利用率。

注意：新版 objective 是对原逻辑的 GPU 向量化近似，尤其矩形斥力和碰撞惩罚使用了更适合 batch 的距离/AABB 表达。目标是保持优化行为一致，而不是逐浮点完全一致。

## 5. 性能目标

- `population >= 128` 时，`evaluate_batch()` 应明显快于逐个 `env.objective()`。
- `agent_num >= 10`、圆形障碍物较多时收益最大。
- 小 batch 或无 CUDA 时，GPU 版可能不比原版快，这是正常情况。

## 6. 风险和处理

- GPU 显存占用：限制 `population` 和 episode batch，必要时分块评估。
- 数值差异：保留 CPU 对照脚本，先比较排序和成功率。
- 渲染瓶颈：训练/比较默认不渲染，只有可视化时调用 CPU 渲染。
- 数据来回拷贝：`step()`、`optimize()`、`evaluate_batch()` 全部接受 tensor，避免 NumPy 往返。

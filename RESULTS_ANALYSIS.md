# Experiment Results Analysis

Generated: 2026-05-13

## Result Files

| Run | Status | Scope | Files | Use |
| --- | --- | --- | --- | --- |
| `version2_results/20260513-101213` | complete | GPU refactor, 540 episodes | `episodes.csv`, `summary.csv` | Main result set |
| `compare_results/20260512-230632` | complete | legacy compare, 150 episodes, 10 robots only | `episodes.csv`, `summary.csv` | Runtime baseline and legacy behavior reference |

Temporary smoke runs, CPU-constrained reduced runs, and the stopped legacy ablation run were removed during cleanup after the final `version2` run completed.

Important caveat: `version2` is a GPU-first migration. It uses tensorized state and objective paths, and the README notes that PPO state generation is not a pixel-perfect pygame/cv2 reproduction. Therefore, legacy and version2 success/reward values should not be treated as strict apples-to-apples method quality comparisons. Runtime comparisons are still useful as migration evidence.

## Version2 Full Run

Full run path: `version2_results/20260513-101213`

Configuration:

| Parameter | Value |
| --- | --- |
| Device | `cuda:0` |
| Suites | compare + ablation |
| Agent counts | 5, 10 |
| Maps | 0, 1, 2 |
| Episodes | 10 per algorithm/map/agent-count |
| Max episode length | 200 |
| Population | 256 |
| Iterations | 30 |
| Total episodes | 540 |

Completion:

| File | Rows |
| --- | ---: |
| `episodes.csv` | 540 data rows |
| `summary.csv` | 54 data rows |

## Version2 Aggregate Results

### Compare Suite

| Algorithm | Episodes | Success | Collision | Timeout | Mean steps | Mean elapsed (s) | Mean step (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `abc` | 60 | 0.217 | 0.000 | 0.783 | 172.883 | 38.178 | 0.220 |
| `abcl` | 60 | 0.217 | 0.017 | 0.767 | 170.933 | 30.118 | 0.175 |
| `cpso` | 60 | 0.233 | 0.000 | 0.767 | 171.350 | 11.791 | 0.069 |
| `ppo_abc` | 60 | 0.200 | 0.017 | 0.783 | 172.850 | 39.965 | 0.230 |
| `pso` | 60 | 0.200 | 0.017 | 0.783 | 172.933 | 10.987 | 0.063 |

### Ablation Suite

| Algorithm | Episodes | Success | Collision | Timeout | Mean steps | Mean elapsed (s) | Mean step (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `origin_abc` | 60 | 0.200 | 0.017 | 0.783 | 173.367 | 26.666 | 0.153 |
| `fixed_iabc` | 60 | 0.217 | 0.017 | 0.767 | 171.750 | 37.860 | 0.220 |
| `manual_iabc` | 60 | 0.217 | 0.000 | 0.783 | 172.850 | 38.114 | 0.220 |
| `ppo_iabc` | 60 | 0.200 | 0.033 | 0.767 | 171.450 | 40.483 | 0.235 |

### By Agent Count

| Suite | Algorithm | Agents | Episodes | Success | Collision | Timeout | Mean elapsed (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| compare | `abc` | 5 | 30 | 0.333 | 0.000 | 0.667 | 35.088 |
| compare | `abc` | 10 | 30 | 0.100 | 0.000 | 0.900 | 41.269 |
| compare | `abcl` | 5 | 30 | 0.333 | 0.000 | 0.667 | 28.304 |
| compare | `abcl` | 10 | 30 | 0.100 | 0.033 | 0.867 | 31.932 |
| compare | `cpso` | 5 | 30 | 0.333 | 0.000 | 0.667 | 10.818 |
| compare | `cpso` | 10 | 30 | 0.133 | 0.000 | 0.867 | 12.764 |
| compare | `ppo_abc` | 5 | 30 | 0.333 | 0.000 | 0.667 | 36.819 |
| compare | `ppo_abc` | 10 | 30 | 0.067 | 0.033 | 0.900 | 43.110 |
| compare | `pso` | 5 | 30 | 0.333 | 0.000 | 0.667 | 10.099 |
| compare | `pso` | 10 | 30 | 0.067 | 0.033 | 0.900 | 11.874 |
| ablation | `origin_abc` | 5 | 30 | 0.333 | 0.000 | 0.667 | 24.520 |
| ablation | `origin_abc` | 10 | 30 | 0.067 | 0.033 | 0.900 | 28.812 |
| ablation | `fixed_iabc` | 5 | 30 | 0.333 | 0.000 | 0.667 | 35.026 |
| ablation | `fixed_iabc` | 10 | 30 | 0.100 | 0.033 | 0.867 | 40.693 |
| ablation | `manual_iabc` | 5 | 30 | 0.333 | 0.000 | 0.667 | 35.024 |
| ablation | `manual_iabc` | 10 | 30 | 0.100 | 0.000 | 0.900 | 41.204 |
| ablation | `ppo_iabc` | 5 | 30 | 0.333 | 0.000 | 0.667 | 36.325 |
| ablation | `ppo_iabc` | 10 | 30 | 0.067 | 0.067 | 0.867 | 44.642 |

## Legacy vs Version2 Runtime

Comparison restricted to 10-robot compare suite because legacy full compare only covers 10 robots.

| Algorithm | Legacy mean elapsed (s) | Version2 mean elapsed (s) | Speedup |
| --- | ---: | ---: | ---: |
| `abc` | 137.852 | 41.269 | 3.34x |
| `abcl` | 111.167 | 31.932 | 3.48x |
| `cpso` | 417.173 | 12.764 | 32.68x |
| `ppo_abc` | 111.851 | 43.110 | 2.60x |
| `pso` | 320.178 | 11.874 | 26.97x |

The largest gains are in PSO/CPSO, matching the expected benefit of batch GPU fitness evaluation. PPO-based methods improve less because each timestep still invokes PPO scheduling and the full environment/optimizer control path.

## Key Findings

1. **GPU refactor solved the runtime bottleneck.**
   Legacy PSO/CPSO took hundreds of seconds per 10-robot episode. Version2 reduces PSO to about 11.9 seconds and CPSO to about 12.8 seconds per episode, a 27-33x speedup.

2. **Algorithm quality is clustered in the version2 run.**
   In the compare suite, success rates are close: `cpso` is highest at 0.233, `abc`/`abcl` at 0.217, and `ppo_abc`/`pso` at 0.200. Differences are small at 60 episodes per method.

3. **5-robot cases are much easier than 10-robot cases.**
   Most methods achieve about 0.333 success on 5 robots but only 0.067-0.133 on 10 robots. The increase in agent count is the dominant difficulty driver.

4. **Map 2 is easiest for 5 robots; map 0 is hardest for 10 robots.**
   In the compare suite, 5-robot map 2 reaches 0.5 success across several algorithms. For 10 robots on map 0, all compare algorithms have 0.0 success in this run.

5. **Ablation variants do not show a clear success improvement over origin ABC.**
   `fixed_iabc` and `manual_iabc` reach 0.217 success versus `origin_abc` and `ppo_iabc` at 0.200. The margin is small and likely not enough to claim superiority without more seeds or calibrated PPO state fidelity.

6. **`origin_abc` is the fastest ablation method.**
   It averages 26.7 seconds per episode, compared with 37.9-40.5 seconds for the IABC variants. If success is statistically similar, origin ABC is a strong runtime baseline.

7. **`ppo_iabc` currently has the highest collision rate among ablations.**
   It has 0.033 collision rate overall, and 0.067 for 10 robots. This may indicate the PPO scheduler is not yet calibrated with the new GPU state representation.

8. **Legacy and version2 behavior differ enough that method-quality claims should use version2 internally, not across old/new.**
   Legacy compare had higher success for `ppo_abc` and `abcl` on 10 robots. Because version2 changes state/objective implementation details, that discrepancy should trigger validation before using version2 results as a direct replacement for legacy paper numbers.

## Suggested Next Experiments

1. **State-fidelity validation for PPO variants.**
   Run a small paired set where legacy and version2 use the same starts/targets and compare PPO strategy probabilities over the same states. This tests whether the GPU-generated state image is compatible with the old checkpoint.

2. **Repeat version2 full run with 3 seeds.**
   Current run has 10 episodes per map/agent/algorithm but one global seed schedule. Three independent seeds would make success-rate comparisons more defensible.

3. **Focused 10-robot benchmark.**
   Since 10 robots are the hard case, run only 10-robot map 0/1/2 with more episodes, e.g. 30 per algorithm, to reduce variance around the low success rates.

4. **Ablate population and iterations.**
   Version2 makes larger populations affordable. Sweep `(population, iterations)` such as `(128, 20)`, `(256, 30)`, `(512, 30)`, `(512, 50)` and report success/runtime Pareto curves.

5. **Checkpoint recalibration or retraining for version2 state tensors.**
   If PPO probability traces differ from legacy, either calibrate the GPU state construction to match the original renderer or retrain/fine-tune the PPO scheduler on version2 states.

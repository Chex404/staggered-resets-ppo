# Staggered vs Synchronous Environment Resets in Massively Parallel PPO

A lightweight, fully runnable reproduction of the core idea of
**"Staggered Environment Resets Improve Massively Parallel On-Policy Reinforcement Learning"**
(Bharthulwar, Tao & Su, NeurIPS 2025, [arXiv:2511.21011](https://arxiv.org/abs/2511.21011)).

The whole project is ~600 lines of PyTorch in 5 files. It runs on a single NVIDIA GPU or on a CPU.

---

## 1. The story (one slide)

**GPU acceleration → massive parallelism → a data-distribution problem → staggered resets → experiment.**

1. **Accelerated AI.** Writing the environment as batched tensor operations lets one `env.step()` advance
   thousands of environments at once on a GPU. PPO then collects `N × K` transitions per update
   (N parallel envs, K steps each).
2. **The catch.** To use that throughput, practitioners use *short* rollouts (K ≪ episode horizon H), so the policy
   updates often. But if all N envs are reset **synchronously**, they all sit at the *same* point in the episode.
   Every PPO batch therefore contains only a thin K-step slice of the task (all "stage 3", then all "stage 4", …),
   and every H/K updates everything jumps back to the start. The learner sees a **cyclically non-stationary** data
   stream → value-function spikes, policy oscillation, catastrophic forgetting.
3. **The fix.** **Staggered resets**: before training, advance env group *j* by *j·K* steps so the envs are spread
   uniformly over the horizon. Every batch now contains samples from the whole episode. Zero change to PPO,
   zero extra compute during training.
4. **The experiment.** Same PPO, same hyperparameters, same seeds: only the reset protocol differs.

---

## 2. How to run

```bash
pip install -r requirements.txt          # torch, numpy, matplotlib

python run_experiments.py --preset quick # ~10 min on 1 CPU core (estimate; faster on a GPU): lighter demo run
python run_experiments.py                # full default suite (~40 min on 1 CPU core: measured; much faster on GPU)
python make_plots.py                     # -> figures/*.png  and  results/summary.md
```

Single training runs (prints learning progress live, ~1 min on CPU):

```bash
python ppo.py --reset_mode sync
python ppo.py --reset_mode staggered
python benchmark.py                      # throughput benchmark only
```

The device is picked automatically (`cuda` if available, else `cpu`). You can run a subset of experiments with
`python run_experiments.py --only main,benchmark`.

**Suggested live demo (≈2 min):** run `python ppo.py --reset_mode sync` and `--reset_mode staggered` side-by-side in two
terminals and point at the `train_acc` column: synchronous jumps between ~0.05 and ~0.95 every few updates, staggered climbs
smoothly.

---

## 3. What is in the project

| File | What it does |
|---|---|
| `env.py` | Vectorized **skill-chain environment** (the paper's own toy benchmark), the `stagger()` function, and a naive one-Python-object-per-env version for the benchmark |
| `ppo.py` | Minimal PPO (clipped objective, GAE, value loss, entropy bonus), CleanRL-style. One flag: `reset_mode = sync / staggered` |
| `benchmark.py` | Throughput: Python loop vs vectorized CPU vs GPU; reset-load profile |
| `run_experiments.py` | Runs experiments E1–E4, saves raw logs to `results/*.json` |
| `make_plots.py` | Generates all figures + `results/summary.md` |

### The environment (paper Sec. 4.1 / App. B.2)
- An episode lasts **H** steps, split into **B = H/L** stages ("blocks") of L = 5 steps.
- Observation: the index of the current stage. Action: one of 20. Each stage has a secret correct action.
- Reward **+0.5** for the correct action, **−0.5** otherwise.
- **Skill gating:** after L steps the agent moves to the next stage if it got ≥ 3 correct (mastery), or with probability
  `p_prog = 0.5` anyway. Episodes always last exactly H steps → a perfectly synchronous reset pattern.
- It is a stand-in for long-horizon robot tasks (e.g. *reach → grasp → lift → place*): each stage needs a different skill.

Why a toy task instead of ManiSkill3 robots? The paper itself uses this environment to isolate the effect, it needs no
physics engine, and a full run takes ~1 minute, so the experiment can be repeated over seeds and settings in a single session.

### The staggering (paper Sec. 3.3 / App. B.1)
```python
n_groups = ceil(H / K)                      # e.g. 200/5 = 40 groups
offset_i = (i % n_groups) * K               # group j starts j*K steps into the episode
advance env i by offset_i steps with random actions (masked, batched step)
```
After this warm-up (not counted as training data), everything is standard synchronous PPO.

### Hyperparameters (paper Table 2)
N = 512 envs, K = 5, H = 200, 150 PPO updates, 4 epochs × 4 minibatches, lr 3e-4, γ 0.99, λ 0.95, clip 0.2,
entropy 0.01, 64-d embedding + 4×256 ReLU MLP for actor and critic.

---

## 4. Experiments & figures

| Figure | Question | Subject |
|---|---|---|
| `fig1_reset_schedule.png` | What do the two strategies look like? (episode timestep per env over time; resets per step) | both |
| `fig2_batch_composition.png` | Which part of the episode does each PPO batch contain? (paper Fig. 6c/d) | RL |
| `fig3_learning_curves.png` | Return, per-stage accuracy, success rate vs environment steps (mean ± std, 3 seeds) | RL |
| `fig4_stability.png` | Critic error and policy KL per update (paper App. E) | RL |
| `fig5_forgetting.png` | Forgetting matrix: best-so-far − current accuracy per stage (paper App. E) | RL |
| `fig6_walltime_throughput.png` | Return vs wall-clock; training throughput sync vs staggered | Accel. AI |
| `fig7_horizon_sweep.png` | How does the gap depend on K/H? (paper Fig. 2a) | RL |
| `fig8_scaling.png` | Does adding more parallel envs N help? (paper Fig. 5) | Accel. AI |
| `fig9_throughput.png` | Python loop vs vectorized CPU vs GPU, env and env+policy steps/s vs N | Accel. AI |

All numbers are written to `results/summary.md`.

---

## 5. Results (from the included run: 1 CPU core, default preset)

All numbers below are in `results/summary.md`; figures are in `figures/`. Hardware: 1 CPU core (no GPU in the sandbox).

**E1 – Main comparison** (N=512, K=5, H=200, 3 seeds, 150 PPO updates, mean ± std)

| Metric | Synchronous | Staggered |
|---|---|---|
| Final eval return (max 100) | 45.4 ± 8.7 | **83.0 ± 12.7** |
| Accuracy over all task stages | 0.739 ± 0.051 | **0.956 ± 0.032** |
| Success rate (reach final stage) | 0.012 ± 0.007 | **0.564 ± 0.311** |
| Std of batch accuracy (oscillation) | 0.286 | **0.028** |
| Max critic error (value MSE) | 87.4 ± 9.8 | **4.7 ± 0.9** |
| Mean policy KL per update | 0.085 | **0.024** |
| Mean forgetting | 0.127 | **0.027** |
| Training throughput (env steps/s) | 6,689 ± 209 | 6,840 ± 45 |

Staggered resets give much better learning at identical compute cost. The critic-error spikes in the synchronous runs fall exactly
on updates 40, 80, 120 (= H/K), the mass-reset moments (fig 4). The success-rate std for staggered is large (one seed reaches
the final stage late), so quote success rate with that caveat.

**E2 – Horizon sweep** (K=5 fixed, 2 seeds): the gap grows as H/K grows.

| H (H/K) | Stage acc. sync | Stage acc. staggered |
|---|---|---|
| 50 (10) | 1.000 | 1.000 |
| 100 (20) | 0.925 ± 0.008 | 1.000 |
| 200 (40) | 0.721 ± 0.054 | 0.933 ± 0.008 |
| 400 (80) | 0.410 ± 0.015 | 0.783 ± 0.004 |

**E3 – Scaling with N** (**1 seed only**, fixed 150 updates, so larger N also means more total env steps)

| N | Final stage acc. sync | Final stage acc. staggered |
|---|---|---|
| 128 | 0.642 | 0.842 |
| 512 | 0.783 | 0.875 |
| 2048 | 0.725 | **1.000** |

Staggered improves with N; synchronous plateaus and slightly drops at N=2048, consistent with the paper's "saturation" claim.
**This is a single seed, so treat it as suggestive, not conclusive.** The threshold-based "steps/time to 80%" panels in fig 8 are
coarse (evaluated every 2 updates, wall time includes evaluation) and sync often never reaches the threshold; use them qualitatively.

**E4 – Throughput (CPU)**: the vectorized torch env overtakes a one-object-per-env Python loop (~1.9 M steps/s, flat) once N ≳ 1000
and reaches ~12 M steps/s at N=4096 and ~20–22 M steps/s at N ≥ 16k (≈ 6–12× faster). Once the policy network is included, rollout throughput is
~60–70 k steps/s on this CPU: the network forward pass, not the environment, is the bottleneck (this is exactly the workload a GPU accelerates).
Reset load: sync = 512 resets in a single step, staggered = at most 13 per step (same total, 1536 each).

---

## 6. Talking points

**Reinforcement Learning**
- PPO is on-policy: it can only learn from the batch just collected. So whatever distribution that batch has is
  *all the learner sees* for that update. Batch composition matters a lot.
- Synchronous + short rollouts ⇒ each batch is one narrow stage of the task ⇒ the network is fine-tuned on stage *k*,
  then stage *k+1*, ... ⇒ it forgets early stages (fig 5: stage 0 is repeatedly forgotten), the critic is badly wrong
  every time the batch jumps back to the start (fig 4: spikes exactly at updates 40, 80, 120 = H/K), and the policy
  oscillates (larger KL per update).
- Staggered ⇒ each batch approximates the true on-policy state distribution over the full episode ⇒ stable, i.i.d.-like
  learning signal, like a long rollout but keeping the high update frequency.
- The effect grows with H/K (fig 7): at H/K = 10 both methods solve the task; at long horizons synchronous collapses.

**Accelerated AI**
- Vectorization: one tensor op steps all envs; no Python loop, no per-env objects, data stays on the device together with
  the policy (no CPU↔GPU copies). Fig 9 shows the throughput gap vs a Python loop.
- Short rollouts are *why* this problem exists: they are the standard way to turn simulator throughput into
  frequent updates (the paper's Table 1 shows K = 8–16 is the wall-clock optimum on a real task).
- Staggering is **free**: the same number of env steps and network passes per update, so throughput is identical
  (fig 6). The one-off warm-up costs < H steps.
- Reset load: synchronous resets hit all N envs in the same step (a burst of N resets every H steps); staggered spreads
  them to ≈ N·K/H per rollout (fig 1, bottom). In real GPU simulators resets are expensive, so smoothing them helps.
- Scalability (fig 8): with synchronous resets, extra environments add *more copies of the same slice* of the task,
  so more parallelism buys less. Staggered makes the extra envs cover different parts of the task.

---

## 7. Honest limitations / what differs from the paper

- **Environment:** the paper's own toy skill-chain task, not the ManiSkill3 / IsaacGym robot tasks (those need a heavy GPU
  physics stack and hours of training). The toy task is where the paper explains the mechanism; the robot results are
  its generalisation.
- **Scale:** 512 envs and ≤ 0.4 M env steps per run instead of 4096 envs and 200 M steps; 3 seeds instead of 10.
- **Compute numbers are hardware-dependent.** The included numbers are from a 1-core CPU sandbox, where time scales
  roughly linearly with N. On a GPU, larger N is almost free up to saturation; re-run `python run_experiments.py`
  on your GPU to get GPU numbers (fig 9 then also includes a GPU curve).
- The toy environment step is very cheap, so the Python-loop baseline is not dramatically slow here; with a real physics
  step the gap between per-env Python simulation and batched GPU simulation is orders of magnitude larger.
- Warm-up for staggering uses random actions (as in the paper). Partial-reset "gates" from the paper are not needed
  because this task has no early termination.

---

## 8. Possible viva questions

- *Why not just use long rollouts (K = H)?* That fixes the distribution but gives far fewer policy updates per env step
  and more memory per update; the paper shows short rollouts are faster in wall-clock time.
- *Does random early termination fix it naturally?* Partly: the paper shows locomotion tasks with frequent random falls
  benefit less. Deterministic, fixed-length tasks (like this one, and many manipulation tasks) suffer the most.
- *Why does the critic error spike at updates 40/80/120?* H/K = 200/5 = 40. After 40 updates all envs reset together,
  the batch jumps from the last stage back to stage 0, which the critic has not seen for 40 updates.
- *Why group envs into ⌈H/K⌉ groups instead of random offsets?* Offsets that are multiples of K give each batch exactly one
  group per K-step window: an even coverage of the horizon, and resets happen in batched groups (cheap on GPU).
- *Is this PPO-specific?* No: it is a change to data collection, so it applies to any on-policy method (the paper also
  shows it with SAPG).

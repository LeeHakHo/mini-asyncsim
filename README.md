# mini-asyncsim

**What happens to a robot policy when the world doesn't wait for it to think?**

CSCI596 (Scientific Computing & Visualization) final project, Fall 2026.
A minimal re-implementation of the asynchronous-simulation idea from
[Robomimic_Async](https://github.com/LeeHakHo/Robomimic_Async), built on top of
stock [robomimic](https://github.com/ARISE-Initiative/robomimic) v0.5 and
[robosuite](https://github.com/ARISE-Initiative/robosuite) v1.5, and run on USC CARC.

![Lift vs Square](docs/compare.png)

## Problem

Robot-learning benchmarks evaluate policies **synchronously**: the simulator freezes
while the policy computes its next action. Real robots don't freeze. A large model
(e.g. Diffusion Policy, which runs many denoising steps per action) may take tens to
hundreds of milliseconds, and during that time the robot keeps moving (or stands
still). Synchronous benchmarks therefore hide the cost of inference latency.

## Method

```
            sync                                     async (this project)
 obs ─► policy (world paused) ─► step        physics process: step ─ step ─ step ─ step   (fixed wall-clock rate)
 obs ─► policy (world paused) ─► step                           ▲ latest action    │ latest obs  (shared memory)
                                              policy process: ──┴── infer ─────────┴── infer ──
```

* `asyncsim/async_env.py`: the physics runs in **its own process** and steps the
  robosuite env at `control_freq × real_time_rate` (20 Hz) in wall-clock time. The
  latest observation and action are exchanged through shared memory; control
  messages (reset / run / result) go over a pipe. No GIL contention between physics
  and inference, like a real robot controller and a separate policy server.
* What the robot does on a tick with no new action (`--hold`):
  * `repeat`: re-apply the last action (zero-order hold). Both tasks use delta-pose
    (OSC_POSE) actions, so this keeps moving the arm and overshoots.
  * `stop`: apply each action once, then hold the arm still (keep the gripper
    command) until the next action arrives.
* `asyncsim/rollout.py`: `sync_rollout` and `async_rollout` use the same policy and
  the same injected latency, so the only difference is whether the world waits.
* Policies: BC (≈1 ms inference) and Diffusion Policy (≈135 ms inference on a CARC
  GPU, spikes up to ~1 s), trained with robomimic (low-dim observations,
  proficient-human demos) on CARC GPUs, on two tasks:
  * **Lift**: pick up a cube. Demos average 48 steps (2.4 s).
  * **Square**: fit a square nut onto a peg, a peg-insertion task with tight
    tolerance. Demos average 151 steps (7.5 s), so episodes get 800 ticks (40 s)
    instead of Lift's 400 (20 s).
* Sweep: {BC, DP} × {sync, async-repeat, async-stop} × extra latency
  {0, 100, 200, 400, 800 ms}, 50 rollouts per cell = 30 cells, each an independent
  CARC Slurm array task (`carc/sweep.slurm`), per task.

## Results

### Lift

Success rate (%) over 50 rollouts. Episodes are limited to 400 ticks (20 s of
sim time). Full numbers with 95% Wilson intervals: [`docs/summary.csv`](docs/summary.csv),
figure with time to success: [`docs/sweep.png`](docs/sweep.png).

| extra latency      | 0 ms | 100 | 200 | 400 | 800 |
|--------------------|-----:|----:|----:|----:|----:|
| BC sync            |  86  |  86 |  86 |  86 |  86 |
| BC async-repeat    |  94  |  94 |  66 |  12 |   2 |
| BC async-stop      |  98  |  98 |  92 |  84 |   0 |
| DP sync            | 100  | 100 | 100 | 100 | 100 |
| DP async-repeat    |   2  |   0 |   2 |   0 |   0 |
| DP async-stop      |  98  |  98 |  98 |  28 |   0 |

1. **Sync hides latency completely.** Success is flat because the world waits. With a
   fixed seed and deterministic policies every sync cell is identical.
2. **Async with zero-order hold collapses.** Repeating a delta-pose action while the
   policy thinks overshoots the target. BC breaks down from 200 ms on. Diffusion
   Policy fails at **0 ms extra latency**: its own ~135 ms inference (~3 control ticks)
   is already enough. Synchronous evaluation ranks DP above BC (100% vs 86%); under
   real-time execution with zero-order hold the ranking flips.
3. **Stopping until the next action keeps success but costs time.** Median time to
   success grows from 2.4 s to 16 s (BC, +400 ms) and from 8 s to 17.6 s (DP).
   The 0% at +800 ms is mostly timeouts: the task is too slow to finish in 20 s, not
   a wrong motion.

Caveat: sync has one outcome per initial state, so the small differences between sync
and async at 0 ms (e.g. BC 86% vs 94–98%) are not a meaningful effect.

### Square (peg insertion)

Same sweep, episodes limited to 800 ticks (40 s). Full numbers:
[`docs/square/summary.csv`](docs/square/summary.csv), figure with time to success:
[`docs/square/sweep.png`](docs/square/sweep.png).

| extra latency      | 0 ms | 100 | 200 | 400 | 800 |
|--------------------|-----:|----:|----:|----:|----:|
| BC sync            |  32  |  30 |  32 |  32 |  32 |
| BC async-repeat    |  26  |  16 |   4 |   0 |   0 |
| BC async-stop      |  26  |  12 |   2 |   0 |   0 |
| DP sync            |  46  |  46 |  46 |  46 |  46 |
| DP async-repeat    |   2  |   0 |   0 |   0 |   0 |
| DP async-stop      |  82  |  82 |  28 |   0 |   0 |

### Lift vs Square

1. **Insertion has a much smaller latency budget.** With the stop hold, BC keeps 84%
   at +400 ms on Lift but is down to 2% at +200 ms on Square; DP-stop keeps 98% at
   +200 ms on Lift but drops to 28% on Square. A stale observation
   matters little when grasping a cube and a lot when aligning a nut with a peg.
2. **Zero-order hold fails Diffusion Policy on both tasks**, already at 0 ms extra
   latency (2% on both): its own inference time is enough.
3. **Stopping is slow, and on Square the clock runs out.** DP-stop's median time to
   success grows from 22 s (+0 ms) to 37 s (+200 ms) against a 40 s limit, so the
   0% at +400 ms is likely mostly timeouts, as for Lift at +800 ms.
4. **Unexpected: on Square, DP-stop beats DP-sync (82% vs 46%, non-overlapping 95%
   intervals).** A likely explanation, not tested here: while the arm holds still
   the OSC controller settles onto each target pose, which helps a precise insertion.

Caveats for Square: the sync baselines are low (BC 32%, DP 46%). Training DP for
3000 instead of 1000 epochs did not help (25–30% sync success on a 20-rollout check vs 40% for epoch 1000), so the sweep
uses epoch 1000. The BC checkpoint (epoch 700) is the best of five on a separate
20-rollout sync check (seed 1; the sweep uses seed 0).

### Videos (Lift)

Same initial state, three modes side by side (sync | async-repeat | async-stop):

* [`docs/videos/bc_lat200_side_by_side.mp4`](docs/videos/bc_lat200_side_by_side.mp4): BC, +200 ms
* [`docs/videos/dp_lat0_side_by_side.mp4`](docs/videos/dp_lat0_side_by_side.mp4): Diffusion Policy, no extra latency
* [`docs/videos/dp_lat200_side_by_side.mp4`](docs/videos/dp_lat200_side_by_side.mp4): Diffusion Policy, +200 ms

Single episodes are illustrations, not statistics; see the table for rates.

## Running on CARC

```bash
# 1. environment (once; run as a job, the login node limits processes)
bash carc/setup_env.sh
# 2. data + training (GPU jobs); second argument = task (lift | square)
sbatch -A <account> carc/train.slurm bc
sbatch -A <account> carc/train.slurm diffusion_policy
sbatch -A <account> carc/train.slurm bc square
# 3. smoke test without a trained policy
python scripts/eval.py --dummy --dataset datasets/lift/ph/low_dim_v15.hdf5 --mode async -n 2
# 4. evaluation sweep (job array, 30 cells)
BC_CKPT=... DP_CKPT=... sbatch -A <account> --array=0-29%10 carc/sweep.slurm
BC_CKPT=... DP_CKPT=... HORIZON=800 sbatch -A <account> --array=0-29%10 --time=12:00:00 carc/sweep.slurm
# 5. plots -> results/sweep_<jobid>/sweep.png + summary.csv, then the task comparison
python scripts/plot.py results/sweep_<jobid>
python scripts/plot.py results/sweep_<jobid> --task Square --horizon 800
python scripts/compare.py docs/compare.png Lift=docs/summary.csv Square=docs/square/summary.csv
# 6. videos
BC_CKPT=... DP_CKPT=... sbatch -A <account> carc/videos.slurm
```

## Status

- [x] Async env (two processes, shared memory) + sync/async rollouts
- [x] Train BC / Diffusion Policy on Lift and Square
- [x] Latency sweeps on CARC (30-task job array per task)
- [x] Plots + analysis

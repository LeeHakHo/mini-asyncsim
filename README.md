# mini-asyncsim

CSCI596 final project, Fall 2026. Hyeonho Oh, JooSeok Park.

A minimal re-implementation of asynchronous simulation from
[Robomimic_Async](https://github.com/LeeHakHo/Robomimic_Async), built on
[robomimic](https://github.com/ARISE-Initiative/robomimic) v0.5 and
[robosuite](https://github.com/ARISE-Initiative/robosuite) v1.5, and run on USC CARC.

![Lift vs Square](docs/compare.png)

## Problem

Standard robot-learning benchmarks evaluate policies synchronously: the simulator
pauses while the policy computes its next action. On a real robot the world keeps
moving during inference, so synchronous evaluation does not measure the effect of
inference latency.

## Method

```
            sync                                     async
 obs ─► policy (world paused) ─► step        physics process: step ─ step ─ step ─ step   (fixed wall-clock rate)
 obs ─► policy (world paused) ─► step                           ▲ latest action    │ latest obs  (shared memory)
                                              policy process: ──┴── infer ─────────┴── infer ──
```

* `asyncsim/async_env.py`: the physics runs in a separate process and steps the
  robosuite env at a fixed wall-clock rate (20 Hz). The latest observation and
  action are exchanged through shared memory. Control messages go over a pipe.
* Behavior on a tick with no new action (`--hold`):
  * `repeat`: re-apply the last action.
  * `stop`: zero the arm command and keep the gripper command.
* `asyncsim/rollout.py`: `sync_rollout` and `async_rollout` use the same policy and
  the same added latency.
* Policies: BC (about 1 ms inference) and Diffusion Policy (about 135 ms inference on
  a CARC GPU), trained with robomimic on low-dim observations and proficient-human
  demos.
* Tasks:
  * Lift: pick up a cube. 400-tick episodes (20 s).
  * Square: fit a square nut onto a peg. 800-tick episodes (40 s).
* Sweep: {BC, DP} × {sync, async-repeat, async-stop} × extra latency
  {0, 100, 200, 400, 800 ms}, 50 rollouts per cell. Each of the 30 cells is one
  Slurm array task (`carc/sweep.slurm`).

## Results

Success rate (%) over 50 rollouts. Full numbers with 95% Wilson intervals and time
to success are in the summary files and figures.

### Lift

[`docs/summary.csv`](docs/summary.csv), [`docs/sweep.png`](docs/sweep.png)

| extra latency      | 0 ms | 100 | 200 | 400 | 800 |
|--------------------|-----:|----:|----:|----:|----:|
| BC sync            |  86  |  86 |  86 |  86 |  86 |
| BC async-repeat    |  94  |  94 |  66 |  12 |   2 |
| BC async-stop      |  98  |  98 |  92 |  84 |   0 |
| DP sync            | 100  | 100 | 100 | 100 | 100 |
| DP async-repeat    |   2  |   0 |   2 |   0 |   0 |
| DP async-stop      |  98  |  98 |  98 |  28 |   0 |

* Sync success does not depend on latency.
* With repeat, DP drops to 2% with no extra latency. BC stays at 94% up to +100 ms.
* Sync ranks DP above BC (100% vs 86%). With repeat, BC is higher.
* With stop, median time to success grows from 2.4 s to 16.2 s (BC, +400 ms) and
  from 8.1 s to 17.6 s (DP, +400 ms).

### Square

[`docs/square/summary.csv`](docs/square/summary.csv), [`docs/square/sweep.png`](docs/square/sweep.png)

| extra latency      | 0 ms | 100 | 200 | 400 | 800 |
|--------------------|-----:|----:|----:|----:|----:|
| BC sync            |  32  |  30 |  32 |  32 |  32 |
| BC async-repeat    |  26  |  16 |   4 |   0 |   0 |
| BC async-stop      |  26  |  12 |   2 |   0 |   0 |
| DP sync            |  46  |  46 |  46 |  46 |  46 |
| DP async-repeat    |   2  |   0 |   0 |   0 |   0 |
| DP async-stop      |  82  |  82 |  28 |   0 |   0 |

### Lift vs Square

* With stop at +200 ms, DP succeeds 98% on Lift and 28% on Square.
* With repeat and no extra latency, DP succeeds 2% on both tasks.
* With stop on Square, DP median time to success is 22.3 s at 0 ms and 36.6 s at
  +200 ms, against a 40 s limit.
* On Square, DP-stop (82%) is higher than DP-sync (46%).

Square checkpoints: BC epoch 700, DP epoch 1000.

### Videos (Lift)

Same initial state, sync | async-repeat | async-stop:

* [`docs/videos/bc_lat200_side_by_side.mp4`](docs/videos/bc_lat200_side_by_side.mp4): BC, +200 ms
* [`docs/videos/dp_lat0_side_by_side.mp4`](docs/videos/dp_lat0_side_by_side.mp4): Diffusion Policy, no extra latency
* [`docs/videos/dp_lat200_side_by_side.mp4`](docs/videos/dp_lat200_side_by_side.mp4): Diffusion Policy, +200 ms

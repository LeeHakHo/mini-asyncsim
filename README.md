# mini-asyncsim

**What happens to a robot policy when the world doesn't wait for it to think?**

CSCI596 (Scientific Computing & Visualization) final project, Fall 2026.
A minimal re-implementation of the asynchronous-simulation idea from
[Robomimic_Async](https://github.com/LeeHakHo/Robomimic_Async), built on top of
stock [robomimic](https://github.com/ARISE-Initiative/robomimic) v0.5 and
[robosuite](https://github.com/ARISE-Initiative/robosuite) v1.5, and run on USC CARC.

## Problem

Robot-learning benchmarks evaluate policies **synchronously**: the simulator freezes
while the policy computes its next action. Real robots don't freeze. A large model
(e.g. Diffusion Policy, which runs many denoising steps per action) may take tens to
hundreds of milliseconds, and during that time the robot keeps executing its previous
command. Synchronous benchmarks therefore hide the cost of inference latency.

## Method

```
            sync                                     async (this project)
 obs ─► policy (world paused) ─► step        physics thread:  step ─ step ─ step ─ step   (fixed wall-clock rate)
 obs ─► policy (world paused) ─► step                            ▲ latest action     │ latest obs
                                              policy thread:  ───┴── infer ──────────┴── infer ──
```

* `asyncsim/async_env.py`: a ~100-line wrapper. A background thread steps the
  robosuite env at `control_freq × real_time_rate` in wall-clock time and applies the
  most recent action (zero-order hold). The policy reads the latest observation.
* `asyncsim/rollout.py`: `sync_rollout` and `async_rollout` share the same policy and
  the same injected latency, so the only difference is whether the world waits.
* Policies: BC (fast) and Diffusion Policy (slow), trained with robomimic on the
  **Lift** task (low-dim observations) on CARC GPUs.
* Experiments: success rate as a function of {sync, async} × {extra latency 0–400 ms}
  × {BC, Diffusion Policy}. Each grid cell is an independent CARC Slurm array task.

## Expected results

* Sync: success rate is flat in latency (latency only costs wall-clock time).
* Async: success rate degrades as latency grows past the control period (50 ms at 20 Hz).
* The slower but stronger Diffusion Policy may lose its advantage over BC once
  latency counts, so model choice depends on the evaluation protocol.

## Running on CARC

```bash
# 1. environment (once)
bash carc/setup_env.sh
# 2. data + training (GPU jobs)
sbatch -A <account> carc/train.slurm bc
sbatch -A <account> carc/train.slurm diffusion_policy
# 3. smoke test without a trained policy
python scripts/eval.py --dummy --dataset datasets/lift/ph/low_dim_v15.hdf5 --mode async -n 2
# 4. evaluation sweep (job array)
BC_CKPT=... DP_CKPT=... sbatch -A <account> --array=0-47 carc/sweep.slurm
# 5. plots
python scripts/plot.py results/sweep_<jobid>
```

## Status

- [x] Async env wrapper + sync/async rollouts
- [ ] Train BC / Diffusion Policy on Lift
- [ ] Latency sweep on CARC
- [ ] Plots + analysis

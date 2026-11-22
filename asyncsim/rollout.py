"""
Synchronous vs asynchronous rollouts for a robomimic RolloutPolicy.

Both modes use the same policy and the same artificial extra latency
(`extra_latency`, seconds added to every policy call). In sync mode latency
only costs wall-clock time; in async mode the world keeps moving meanwhile.
"""
import time
from collections import deque

import numpy as np

from asyncsim.async_env import AsyncEnv


class FrameStacker:
    """Keeps the last `n` observations and stacks them along a new axis 0
    (robomimic policies trained with frame_stack > 1 expect [n, ...])."""

    def __init__(self, n):
        self.n = n
        self.hist = None

    def __call__(self, obs):
        if self.n <= 1:
            return obs
        if self.hist is None:
            self.hist = {k: deque([v] * self.n, maxlen=self.n) for k, v in obs.items()}
        else:
            for k, v in obs.items():
                self.hist[k].append(v)
        return {k: np.stack(list(d), axis=0) for k, d in self.hist.items()}


def _act(policy, obs, extra_latency):
    t = time.perf_counter()
    action = policy(ob=obs)
    if extra_latency > 0:
        remaining = extra_latency - (time.perf_counter() - t)
        if remaining > 0:
            time.sleep(remaining)
    return action, time.perf_counter() - t


def sync_rollout(policy, env, horizon, frame_stack=1, extra_latency=0.0,
                 record=False, init_state=None):
    policy.start_episode()
    stack = FrameStacker(frame_stack)
    obs = env.reset()
    if init_state is not None:
        obs = env.reset_to({"states": init_state})
    states = [env.get_state()["states"]] if record else None
    latencies = []
    success = False
    t0 = time.perf_counter()
    for step in range(horizon):
        action, lat = _act(policy, stack(obs), extra_latency)
        latencies.append(lat)
        obs, _, _, _ = env.step(action)
        if record:
            states.append(env.get_state()["states"])
        if env.is_success()["task"]:
            success = True
            break
    return dict(
        states=states,
        mode="sync",
        success=float(success),
        steps=step + 1,
        wall_time=time.perf_counter() - t0,
        mean_latency=float(np.mean(latencies)),
        max_latency=float(np.max(latencies)),
        policy_calls=len(latencies),
    )


def async_rollout(policy, env, horizon, control_freq=20.0, real_time_rate=1.0,
                  frame_stack=1, extra_latency=0.0, record=False, init_state=None,
                  hold="repeat"):
    """Policy runs in this (main) thread as fast as it can; physics runs in a
    background thread at a fixed wall-clock rate. `hold` = what the robot does
    while waiting for the next action ("repeat" last action or "stop")."""
    policy.start_episode()
    stack = FrameStacker(frame_stack)
    aenv = AsyncEnv(env, control_freq=control_freq, real_time_rate=real_time_rate,
                    horizon=horizon, record=record, hold=hold)
    aenv.reset(init_state)

    latencies, staleness = [], []
    last_obs_step = -1
    t0 = time.perf_counter()
    aenv.start()
    try:
        while not aenv.done():
            obs, obs_step = aenv.latest_obs()
            if obs_step == last_obs_step:
                # no new observation yet -> don't recompute on the same frame
                time.sleep(aenv.wall_dt / 10)
                continue
            last_obs_step = obs_step
            action, lat = _act(policy, stack(obs), extra_latency)
            latencies.append(lat)
            # how many physics ticks passed between observing and acting
            staleness.append(aenv.step_count - obs_step)
            aenv.push_action(action)
    finally:
        aenv.stop()

    return dict(
        states=aenv.states if record else None,
        mode=f"async_{hold}",
        idle_ticks=aenv.idle_ticks,
        success=float(aenv.success),
        steps=aenv.step_count,
        wall_time=time.perf_counter() - t0,
        mean_latency=float(np.mean(latencies)) if latencies else 0.0,
        max_latency=float(np.max(latencies)) if latencies else 0.0,
        policy_calls=len(latencies),
        mean_staleness_ticks=float(np.mean(staleness)) if staleness else 0.0,
        physics_overruns=aenv.overruns,
    )

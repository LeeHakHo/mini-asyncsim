"""
Sync and async rollouts for a robomimic RolloutPolicy.

extra_latency adds a fixed delay (seconds) to every policy call.
"""
import time
from collections import deque

import numpy as np


class FrameStacker:
    """Stack the last n observations along a new first axis."""

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


def async_rollout(policy, aenv, env, frame_stack=1, extra_latency=0.0, hold="repeat",
                  record=False, init_state=None):
    """Run the policy in this process while physics runs in the AsyncEnv process.

    Args:
        aenv: AsyncEnv
        env: local env, used to sample the initial state
        hold: "repeat" or "stop" (see AsyncEnv.start)
    """
    policy.start_episode()
    stack = FrameStacker(frame_stack)
    if init_state is None:
        env.reset()
        init_state = env.get_state()["states"]
    aenv.reset(init_state)

    latencies, staleness = [], []
    last_obs_step = -1
    t0 = time.perf_counter()
    aenv.start(hold=hold, record=record)
    try:
        while not aenv.done():
            obs, obs_step = aenv.latest_obs()
            if obs_step == last_obs_step:
                # Wait for a new observation.
                time.sleep(aenv.wall_dt / 10)
                continue
            last_obs_step = obs_step
            action, lat = _act(policy, stack(obs), extra_latency)
            latencies.append(lat)
            # Physics ticks between observation and action.
            staleness.append(aenv.step_count - obs_step)
            aenv.push_action(action)
    finally:
        res = aenv.finish()

    return dict(
        states=res["states"],
        mode=f"async_{hold}",
        idle_ticks=res["idle_ticks"],
        success=float(res["success"]),
        steps=res["steps"],
        wall_time=time.perf_counter() - t0,
        mean_latency=float(np.mean(latencies)) if latencies else 0.0,
        max_latency=float(np.max(latencies)) if latencies else 0.0,
        policy_calls=len(latencies),
        mean_staleness_ticks=float(np.mean(staleness)) if staleness else 0.0,
        physics_overruns=res["overruns"],
    )

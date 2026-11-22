"""
Minimal asynchronous (real-time) wrapper around a robomimic/robosuite env.

In a normal (synchronous) rollout the world freezes while the policy thinks:
    obs -> policy(obs) -> env.step(action) -> obs -> ...

Here a background *physics thread* steps the env at a fixed wall-clock rate
(control_freq * real_time_rate) no matter what the policy is doing. The policy
reads the most recent observation and writes the most recent action. If the
policy is slow, the physics thread keeps applying the last action it received
(zero-order hold), so inference latency turns directly into stale control.
"""
import copy
import threading
import time

import numpy as np


class AsyncEnv:
    def __init__(self, env, control_freq=20.0, real_time_rate=1.0, horizon=400, record=False,
                 hold="repeat"):
        """
        Args:
            env: robomimic EnvBase (e.g. EnvRobosuite) - must have reset/step/is_success
            control_freq: physics/control ticks per simulated second
            real_time_rate: 1.0 = real time, 2.0 = world runs 2x faster than wall clock
            horizon: max number of physics ticks per episode
            hold: what the robot does on ticks with no new action
                "repeat": re-apply the last action (zero-order hold). With delta-pose
                          control this keeps moving and overshoots.
                "stop":   apply each action once, then zero the arm command (keep the
                          gripper command) until the next action arrives.
        """
        assert hold in ("repeat", "stop")
        self.hold = hold
        self.env = env
        self.dt = 1.0 / control_freq
        self.wall_dt = self.dt / real_time_rate
        self.horizon = horizon
        self.record = record
        self.states = []  # flattened sim states per tick (for video replay)

        self._lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        self._done = threading.Event()

        self._obs = None
        self._obs_step = 0
        self._action = None
        self._action_updates = 0

        self.step_count = 0
        self.success = False
        self.overruns = 0  # ticks where physics itself fell behind wall clock
        self.idle_ticks = 0  # ticks with no new action (robot held/stopped)
        self.actions_applied = []  # (tick, action_update_id) for analysis

    # ---------------- policy side API ----------------
    def reset(self, init_state=None):
        obs = self.env.reset()
        if init_state is not None:
            obs = self.env.reset_to({"states": init_state})
        self.states = [self.env.get_state()["states"]] if self.record else []
        with self._lock:
            self._obs = copy.deepcopy(obs)
            self._obs_step = 0
            self._action = np.zeros(self.env.action_dimension)
            self._action_updates = 0
        self.step_count = 0
        self.success = False
        self.overruns = 0
        self.idle_ticks = 0
        self.actions_applied = []
        self._stop.clear()
        self._done.clear()
        return obs

    def start(self):
        self._thread = threading.Thread(target=self._physics_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def done(self):
        return self._done.is_set()

    def latest_obs(self):
        """Returns (obs_dict, tick at which it was observed)."""
        with self._lock:
            return copy.deepcopy(self._obs), self._obs_step

    def push_action(self, action):
        with self._lock:
            self._action = np.asarray(action, dtype=np.float64).copy()
            self._action_updates += 1

    # ---------------- physics thread ----------------
    def _physics_loop(self):
        t0 = time.perf_counter()
        last_update_id = -1
        for tick in range(self.horizon):
            if self._stop.is_set():
                break
            target = t0 + tick * self.wall_dt
            now = time.perf_counter()
            if now < target:
                time.sleep(target - now)
            elif now - target > self.wall_dt:
                self.overruns += 1

            with self._lock:
                action = self._action.copy()
                update_id = self._action_updates
            if update_id == last_update_id:
                self.idle_ticks += 1
                if self.hold == "stop":
                    action[:-1] = 0.0  # no arm motion; last dim is the gripper command
            last_update_id = update_id
            obs, _, _, _ = self.env.step(action)
            success = bool(self.env.is_success()["task"])
            if self.record:
                self.states.append(self.env.get_state()["states"])

            with self._lock:
                self._obs = obs
                self._obs_step = tick + 1
            self.actions_applied.append((tick, update_id))
            self.step_count = tick + 1
            if success:
                self.success = True
                break
        self._done.set()

"""
Asynchronous (real-time) simulation with TWO processes.

In a normal (synchronous) rollout the world freezes while the policy thinks:
    obs -> policy(obs) -> env.step(action) -> obs -> ...

Here the physics runs in its own *process* and steps the env at a fixed
wall-clock rate (control_freq * real_time_rate) no matter what the policy is
doing. The policy process reads the most recent observation and writes the most
recent action through shared memory:

    policy process (parent)                 physics process (child)
    -----------------------                 -----------------------
    obs = latest_obs()   <-- shared mem --  write obs after every tick
    action = policy(obs)
    push_action(action)  -- shared mem -->  read action before every tick

Control messages (reset / run / result / close) go over a Pipe. Separate
processes mean no Python GIL contention between physics and inference, and
mirror a real robot where the controller and the policy server are separate.
"""
import multiprocessing as mp
import time
import traceback

import numpy as np

# indices into the shared int64 counter array
OBS_STEP, ACT_UPDATES, STEP_COUNT, DONE, STOP, IDLE = range(6)


class _Layout:
    """Maps an obs dict {key: array} <-> one flat float64 buffer."""

    def __init__(self, obs_spec):
        self.items, off = [], 0
        for k, shape in obs_spec.items():
            n = int(np.prod(shape)) if len(shape) else 1
            self.items.append((k, tuple(shape), off, n))
            off += n
        self.size = off

    def write(self, buf, obs):
        for k, _, off, n in self.items:
            buf[off:off + n] = np.asarray(obs[k], dtype=np.float64).ravel()

    def read(self, buf):
        return {k: buf[off:off + n].reshape(shape).copy() for k, shape, off, n in self.items}


def _physics_main(env_fn, conn, obs_arr, act_arr, cnt_arr, lock, obs_spec, ac_dim, cfg):
    """Entry point of the physics process."""
    try:
        env = env_fn()
        layout = _Layout(obs_spec)
        obs_buf = np.frombuffer(obs_arr, dtype=np.float64)
        act_buf = np.frombuffer(act_arr, dtype=np.float64)
        cnt = np.frombuffer(cnt_arr, dtype=np.int64)
        conn.send(("ready", None))
    except Exception:
        conn.send(("error", traceback.format_exc()))
        return

    wall_dt = 1.0 / (cfg["control_freq"] * cfg["real_time_rate"])
    while True:
        cmd, arg = conn.recv()
        try:
            if cmd == "close":
                break

            if cmd == "reset":
                env.reset()
                obs = env.reset_to({"states": arg})
                with lock:
                    layout.write(obs_buf, obs)
                    act_buf[:] = 0.0
                    cnt[:] = 0
                conn.send(("ok", None))

            elif cmd == "run":
                record, hold = arg
                states = [env.get_state()["states"]] if record else None
                success, overruns, last_update = False, 0, -1
                t0 = time.perf_counter()
                for tick in range(cfg["horizon"]):
                    if cnt[STOP]:
                        break
                    target = t0 + tick * wall_dt
                    now = time.perf_counter()
                    if now < target:
                        time.sleep(target - now)
                    elif now - target > wall_dt:
                        # fell behind (e.g. a slow step): count it and re-anchor the
                        # schedule instead of bursting ticks to catch up
                        overruns += 1
                        t0 = now - tick * wall_dt

                    with lock:
                        action = act_buf.copy()
                        update = int(cnt[ACT_UPDATES])
                    if update == last_update:  # no new action since last tick
                        cnt[IDLE] += 1
                        if hold == "stop":
                            action[:-1] = 0.0  # freeze the arm, keep the gripper command
                    last_update = update

                    obs, _, _, _ = env.step(action)
                    success = bool(env.is_success()["task"])
                    if record:
                        states.append(env.get_state()["states"])
                    with lock:
                        layout.write(obs_buf, obs)
                        cnt[OBS_STEP] = tick + 1
                        cnt[STEP_COUNT] = tick + 1
                    if success:
                        break
                cnt[DONE] = 1
                conn.send(("result", dict(success=success, overruns=overruns, states=states)))
        except Exception:
            cnt[DONE] = 1
            conn.send(("error", traceback.format_exc()))


class AsyncEnv:
    """Policy-side handle to the physics process. Create once, reuse for many
    episodes (the env is built only once inside the child process)."""

    def __init__(self, env_fn, obs_spec, ac_dim, control_freq=20.0, real_time_rate=1.0,
                 horizon=400):
        """
        Args:
            env_fn: picklable zero-arg callable that builds the env *inside* the
                physics process (e.g. functools.partial of a top-level function)
            obs_spec: {obs_key: shape} of the observations the env returns
            ac_dim: action dimension
        """
        self.wall_dt = 1.0 / (control_freq * real_time_rate)
        self._layout = _Layout(obs_spec)

        ctx = mp.get_context("spawn")  # fresh interpreter: safe with torch/CUDA/MuJoCo
        self._lock = ctx.Lock()
        self._obs_arr = ctx.RawArray("d", self._layout.size)
        self._act_arr = ctx.RawArray("d", ac_dim)
        self._cnt_arr = ctx.RawArray("q", 6)
        self._obs = np.frombuffer(self._obs_arr, dtype=np.float64)
        self._act = np.frombuffer(self._act_arr, dtype=np.float64)
        self._cnt = np.frombuffer(self._cnt_arr, dtype=np.int64)

        self._conn, child_conn = ctx.Pipe()
        cfg = dict(control_freq=control_freq, real_time_rate=real_time_rate, horizon=horizon)
        self._proc = ctx.Process(
            target=_physics_main, daemon=True,
            args=(env_fn, child_conn, self._obs_arr, self._act_arr, self._cnt_arr,
                  self._lock, obs_spec, ac_dim, cfg))
        self._proc.start()
        self._expect("ready")
        self._result = None

    def _expect(self, tag):
        msg, payload = self._conn.recv()
        if msg == "error":
            raise RuntimeError("physics process failed:\n" + payload)
        assert msg == tag, (msg, tag)
        return payload

    # ---------------- per-episode API ----------------
    def reset(self, init_state):
        self._conn.send(("reset", init_state))
        self._expect("ok")
        self._result = None

    def start(self, hold="repeat", record=False):
        """Start one real-time episode in the physics process.
        hold: what the robot does on ticks with no new action
            "repeat": re-apply the last action (zero-order hold). With delta-pose
                      control this keeps moving and overshoots.
            "stop":   apply each action once, then zero the arm command (keep the
                      gripper command) until the next action arrives.
        """
        assert hold in ("repeat", "stop")
        self._conn.send(("run", (record, hold)))

    def done(self):
        return bool(self._cnt[DONE])

    def latest_obs(self):
        """Returns (obs_dict, tick at which it was observed)."""
        with self._lock:
            return self._layout.read(self._obs), int(self._cnt[OBS_STEP])

    def push_action(self, action):
        with self._lock:
            self._act[:] = np.asarray(action, dtype=np.float64)
            self._cnt[ACT_UPDATES] += 1

    @property
    def step_count(self):
        return int(self._cnt[STEP_COUNT])

    def finish(self):
        """Stop the episode (if still running) and return the physics-side result."""
        if not self.done():
            self._cnt[STOP] = 1
        self._result = self._expect("result")
        self._result.update(steps=self.step_count, idle_ticks=int(self._cnt[IDLE]))
        return self._result

    def close(self):
        if self._proc.is_alive():
            self._conn.send(("close", None))
            self._proc.join(timeout=10)
        if self._proc.is_alive():
            self._proc.terminate()

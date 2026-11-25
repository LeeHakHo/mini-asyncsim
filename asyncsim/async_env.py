"""
Real-time simulation in a separate physics process.

The physics process steps the env at a fixed wall-clock rate. The policy process
reads the latest observation and writes the latest action through shared memory.
Control messages (reset, run, result, close) are sent over a Pipe.
"""
import multiprocessing as mp
import time
import traceback

import numpy as np

# Indices into the shared counter array.
OBS_STEP, ACT_UPDATES, STEP_COUNT, DONE, STOP, IDLE = range(6)


class _Layout:
    """Map an observation dict to and from a flat float64 buffer."""

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
                        # Behind schedule: count an overrun and re-anchor.
                        overruns += 1
                        t0 = now - tick * wall_dt

                    with lock:
                        action = act_buf.copy()
                        update = int(cnt[ACT_UPDATES])
                    if update == last_update:  # No new action since the last tick.
                        cnt[IDLE] += 1
                        if hold == "stop":
                            action[:-1] = 0.0  # Zero the arm command, keep the gripper command.
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
    """Policy-side handle to the physics process. Reusable across episodes."""

    def __init__(self, env_fn, obs_spec, ac_dim, control_freq=20.0, real_time_rate=1.0,
                 horizon=400):
        """
        Args:
            env_fn: picklable callable that builds the env in the physics process
            obs_spec: {obs_key: shape} of the env observations
            ac_dim: action dimension
        """
        self.wall_dt = 1.0 / (control_freq * real_time_rate)
        self._layout = _Layout(obs_spec)

        ctx = mp.get_context("spawn")
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

    def reset(self, init_state):
        self._conn.send(("reset", init_state))
        self._expect("ok")
        self._result = None

    def start(self, hold="repeat", record=False):
        """Start one real-time episode.

        hold: behavior on ticks with no new action
            "repeat": re-apply the last action
            "stop": zero the arm command and keep the gripper command
        """
        assert hold in ("repeat", "stop")
        self._conn.send(("run", (record, hold)))

    def done(self):
        return bool(self._cnt[DONE])

    def latest_obs(self):
        """Return (obs, tick of the observation)."""
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
        """Stop the episode if running and return its result."""
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

"""
Record sync / async_repeat / async_stop rollouts of the same policy from the
SAME initial state, then render each to mp4 (plus a side-by-side comparison).

Async episodes run with physics in a separate process (asyncsim.AsyncEnv).
Only sim states are recorded during the rollouts. Rendering happens afterwards
in a fresh process that builds its own offscreen env and replays the states:
offscreen rendering in the process that ran the async rollouts returned
corrupted frames.

    python scripts/make_video.py --ckpt runs/bc/.../last.pth --extra_latency 0.2 --out videos_new/bc_lat200
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import multiprocessing as mp
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODES = ["sync", "async_repeat", "async_stop"]


def write(path, frames, fps):
    import imageio
    with imageio.get_writer(path, fps=fps) as w:
        for f in frames:
            w.append_data(f)
    print("wrote", path, len(frames), "frames", flush=True)


def render_worker(ckpt, recorded, out, size, camera, control_freq):
    """Runs in a fresh process: build an offscreen env, replay states, write mp4s."""
    from PIL import Image, ImageDraw
    from asyncsim.envs import env_from_ckpt

    env = env_from_ckpt(ckpt, render_offscreen=True)
    results = {}
    for mode, (label, states) in recorded.items():
        frames = []
        for t, s in enumerate(states):
            env.reset_to({"states": s})
            img = env.render(mode="rgb_array", height=size, width=size, camera_name=camera)
            im = Image.fromarray(np.array(img, dtype=np.uint8, copy=True))
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, size, 22], fill=(0, 0, 0))
            d.text((6, 5), f"{label}   t={t / control_freq:5.2f}s", fill=(255, 255, 255))
            frames.append(np.array(im))
        results[mode] = frames

    fps = int(control_freq)  # one frame per control tick -> plays in real (sim) time
    for mode in MODES:
        write(f"{out}_{mode}.mp4", results[mode], fps)
    n = max(len(results[m]) for m in MODES)
    padded = [results[m] + [results[m][-1]] * (n - len(results[m])) for m in MODES]
    write(f"{out}_side_by_side.mp4", [np.concatenate(f, axis=1) for f in zip(*padded)], fps)


def main():
    import torch
    from eval import load, make_async_env
    from asyncsim.rollout import async_rollout, sync_rollout

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--extra_latency", type=float, default=0.0)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--control_freq", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--camera", default="frontview")
    p.add_argument("--out", default="videos_new/run", help="output prefix")
    a = p.parse_args()

    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    policy, env, env_fn, fs, algo = load(a.ckpt, render_offscreen=False)
    aenv = make_async_env(env, env_fn, a.control_freq, 1.0, a.horizon)

    np.random.seed(a.seed)
    env.reset()
    init_state = env.get_state()["states"]
    lat_ms = int(round(a.extra_latency * 1000))

    recorded = {}
    try:
        for mode in MODES:
            np.random.seed(a.seed)
            torch.manual_seed(a.seed)
            if mode == "sync":
                r = sync_rollout(policy, env, a.horizon, fs, a.extra_latency,
                                 record=True, init_state=init_state)
            else:
                r = async_rollout(policy, aenv, env, fs, a.extra_latency,
                                  hold=mode.split("_")[1], record=True, init_state=init_state)
            status = "SUCCESS" if r["success"] else "FAIL"
            label = f"{algo} {mode.upper()} +{lat_ms}ms  {status}"
            print(label, {k: v for k, v in r.items() if k != "states"}, flush=True)
            recorded[mode] = (label, [np.asarray(s) for s in r["states"]])
    finally:
        aenv.close()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    proc = mp.get_context("spawn").Process(
        target=render_worker,
        args=(a.ckpt, recorded, a.out, a.size, a.camera, a.control_freq))
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise SystemExit(f"render process failed with exit code {proc.exitcode}")


if __name__ == "__main__":
    main()

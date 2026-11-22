"""
Record one sync and one async rollout of the same policy from the SAME initial
state, then render both to mp4 (plus a side-by-side comparison).

Rendering happens after the rollout by replaying the recorded sim states, so it
does not slow down the real-time physics thread.

    python scripts/make_video.py --ckpt runs/bc/.../last.pth --extra_latency 0.2 --out videos/bc
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import sys

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

from asyncsim.rollout import async_rollout, sync_rollout


def render(env, states, label, size, camera):
    frames = []
    for t, s in enumerate(states):
        env.reset_to({"states": s})
        img = env.render(mode="rgb_array", height=size, width=size, camera_name=camera)
        im = Image.fromarray(np.asarray(img, dtype=np.uint8))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, size, 22], fill=(0, 0, 0))
        d.text((6, 5), f"{label}   t={t / 20.0:5.2f}s", fill=(255, 255, 255))
        frames.append(np.asarray(im))
    return frames


def write(path, frames, fps):
    with imageio.get_writer(path, fps=fps) as w:
        for f in frames:
            w.append_data(f)
    print("wrote", path, len(frames), "frames")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--extra_latency", type=float, default=0.0)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--control_freq", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--camera", default="agentview")
    p.add_argument("--out", default="videos/run", help="output prefix")
    a = p.parse_args()

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt = FileUtils.policy_from_checkpoint(ckpt_path=a.ckpt, device=device, verbose=False)
    env, _ = FileUtils.env_from_checkpoint(ckpt_dict=ckpt, render=False, render_offscreen=True, verbose=False)
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt)
    from robomimic.envs.wrappers import FrameStackWrapper
    # env_from_checkpoint already stacks frames when frame_stack > 1
    fs = 1 if isinstance(env, FrameStackWrapper) else config.train.frame_stack
    algo = config.algo_name

    np.random.seed(a.seed)
    env.reset()
    init_state = env.get_state()["states"]
    lat_ms = int(round(a.extra_latency * 1000))

    modes = ["sync", "async_repeat", "async_stop"]
    results = {}
    for mode in modes:
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
        if mode == "sync":
            r = sync_rollout(policy, env, a.horizon, fs, a.extra_latency,
                             record=True, init_state=init_state)
        else:
            r = async_rollout(policy, env, a.horizon, a.control_freq, 1.0, fs,
                              a.extra_latency, record=True, init_state=init_state,
                              hold=mode.split("_")[1])
        status = "SUCCESS" if r["success"] else "FAIL"
        label = f"{algo} {mode.upper()} +{lat_ms}ms  {status}"
        print(label, {k: v for k, v in r.items() if k != "states"})
        results[mode] = render(env, r["states"], label, a.size, a.camera)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fps = int(a.control_freq)  # one frame per control tick -> plays in real time
    for mode in modes:
        write(f"{a.out}_{mode}.mp4", results[mode], fps)

    n = max(len(results[m]) for m in modes)
    pad = lambda fr: fr + [fr[-1]] * (n - len(fr))
    padded = [pad(results[m]) for m in modes]
    side = [np.concatenate(fs_, axis=1) for fs_ in zip(*padded)]
    write(f"{a.out}_side_by_side.mp4", side, fps)


if __name__ == "__main__":
    main()

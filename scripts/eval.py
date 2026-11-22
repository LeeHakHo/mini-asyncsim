"""
Evaluate one configuration (policy, mode, latency, ...) and append results to a CSV.

Examples:
    # smoke test without a trained policy (random actions)
    python scripts/eval.py --dummy --dataset datasets/lift/ph/low_dim_v15.hdf5 --mode async -n 2

    # trained policy, async, +50 ms artificial latency
    python scripts/eval.py --ckpt runs/bc/.../last.pth --mode async --extra_latency 0.05 -n 50
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils

from asyncsim.rollout import async_rollout, sync_rollout

LOW_DIM_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]


class RandomPolicy:
    def __init__(self, ac_dim):
        self.ac_dim = ac_dim

    def start_episode(self):
        pass

    def __call__(self, ob):
        return np.random.uniform(-1, 1, self.ac_dim)


def load(args):
    if args.dummy:
        ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": LOW_DIM_KEYS}})
        env_meta = FileUtils.get_env_metadata_from_dataset(args.dataset)
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta, render=False, render_offscreen=False, use_image_obs=False)
        return RandomPolicy(env.action_dimension), env, 1, "random"

    device = TorchUtils.get_torch_device(try_to_use_cuda=(args.device == "cuda"))
    policy, ckpt = FileUtils.policy_from_checkpoint(ckpt_path=args.ckpt, device=device, verbose=False)
    env, _ = FileUtils.env_from_checkpoint(ckpt_dict=ckpt, render=False, render_offscreen=False, verbose=False)
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt)
    return policy, env, policy_frame_stack(env, config), config.algo_name


def policy_frame_stack(env, config):
    """robomimic's env_from_checkpoint already wraps the env in a FrameStackWrapper
    when frame_stack > 1; only stack ourselves if it did not."""
    from robomimic.envs.wrappers import FrameStackWrapper
    return 1 if isinstance(env, FrameStackWrapper) else config.train.frame_stack


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt")
    p.add_argument("--dummy", action="store_true")
    p.add_argument("--dataset", help="needed with --dummy to build the env")
    p.add_argument("--mode", choices=["sync", "async"], default="async")
    p.add_argument("--hold", choices=["repeat", "stop"], default="repeat",
                   help="async only: repeat last action or stop the arm while waiting")
    p.add_argument("-n", "--n_rollouts", type=int, default=10)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--control_freq", type=float, default=20.0)
    p.add_argument("--real_time_rate", type=float, default=1.0)
    p.add_argument("--extra_latency", type=float, default=0.0, help="seconds added to each policy call")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/results.csv")
    p.add_argument("--tag", default="", help="free-form label stored in the CSV")
    args = p.parse_args()
    assert args.dummy or args.ckpt, "need --ckpt or --dummy"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    policy, env, frame_stack, algo = load(args)

    rows = []
    for i in range(args.n_rollouts):
        if args.mode == "sync":
            r = sync_rollout(policy, env, args.horizon, frame_stack, args.extra_latency)
        else:
            r = async_rollout(policy, env, args.horizon, args.control_freq,
                              args.real_time_rate, frame_stack, args.extra_latency,
                              hold=args.hold)
        r.pop("states", None)
        r.update(algo=algo, rollout=i, seed=args.seed, extra_latency=args.extra_latency,
                 control_freq=args.control_freq, real_time_rate=args.real_time_rate,
                 device=args.device, tag=args.tag, ckpt=args.ckpt or "")
        rows.append(r)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)

    print(f"== {algo} {rows[0]['mode']} latency+{args.extra_latency}s: "
          f"success {np.mean([r['success'] for r in rows]):.2%} over {len(rows)} rollouts")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

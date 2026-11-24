"""
Write a robomimic training config (low-dim) from robomimic's own templates.

    python scripts/make_config.py bc                       -> configs/lift_bc.json
    python scripts/make_config.py diffusion_policy square  -> configs/square_diffusion_policy.json
    python scripts/make_config.py diffusion_policy square 3000  -> configs/square_diffusion_policy_e3000.json

Tasks: lift (pick up a cube), square (fit a square nut onto a peg: an insertion task).
"""
import json
import os
import sys

import robomimic

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EPOCHS = {"lift": 600, "square": 1000}  # square is longer and harder -> train longer


def dataset_path(task):
    return os.path.join(ROOT, f"datasets/{task}/ph/low_dim_v15.hdf5")


def main(algo, task="lift", epochs=None):
    name = f"{task}_{algo}" + (f"_e{epochs}" if epochs else "")
    tmpl = os.path.join(robomimic.__path__[0], "exps/templates", f"{algo}.json")
    with open(tmpl) as f:
        c = json.load(f)

    c["experiment"]["name"] = name
    c["experiment"]["logging"]["log_wandb"] = False
    # we evaluate ourselves (sync vs async), so skip robomimic's rollouts during training
    c["experiment"]["rollout"]["enabled"] = False
    c["experiment"]["save"]["enabled"] = True
    c["experiment"]["save"]["every_n_epochs"] = 100
    c["experiment"]["validate"] = False

    c["train"]["data"] = [{"path": dataset_path(task)}]
    c["train"]["output_dir"] = os.path.join(ROOT, "runs", algo)
    c["train"]["num_epochs"] = int(epochs) if epochs else EPOCHS[task]

    if algo == "diffusion_policy":
        h = c["algo"]["horizon"]
        c["train"]["frame_stack"] = h["observation_horizon"]  # 2
        c["train"]["seq_length"] = h["prediction_horizon"]    # 16

    os.makedirs(os.path.join(ROOT, "configs"), exist_ok=True)
    out = os.path.join(ROOT, "configs", f"{name}.json")
    with open(out, "w") as f:
        json.dump(c, f, indent=4)
    print(out)


if __name__ == "__main__":
    main(*sys.argv[1:4])

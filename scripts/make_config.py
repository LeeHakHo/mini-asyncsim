"""
Write a robomimic training config for Lift (low-dim) from robomimic's own templates.

    python scripts/make_config.py bc               -> configs/bc.json
    python scripts/make_config.py diffusion_policy -> configs/diffusion_policy.json
"""
import json
import os
import sys

import robomimic

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET = os.path.join(ROOT, "datasets/lift/ph/low_dim_v15.hdf5")

EPOCHS = {"bc": 600, "diffusion_policy": 600}


def main(algo):
    tmpl = os.path.join(robomimic.__path__[0], "exps/templates", f"{algo}.json")
    with open(tmpl) as f:
        c = json.load(f)

    c["experiment"]["name"] = f"lift_{algo}"
    c["experiment"]["logging"]["log_wandb"] = False
    # we evaluate ourselves (sync vs async), so skip robomimic's rollouts during training
    c["experiment"]["rollout"]["enabled"] = False
    c["experiment"]["save"]["enabled"] = True
    c["experiment"]["save"]["every_n_epochs"] = 100
    c["experiment"]["validate"] = False

    c["train"]["data"] = [{"path": DATASET}]
    c["train"]["output_dir"] = os.path.join(ROOT, "runs", algo)
    c["train"]["num_epochs"] = EPOCHS[algo]

    if algo == "diffusion_policy":
        h = c["algo"]["horizon"]
        c["train"]["frame_stack"] = h["observation_horizon"]  # 2
        c["train"]["seq_length"] = h["prediction_horizon"]    # 16

    os.makedirs(os.path.join(ROOT, "configs"), exist_ok=True)
    out = os.path.join(ROOT, "configs", f"{algo}.json")
    with open(out, "w") as f:
        json.dump(c, f, indent=4)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1])

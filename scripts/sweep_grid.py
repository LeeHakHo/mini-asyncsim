"""
Defines the evaluation grid and prints the eval.py arguments for one index.
Used by carc/sweep.slurm: each Slurm array task evaluates one grid cell.

    python scripts/sweep_grid.py --count        # number of cells
    python scripts/sweep_grid.py --index 7      # args for cell 7
"""
import argparse
import itertools
import os

CKPTS = {  # name -> checkpoint path (set via env vars so the grid stays generic)
    "bc": os.environ.get("BC_CKPT", ""),
    "diffusion_policy": os.environ.get("DP_CKPT", ""),
}
MODES = {  # name -> eval.py args
    "sync": "--mode sync",
    "async_repeat": "--mode async --hold repeat",
    "async_stop": "--mode async --hold stop",
}
LATENCIES = [0.0, 0.1, 0.2, 0.4, 0.8]  # extra seconds per policy call
SEED = 0
HORIZON = int(os.environ.get("HORIZON", 400))  # control ticks per episode


def grid():
    return list(itertools.product(CKPTS.keys(), MODES.keys(), LATENCIES))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--count", action="store_true")
    p.add_argument("--index", type=int)
    a = p.parse_args()
    g = grid()
    if a.count:
        print(len(g))
    else:
        algo, mode, lat = g[a.index]
        print(f"--ckpt {CKPTS[algo]} {MODES[mode]} --extra_latency {lat} --seed {SEED} --horizon {HORIZON} --tag {algo}")

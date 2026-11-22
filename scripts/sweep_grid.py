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
MODES = ["sync", "async"]
LATENCIES = [0.0, 0.025, 0.05, 0.1, 0.2, 0.4]  # extra seconds per policy call
SEEDS = [0, 1]


def grid():
    return list(itertools.product(CKPTS.keys(), MODES, LATENCIES, SEEDS))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--count", action="store_true")
    p.add_argument("--index", type=int)
    a = p.parse_args()
    g = grid()
    if a.count:
        print(len(g))
    else:
        algo, mode, lat, seed = g[a.index]
        print(f"--ckpt {CKPTS[algo]} --mode {mode} --extra_latency {lat} --seed {seed} --tag {algo}")

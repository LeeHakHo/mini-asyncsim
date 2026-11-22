"""
Plot a sweep: success rate and time-to-success vs. extra policy latency,
one column per policy, one line per mode (sync / async_repeat / async_stop).

    python scripts/plot.py results/sweep_<jobid>            -> results/sweep_<jobid>/sweep.png + summary.csv
"""
import csv
import glob
import math
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CONTROL_FREQ = 20.0
HORIZON_S = 400 / CONTROL_FREQ  # async episodes end after 400 ticks
MODES = [  # fixed order -> fixed colour (validated categorical slots 1-3) + marker
    ("sync", "sync (world waits)", "#2a78d6", "o"),
    ("async_repeat", "async: repeat last action", "#eb6834", "s"),
    ("async_stop", "async: stop until next action", "#1baf7a", "^"),
]
ALGOS = [("bc", "BC"), ("diffusion_policy", "Diffusion Policy")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load(sweep_dir):
    cells = defaultdict(list)  # (algo, mode, latency) -> rows
    for f in glob.glob(os.path.join(sweep_dir, "task_*.csv")):
        for r in csv.DictReader(open(f)):
            cells[(r["algo"], r["mode"], float(r["extra_latency"]))].append(r)
    return cells


def summarize(cells):
    out = []
    for (algo, mode, lat), rows in sorted(cells.items()):
        n = len(rows)
        ok = [r for r in rows if float(r["success"]) > 0]
        lo, hi = wilson(len(ok), n)
        t = sorted(int(r["steps"]) / CONTROL_FREQ for r in ok)
        out.append(dict(algo=algo, mode=mode, extra_latency_ms=int(round(lat * 1000)), n=n,
                        success=len(ok) / n, ci_lo=lo, ci_hi=hi,
                        median_time_s=t[len(t) // 2] if t else float("nan"),
                        mean_policy_latency_ms=1000 * sum(float(r["mean_latency"]) for r in rows) / n))
    return out


def main(sweep_dir):
    summary = summarize(load(sweep_dir))
    with open(os.path.join(sweep_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for col, (algo, algo_name) in enumerate(ALGOS):
        for mode, label, color, marker in MODES:
            pts = sorted((s for s in summary if s["algo"] == algo and s["mode"] == mode),
                         key=lambda s: s["extra_latency_ms"])
            if not pts:
                continue
            x = [s["extra_latency_ms"] for s in pts]
            ax = axes[0, col]
            ax.fill_between(x, [100 * s["ci_lo"] for s in pts], [100 * s["ci_hi"] for s in pts],
                            color=color, alpha=0.12, linewidth=0)
            ax.plot(x, [100 * s["success"] for s in pts], color=color, marker=marker,
                    markersize=7, linewidth=2, label=label)
            axes[1, col].plot(x, [s["median_time_s"] for s in pts], color=color, marker=marker,
                              markersize=7, linewidth=2, label=label)
        axes[0, col].set_title(algo_name, color=INK, fontsize=12, loc="left")
        axes[0, col].set_ylim(0, 105)
        axes[1, col].axhline(HORIZON_S, color=MUTED, linestyle="--", linewidth=1)
        axes[1, col].text(0, HORIZON_S, " episode time limit", color=MUTED, fontsize=8.5, va="bottom")
        axes[1, col].set_ylim(0, HORIZON_S + 2)
        axes[1, col].set_xlabel("extra policy latency (ms)")
    axes[0, 0].set_ylabel("success rate (%)")
    axes[1, 0].set_ylabel("median time to success (s, sim time)")
    for ax in axes.flat:
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.005, "Lift task, 20 Hz control, 50 rollouts per point; "
             "shaded = 95% Wilson interval. Diffusion Policy also has its own inference time.",
             ha="center", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    out = os.path.join(sweep_dir, "sweep.png")
    fig.savefig(out, dpi=160, facecolor="#fcfcfb")
    print("wrote", out)
    for s in summary:
        print(s)


if __name__ == "__main__":
    main(sys.argv[1])

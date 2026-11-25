"""
Plot success rate vs. extra latency for several tasks from plot.py summaries.

    python scripts/compare.py docs/compare.png Lift=docs/summary.csv Square=docs/square/summary.csv
"""
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plot import ALGOS, GRID, INK, MODES, MUTED


def read(path):
    return [dict(r, extra_latency_ms=int(r["extra_latency_ms"]), success=float(r["success"]),
                 ci_lo=float(r["ci_lo"]), ci_hi=float(r["ci_hi"])) for r in csv.DictReader(open(path))]


def main(out, specs):
    tasks = [s.split("=", 1) for s in specs]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(len(tasks), len(ALGOS), figsize=(10, 3.4 * len(tasks)),
                             sharex=True, sharey=True, squeeze=False)
    for row, (task, path) in enumerate(tasks):
        summary = read(path)
        for col, (algo, algo_name) in enumerate(ALGOS):
            ax = axes[row, col]
            for mode, label, color, marker in MODES:
                pts = sorted((s for s in summary if s["algo"] == algo and s["mode"] == mode),
                             key=lambda s: s["extra_latency_ms"])
                if not pts:
                    continue
                x = [s["extra_latency_ms"] for s in pts]
                ax.fill_between(x, [100 * s["ci_lo"] for s in pts], [100 * s["ci_hi"] for s in pts],
                                color=color, alpha=0.12, linewidth=0)
                ax.plot(x, [100 * s["success"] for s in pts], color=color, marker=marker,
                        markersize=7, linewidth=2, label=label)
            ax.set_title(f"{task} · {algo_name}", color=INK, fontsize=12, loc="left")
            ax.set_ylim(0, 105)
            ax.grid(True, color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        axes[row, 0].set_ylabel("success rate (%)")
    for ax in axes[-1]:
        ax.set_xlabel("extra policy latency (ms)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.005, "20 Hz control, 50 rollouts per point; shaded = 95% Wilson interval. "
             "Lift episodes end after 20 s, Square after 40 s.", ha="center", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(out, dpi=160, facecolor="#fcfcfb")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])

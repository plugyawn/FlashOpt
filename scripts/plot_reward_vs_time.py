#!/usr/bin/env python3
"""
Reconstruct a "best train reward so far vs wallclock" plot for a RandOpt speedrun
from Modal logs (or any log file with timestamped sampling-batch lines of the
form: `<ts>  Batch N | N/512 | ['0.703']`).

This is the optimization curve: random search's progress is the running max of
per-seed train reward over time, so the staircase shows how quickly perturbation
search finds better-than-base weights.

Usage:
  # straight from Modal (recommended — gets the timestamps):
  modal app logs <app-id> --tail 5000 --timestamps | \
      python scripts/plot_reward_vs_time.py --out reward_vs_time.png --base 0.625
  # or from a saved log file:
  python scripts/plot_reward_vs_time.py --log mylog.txt --out p.png --base 0.625
"""
import argparse
import re
import sys
from datetime import datetime

# `2026-05-30 16:27:17+05:30  Batch 140 | 140/512 | ['0.641']`
LINE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[+\d:]*\s+Batch\s+(\d+)\s*\|\s*(\d+)/(\d+)\s*\|\s*\['([0-9.]+)'\]"
)


def parse(lines):
    """Return list of (elapsed_seconds, batch, reward), t0 = first batch time."""
    rows = []
    for ln in lines:
        m = LINE.search(ln)
        if m:
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            rows.append((ts, int(m.group(2)), int(m.group(3)), int(m.group(4)), float(m.group(5))))
    if not rows:
        return [], 0
    rows.sort(key=lambda r: r[1])             # by batch index
    t0 = rows[0][0]
    pop = rows[0][4] if False else rows[0][3]  # population from N/POP
    return [((r[0] - t0).total_seconds(), r[1], r[4]) for r in rows], pop


def running_best(points):
    best = -1.0
    out = []
    for elapsed, batch, reward in points:
        best = max(best, reward)
        out.append((elapsed, batch, reward, best))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="log file (default: stdin)")
    ap.add_argument("--out", default="reward_vs_time.png")
    ap.add_argument("--base", type=float, default=None, help="base model train reward (draws a reference line)")
    ap.add_argument("--title", default="RandOpt: best train reward vs wallclock")
    args = ap.parse_args()

    lines = open(args.log) if args.log else sys.stdin
    points, pop = parse(lines)
    if not points:
        print("no timestamped batch lines found (need `modal app logs --timestamps`)", file=sys.stderr)
        sys.exit(1)

    series = running_best(points)
    xs = [p[0] / 60.0 for p in series]         # minutes
    per_seed = [p[2] for p in series]
    best = [p[3] for p in series]
    n_done = series[-1][1]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(xs, per_seed, s=10, alpha=0.35, color="#888", label="per-seed train reward")
    ax.step(xs, best, where="post", color="#1f77b4", lw=2.2, label="best so far (running max)")
    if args.base is not None:
        ax.axhline(args.base, color="#d62728", ls="--", lw=1.5, label=f"base model ({args.base:.3f})")

    ax.set_xlabel("wallclock (minutes since first seed)")
    ax.set_ylabel("train reward")
    ax.set_title(f"{args.title}\n{n_done}/{pop} seeds · best={best[-1]:.3f}"
                 + (f" (+{best[-1]-args.base:.3f} vs base)" if args.base is not None else ""))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}  ({n_done}/{pop} seeds, best={best[-1]:.3f}, "
          f"{xs[-1]:.1f} min elapsed)")


if __name__ == "__main__":
    main()

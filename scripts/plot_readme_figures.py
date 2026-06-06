#!/usr/bin/env python3
"""Generate README figures from checked-in RandOpt result artifacts."""

from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"


def load_json(path: str | Path) -> dict:
    with Path(path).open() as f:
        return json.load(f)


def save(fig, name: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / name
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"wrote {out.relative_to(ROOT)}")


def pct_acc(value: float) -> float:
    return value * 100.0 if value <= 1.0 else value


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )
    return plt


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def plot_k1_wallclock_hero(plt):
    runs = [
        ("default", "hotpath-runs/sweep1b_default"),
        ("train32", "hotpath-runs/sweep1b_train32"),
        ("train16", "hotpath-runs/sweep1b_train16"),
        ("lvl5 train", "hotpath-runs/sweep1b_lvl5train"),
        ("stratified", "hotpath-runs/sweep1b_stratify"),
        ("rad 1e-4", "hotpath-runs/sweepscheme_sig1e4"),
        ("rad 1e-3", "hotpath-runs/sweepscheme_sig1e3"),
        ("gauss 5e-4", "hotpath-runs/sweepscheme_gauss5e4"),
        ("gauss 1e-3", "hotpath-runs/sweepscheme_gauss1e3"),
    ]

    curves_by_test_sequence = {}
    for label, rel in runs:
        run = ROOT / rel
        rec = load_json(run / "record.json")
        rows = sorted(load_jsonl(run / "seeds.jsonl"), key=lambda row: row["idx"])
        sequence_key = tuple(round(row["test_acc"], 10) for row in rows)
        if sequence_key in curves_by_test_sequence:
            curves_by_test_sequence[sequence_key]["label"] += f", {label}"
            continue
        sampling_minutes = rec["timings_s"]["sampling"] / 60.0
        elapsed = [((i + 1) / len(rows)) * sampling_minutes for i in range(len(rows))]
        running_best = []
        best = -1e9
        for row in rows:
            best = max(best, row["test_acc"])
            running_best.append(best - rec["base_test_accuracy"])
        curves_by_test_sequence[sequence_key] = {
            "label": label,
            "elapsed": elapsed,
            "running_best_gain": running_best,
            "final_gain": running_best[-1],
            "duration": sampling_minutes,
        }
    curves = list(curves_by_test_sequence.values())

    def value_at(curve, minute):
        idx = bisect_right(curve["elapsed"], minute) - 1
        return 0.0 if idx < 0 else curve["running_best_gain"][idx]

    def percentile(values, q):
        vals = sorted(values)
        if len(vals) == 1:
            return vals[0]
        pos = (len(vals) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(vals) - 1)
        frac = pos - lo
        return vals[lo] * (1 - frac) + vals[hi] * frac

    max_duration = max(curve["duration"] for curve in curves)
    grid = [max_duration * i / 180 for i in range(181)]
    per_minute = [[value_at(curve, minute) for curve in curves] for minute in grid]
    median = [percentile(vals, 0.50) for vals in per_minute]
    q25 = [percentile(vals, 0.25) for vals in per_minute]
    q75 = [percentile(vals, 0.75) for vals in per_minute]
    lo = [min(vals) for vals in per_minute]
    hi = [max(vals) for vals in per_minute]
    final_positive = sum(curve["final_gain"] > 0 for curve in curves)
    sign_p = 0.5 ** len(curves) if final_positive == len(curves) else None

    fig, ax = plt.subplots(figsize=(11.8, 5.2))
    palette = ["#2B8CBE", "#2F855A", "#D9902F", "#8F5252", "#6B5B95", "#4C566A", "#8A8F35", "#C26D3D", "#6A9FB5"]
    for i, curve in enumerate(curves):
        ax.step(
            curve["elapsed"],
            curve["running_best_gain"],
            where="post",
            linewidth=1.2,
            alpha=0.28,
            color=palette[i],
        )

    ax.fill_between(grid, lo, hi, color="#2B8CBE", alpha=0.10, label="unique-curve range")
    ax.fill_between(grid, q25, q75, color="#2B8CBE", alpha=0.22, label="IQR")
    ax.plot(grid, median, color="#174A6A", linewidth=3.1, label="median best K=1 gain")
    ax.axhline(0, color="#4C566A", linestyle="--", linewidth=1.5, alpha=0.72)

    final_median = median[-1]
    ax.scatter([grid[-1]], [final_median], s=95, color="#174A6A", edgecolor="white", linewidth=1.5, zorder=5)
    annotation = f"median final gain: {final_median:+.2f} pp\n{final_positive}/{len(curves)} unique curves > base"
    if sign_p is not None:
        annotation += f"  (sign p={sign_p:.3f})"
    ax.annotate(
        annotation,
        xy=(grid[-1], final_median),
        xytext=(max_duration * 0.48, final_median + 2.5),
        arrowprops={"arrowstyle": "->", "color": "#174A6A", "linewidth": 1.3},
        color="#174A6A",
        fontsize=10,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
    )

    ax.set_title("K=1 RandOpt best-so-far gain across unique hotpath curves")
    ax.set_xlabel("elapsed sampling wallclock (minutes, interpolated from recorded run timing)")
    ax.set_ylabel("held-out test gain vs base (pp)")
    ax.set_xlim(0, max_duration)
    ax.set_ylim(-1.0, max(hi) + 3.5)
    ax.legend(loc="lower right", fontsize=8.4)
    ax.text(
        0.01,
        0.97,
        "Qwen2.5-1.5B-Instruct | MATH-500 levels 4-5 | 6 deduplicated 64-seed curves | 1x NVIDIA L4",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="#4C566A",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
    )

    save(fig, "readme_k1_wallclock_hero.png")
    plt.close(fig)


def plot_accuracy_summary(plt):
    runs = [
        ("GSM8K\n7B/H100/512", "speedrun-runs/modal_run512/record.json"),
        ("MATH-500 L4-5\n7B/H100/512", "speedrun-runs/modal_math512/record.json"),
        ("MATH-500 low-sigma\n7B/H100/512", "speedrun-runs/modal_mathlow512/record.json"),
    ]
    labels, base, ensemble = [], [], []
    for label, rel in runs:
        rec = load_json(ROOT / rel)
        labels.append(label)
        base.append(pct_acc(rec["base_test_accuracy"]))
        ensemble.append(pct_acc(rec["best_ensemble_accuracy"]))

    x = range(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar([i - width / 2 for i in x], base, width, label="base", color="#4C566A")
    bars = ax.bar([i + width / 2 for i in x], ensemble, width, label="RandOpt ensemble", color="#2B8CBE")

    for i, bar in enumerate(bars):
        gain = ensemble[i] - base[i]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.9,
            f"+{gain:.2f} pp",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#174A6A",
        )
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("held-out task accuracy (%)")
    ax.set_ylim(0, max(ensemble) + 10)
    ax.set_title("Measured 7B RandOpt ensemble gains")
    ax.legend(loc="upper left")
    save(fig, "readme_accuracy_summary.png")
    plt.close(fig)


def plot_kernel_speedup(plt):
    rec = load_json(ROOT / "speedrun-runs/gpu_kernel_validation/record.json")
    rows = rec["throughput_fused_vs_perturb_restore"]
    labels = [f"{r['n'] // 1_000_000}M\nparams" for r in rows]
    old = [r["old_ms"] for r in rows]
    new = [r["new_ms"] for r in rows]
    speedup = [r["speedup"] for r in rows]
    x = range(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar([i - width / 2 for i in x], old, width, label="perturb + restore", color="#8F5252")
    bars = ax.bar([i + width / 2 for i in x], new, width, label="fused reconstruct", color="#2F855A")
    for i, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(old[i], new[i]) + 0.65,
            f"{speedup[i]:.1f}x",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#25543F",
        )

    ax.set_xticks(list(x), labels)
    ax.set_ylabel("weight switch time (ms, lower is better)")
    ax.set_title("Fused dense-Rademacher switching removes materialized noise")
    ax.legend(loc="upper left")
    save(fig, "readme_kernel_speedup.png")
    plt.close(fig)


def plot_k1_transfer(plt):
    runs = [
        ("default", "hotpath-runs/sweep1b_default/record.json"),
        ("train32", "hotpath-runs/sweep1b_train32/record.json"),
        ("train16", "hotpath-runs/sweep1b_train16/record.json"),
        ("lvl5 train", "hotpath-runs/sweep1b_lvl5train/record.json"),
        ("stratified", "hotpath-runs/sweep1b_stratify/record.json"),
        ("rad 1e-4", "hotpath-runs/sweepscheme_sig1e4/record.json"),
        ("rad 1e-3", "hotpath-runs/sweepscheme_sig1e3/record.json"),
        ("gauss 5e-4", "hotpath-runs/sweepscheme_gauss5e4/record.json"),
        ("gauss 1e-3", "hotpath-runs/sweepscheme_gauss1e3/record.json"),
    ]
    labels, oracle_gain, train_gain, rho = [], [], [], []
    for label, rel in runs:
        rec = load_json(ROOT / rel)
        base = rec["base_test_accuracy"]
        labels.append(label)
        oracle_gain.append(rec["best_k1_test"]["test_acc"] - base)
        train_gain.append(rec["k1_selected_by_train"]["test_acc"] - base)
        rho.append(rec["train_test_spearman"])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10.6, 6.6), gridspec_kw={"height_ratios": [1.25, 1]}, sharex=True
    )
    x = range(len(labels))
    width = 0.36
    ax1.axhline(0, color="#5E6773", linewidth=1)
    ax1.bar([i - width / 2 for i in x], oracle_gain, width, label="best K=1 on test (oracle)", color="#2B8CBE")
    ax1.bar([i + width / 2 for i in x], train_gain, width, label="K=1 selected by train reward", color="#D9902F")
    ax1.set_ylabel("test gain vs base (pp)")
    ax1.set_title("K=1 search finds winners, but train reward is a weak selector")
    ax1.legend(loc="upper right", bbox_to_anchor=(1.0, 1.18), ncol=2)

    colors = ["#2F855A" if r > 0 else "#8F5252" for r in rho]
    ax2.axhline(0, color="#5E6773", linewidth=1)
    ax2.bar(list(x), rho, width=0.62, color=colors)
    ax2.set_ylabel("Spearman rho\n(train reward, test acc)")
    ax2.set_xticks(list(x), labels, rotation=25, ha="right")
    ax2.set_ylim(-0.25, 0.25)

    save(fig, "readme_k1_transfer.png")
    plt.close(fig)


def plot_merge_summary(plt):
    rec = load_json(ROOT / "merge-runs/merge7b_l40s_pop48_pairs8_fused_20260601/record.json")
    summary = rec["merge_summary"]
    families = [
        ("test_good_test_good", "good + good"),
        ("train_top_train_top", "train-top + train-top"),
        ("random_random", "random + random"),
        ("test_bad_test_bad", "bad + bad"),
    ]
    ops = [("avg", "avg"), ("normsum", "normsum"), ("sum", "sum")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=False)
    width = 0.22
    colors = {"avg": "#2B8CBE", "normsum": "#2F855A", "sum": "#8F5252"}
    x = range(len(families))
    for j, (op, label) in enumerate(ops):
        vals = [summary[key][op]["mean_delta_vs_base"] for key, _ in families]
        offset = (j - 1) * width
        ax1.bar([i + offset for i in x], vals, width, label=label, color=colors[op])
    ax1.axhline(0, color="#5E6773", linewidth=1)
    ax1.set_xticks(list(x), [label for _, label in families], rotation=20, ha="right")
    ax1.set_ylabel("mean test gain vs base (pp)")
    ax1.set_title("Pairwise merge accuracy by family")
    ax1.legend(loc="upper right")

    parent_vals = []
    base_vals = []
    labels = []
    for key, label in families:
        labels.append(label)
        parent_vals.append(summary[key]["avg"]["frac_beats_best_parent"] * 100)
        base_vals.append(summary[key]["avg"]["frac_beats_base"] * 100)
    ax2.bar([i - width / 2 for i in x], base_vals, width, label="beats base", color="#2B8CBE")
    ax2.bar([i + width / 2 for i in x], parent_vals, width, label="beats best parent", color="#D9902F")
    ax2.set_xticks(list(x), labels, rotation=20, ha="right")
    ax2.set_ylim(0, 108)
    ax2.set_ylabel("fraction of avg merges (%)")
    ax2.set_title("Merges preserve gains more often than they add")
    ax2.legend(loc="upper right")

    save(fig, "readme_merge_summary.png")
    plt.close(fig)


def main() -> None:
    plt = setup_matplotlib()
    plot_k1_wallclock_hero(plt)
    plot_accuracy_summary(plt)
    plot_kernel_speedup(plt)
    plot_k1_transfer(plt)
    plot_merge_summary(plt)


if __name__ == "__main__":
    main()

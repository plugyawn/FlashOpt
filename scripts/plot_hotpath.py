#!/usr/bin/env python3
"""
Plot a RandOpt K=1 hotpath run (hotpath.py output).

Two panels:
  (1) best-K=1-on-TEST so far vs seed index — the held-out optimization curve
      (does random search find a single perturbation that beats base on test?),
      with the realistic "selected by train reward" point marked.
  (2) train_reward vs test_acc scatter — the TRANSFER diagnostic: if selecting
      by train reward picked test winners, points trend up-right (high Spearman).
      If flat/noisy, train reward does NOT predict held-out gain.

Usage:
  python scripts/plot_hotpath.py --run hotpath-runs/<name> --out plot.png
  # or point at the jsonl + record directly
"""
import argparse
import json
import os


def load(run_dir):
    rec = json.load(open(os.path.join(run_dir, "record.json")))
    rows = [json.loads(l) for l in open(os.path.join(run_dir, "seeds.jsonl")) if l.strip()]
    rows.sort(key=lambda r: r["idx"])
    return rec, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="hotpath run dir (has record.json + seeds.jsonl)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rec, rows = load(args.run)
    out = args.out or os.path.join(args.run, "hotpath.png")

    base = rec["base_test_accuracy"]
    base_tr = rec["base_train_reward"]
    idx = [r["idx"] + 1 for r in rows]
    test = [r["test_acc"] for r in rows]
    train = [r["train_reward"] for r in rows]
    # running best on test
    best, run_best = -1e9, []
    for t in test:
        best = max(best, t); run_best.append(best)
    rho = rec.get("train_test_spearman", float("nan"))
    by_train = rec.get("k1_selected_by_train", {})
    oracle = rec.get("best_k1_test", {})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: held-out optimization curve
    ax1.scatter(idx, test, s=12, alpha=0.3, color="#888", label="per-seed test acc")
    ax1.step(idx, run_best, where="post", color="#1f77b4", lw=2.2, label="best-K=1-on-test (oracle)")
    ax1.axhline(base, color="#d62728", ls="--", lw=1.5, label=f"base ({base:.1f}%)")
    if by_train:
        ax1.axhline(by_train["test_acc"], color="#2ca02c", ls=":", lw=1.8,
                    label=f"K=1 by TRAIN reward ({by_train['test_acc']:.1f}%)")
    ax1.set_xlabel("seeds evaluated"); ax1.set_ylabel("test accuracy (%)")
    ax1.set_title(f"K=1 held-out optimization\nbest(oracle)={oracle.get('test_acc',float('nan')):.1f}%  "
                  f"realistic(by-train)={by_train.get('test_acc',float('nan')):.1f}%  base={base:.1f}%")
    ax1.grid(alpha=0.3); ax1.legend(loc="lower right", fontsize=8)

    # Panel 2: transfer scatter
    ax2.scatter(train, test, s=18, alpha=0.5, color="#1f77b4")
    ax2.axhline(base, color="#d62728", ls="--", lw=1, alpha=0.7, label=f"base test ({base:.1f}%)")
    ax2.axvline(base_tr, color="#ff7f0e", ls="--", lw=1, alpha=0.7, label=f"base train ({base_tr:.3f})")
    ax2.set_xlabel("train reward (selection signal)"); ax2.set_ylabel("test accuracy (%)")
    ax2.set_title(f"Transfer: does train reward predict test?\nSpearman rho = {rho:.3f}  "
                  f"({rec['model'].split('/')[-1]}, n={len(rows)})")
    ax2.grid(alpha=0.3); ax2.legend(loc="lower right", fontsize=8)

    fig.suptitle(f"{rec['config_name']}  ·  {rec['dataset']}  ·  noise={rec.get('noise')}  "
                 f"·  sigma={rec.get('sigma_values')}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    print(f"  base={base:.1f}%  best-oracle={oracle.get('test_acc',float('nan')):.1f}%  "
          f"by-train={by_train.get('test_acc',float('nan')):.1f}%  rho={rho:.3f}  "
          f"frac>base={rec.get('frac_seeds_beating_base_test',float('nan')):.2%}")


if __name__ == "__main__":
    main()

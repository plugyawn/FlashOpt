#!/usr/bin/env python3
"""
Build hard MATH-500 train/test slices for the RandOpt research loop, in the jsonl
format data_handlers/math500.py expects ({problem, answer, subject, level}).

MATH-500 levels 4-5 are the hardest subset where a 7B base has real headroom.
This script supports the research-loop knobs:

  --levels 4 5            pool of levels to draw the TEST set from (default 4 5)
  --train-levels 5        draw TRAIN only from these levels (default = --levels)
  --stratify              balance TRAIN and TEST to equal level proportions
                          (removes the train-easier/test-harder confound)
  --n-train / --n-test    sizes; TEST is built FIRST and held fixed so every
                          train variant is scored on the SAME held-out set.
  --seed                  deterministic.

The TEST set is always carved from --levels with a fixed seed, so experiments
that only vary the TRAIN design remain comparable on an identical test slice.
Train and test are always DISJOINT (by problem text).

Examples:
  # default lvl4-5, fixed test, random train
  python scripts/make_math500_hard.py --n-train 64 --n-test 192 --out data/m
  # train ONLY on level 5, same lvl4-5 test
  python scripts/make_math500_hard.py --train-levels 5 --n-train 64 --n-test 192 --out data/m_l5
  # stratified train+test (matched level mix)
  python scripts/make_math500_hard.py --stratify --n-train 64 --n-test 192 --out data/m_strat
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict


def load_math500():
    from datasets import load_dataset
    last = None
    for _ in range(5):
        try:
            return load_dataset("HuggingFaceH4/MATH-500", split="test")
        except Exception as e:
            last = e
    raise RuntimeError(f"failed to load MATH-500: {last}")


def _level_int(lvl):
    if isinstance(lvl, int):
        return lvl
    s = str(lvl).lower().replace("level", "").strip()
    try:
        return int(s)
    except ValueError:
        return 0


def _row(ex):
    return {"problem": ex["problem"], "answer": ex["answer"],
            "subject": ex.get("subject", ex.get("type", "")),
            "level": _level_int(ex.get("level", 0))}


def _by_level(rows):
    d = defaultdict(list)
    for r in rows:
        d[r["level"]].append(r)
    return d


def _take_stratified(pool_by_level, n, levels, rng):
    """Take ~n rows with equal counts per level (as even as the pool allows)."""
    levels = sorted(levels)
    per = max(1, n // len(levels))
    out = []
    for lv in levels:
        avail = pool_by_level.get(lv, [])
        rng.shuffle(avail)
        out += avail[:per]
    # top up to n from whatever remains, then trim
    remaining = [r for lv in levels for r in pool_by_level.get(lv, []) if r not in out]
    rng.shuffle(remaining)
    out += remaining[: max(0, n - len(out))]
    rng.shuffle(out)
    return out[:n]


def _dist(rows):
    c = Counter(r["level"] for r in rows)
    n = len(rows) or 1
    mean = sum(r["level"] for r in rows) / n
    return {lv: f"{c[lv]}({c[lv]/n:.0%})" for lv in sorted(c)}, round(mean, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[4, 5],
                    help="levels the TEST set is drawn from")
    ap.add_argument("--train-levels", type=int, nargs="+", default=None,
                    help="levels the TRAIN set is drawn from (default = --levels)")
    ap.add_argument("--stratify", action="store_true",
                    help="balance train AND test to equal per-level proportions")
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--n-test", type=int, default=192)
    ap.add_argument("--out", default="data/math-500-hard")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    test_levels = sorted(set(args.levels))
    train_levels = sorted(set(args.train_levels)) if args.train_levels else test_levels

    ds = load_math500()
    all_rows = [_row(ex) for ex in ds]
    pool = [r for r in all_rows if r["level"] in set(test_levels) | set(train_levels)]

    # ---- TEST first, fixed seed, from test_levels, so it's identical across runs ----
    test_pool = [r for r in pool if r["level"] in test_levels]
    rng_test = random.Random(args.seed)  # fixed -> same test set regardless of train knobs
    if args.stratify:
        test = _take_stratified(_by_level(test_pool), args.n_test, test_levels, rng_test)
    else:
        rng_test.shuffle(test_pool)
        test = test_pool[: args.n_test]
    test_ids = {r["problem"] for r in test}

    # ---- TRAIN from train_levels, disjoint from test ----
    train_pool = [r for r in pool if r["level"] in train_levels and r["problem"] not in test_ids]
    rng_train = random.Random(args.seed + 1)
    if args.stratify:
        train = _take_stratified(_by_level(train_pool), args.n_train, train_levels, rng_train)
    else:
        rng_train.shuffle(train_pool)
        train = train_pool[: args.n_train]

    if len(test) < args.n_test:
        print(f"  WARNING: only {len(test)} test rows (wanted {args.n_test}) at levels {test_levels}")
    if len(train) < args.n_train:
        print(f"  WARNING: only {len(train)} train rows (wanted {args.n_train}) at levels {train_levels}")

    os.makedirs(args.out, exist_ok=True)
    for name, split in [("train.jsonl", train), ("test.jsonl", test)]:
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
            for r in split:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # manifest for provenance
    td, tm = _dist(train); ed, em = _dist(test)
    overlap = len({r["problem"] for r in train} & test_ids)
    manifest = {"levels_test": test_levels, "levels_train": train_levels, "stratify": args.stratify,
                "seed": args.seed, "n_train": len(train), "n_test": len(test),
                "train_level_dist": td, "train_mean_level": tm,
                "test_level_dist": ed, "test_mean_level": em, "overlap": overlap}
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"TRAIN n={len(train)} levels={train_levels} dist={td} mean={tm}")
    print(f"TEST  n={len(test)} levels={test_levels} dist={ed} mean={em}")
    print(f"overlap={overlap} (must be 0) | stratify={args.stratify} -> {args.out}/")


if __name__ == "__main__":
    main()

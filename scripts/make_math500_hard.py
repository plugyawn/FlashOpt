#!/usr/bin/env python3
"""
Build a hard MATH-500 slice (levels 4-5 only) for the RandOpt speedrun, in the
jsonl format data_handlers/math500.py expects ({problem, answer, subject, level}).

MATH-500 is 500 problems; filtering to levels 4-5 leaves the hardest subset where
a 7B base has real headroom (~40-55%) rather than the ~70% all-levels ceiling.
Writes DISJOINT train/test splits (selection train must not overlap eval test).

    python scripts/make_math500_hard.py --levels 4 5 --n-train 128 --n-test 256 --out data/math-500-hard
"""
import argparse
import json
import os


def load_math500():
    """Load MATH-500 via `datasets`. Tries the canonical HF repo."""
    from datasets import load_dataset
    last = None
    for repo, cfg in [("HuggingFaceH4/MATH-500", None)]:
        for _ in range(5):
            try:
                return load_dataset(repo, cfg, split="test") if cfg else load_dataset(repo, split="test")
            except Exception as e:  # transient HF errors
                last = e
    raise RuntimeError(f"failed to load MATH-500: {last}")


def _level_int(lvl):
    # MATH-500 'level' is sometimes "Level 5" or an int.
    if isinstance(lvl, int):
        return lvl
    s = str(lvl).lower().replace("level", "").strip()
    try:
        return int(s)
    except ValueError:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[4, 5])
    ap.add_argument("--n-train", type=int, default=128)
    ap.add_argument("--n-test", type=int, default=256)
    ap.add_argument("--out", default="data/math-500-hard")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = load_math500()
    levels = set(args.levels)
    rows = []
    for ex in ds:
        if _level_int(ex.get("level", 0)) in levels:
            rows.append({
                "problem": ex["problem"],
                "answer": ex["answer"],
                "subject": ex.get("subject", ex.get("type", "")),
                "level": _level_int(ex.get("level", 0)),
            })

    # Deterministic shuffle, then disjoint train/test.
    import random
    random.Random(args.seed).shuffle(rows)
    n_train = min(args.n_train, len(rows))
    train = rows[:n_train]
    test = rows[n_train:n_train + args.n_test]
    if len(test) < args.n_test:
        print(f"  WARNING: only {len(test)} test rows available at levels {sorted(levels)} "
              f"after taking {n_train} train (total {len(rows)}).")

    os.makedirs(args.out, exist_ok=True)
    for name, split in [("train.jsonl", train), ("test.jsonl", test)]:
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
            for r in split:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"levels {sorted(levels)}: {len(rows)} total -> {len(train)} train / {len(test)} test "
          f"(disjoint) in {args.out}/")


if __name__ == "__main__":
    main()

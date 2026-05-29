#!/usr/bin/env python3
"""
Make a tiny GSM8K parquet in the format data_handlers/gsm8k.py expects
(``prompt`` = list of chat messages, ``reward_model`` = {"ground_truth": ...}),
for the speedrun smoke test. Downloads ``openai/gsm8k`` via `datasets`.

    python scripts/make_gsm8k_smoke.py --n-train 64 --n-test 64 --out data/gsm8k
"""
import argparse
import os

INSTRUCTION = ("Solve the math problem. Reason step by step, then give the final "
               "answer on its own line as: #### <number>")


def _load_split(split, retries=5):
    """Load a GSM8K split, retrying through transient HF Hub errors (e.g. 504)."""
    import time as _t
    from datasets import load_dataset
    last = None
    for attempt in range(retries):
        try:
            return load_dataset("openai/gsm8k", "main", split=split)
        except Exception as e:  # HfHubHTTPError, ConnectionError, etc.
            last = e
            wait = 3 * (attempt + 1)
            print(f"  [make_gsm8k] {split} load failed ({type(e).__name__}); "
                  f"retry {attempt+1}/{retries} in {wait}s")
            _t.sleep(wait)
    raise RuntimeError(f"failed to load gsm8k {split} after {retries} retries: {last}")


def build(split, n):
    import pandas as pd  # noqa
    ds = _load_split(split)
    n = min(n, len(ds))
    rows = []
    for ex in ds.select(range(n)):
        gt = ex["answer"].split("####")[-1].strip().replace(",", "")
        rows.append({
            "prompt": [{"role": "user", "content": f"{ex['question']}\n\n{INSTRUCTION}"}],
            "reward_model": {"ground_truth": gt, "style": "rule"},
        })
    import pandas as pd
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--n-test", type=int, default=64)
    ap.add_argument("--out", default="data/gsm8k")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tr = build("train", args.n_train)
    te = build("test", args.n_test)
    tr.to_parquet(os.path.join(args.out, "train.parquet"))
    te.to_parquet(os.path.join(args.out, "test.parquet"))
    print(f"wrote {len(tr)} train / {len(te)} test rows to {args.out}/")


if __name__ == "__main__":
    main()

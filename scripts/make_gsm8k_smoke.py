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


def build(split, n):
    import pandas as pd  # noqa
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
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

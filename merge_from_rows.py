#!/usr/bin/env python3
"""
Pairwise RandOpt merge check from a precomputed seed snapshot.

Use this when a population run was intentionally stopped after enough seeds:
the seed identities are deterministic, and the logged train/test scores are
enough to define good-seed and control families without rerunning the population.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from merge_run import (
    RUNS_DIR,
    _coeffs_for_op,
    _fmt_prompt,
    _row_brief,
    _sample_same_pairs,
    _summarize_merges,
)
from speedrun import detect_hardware, git_commit, load_config, _single_model_accuracy


def _load_rows(path: str) -> List[Dict]:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if len(rows) < 2:
        raise ValueError(f"need at least two seed rows, got {len(rows)} from {path}")
    return rows


def main_from_rows(cfg: Dict[str, Any], seed_rows: List[Dict], run_dir: str,
                   logged_base_test_acc: float | None = None):
    import ray
    import torch  # noqa
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    import randopt
    from core import cleanup_engines, launch_engines
    from data_handlers import get_dataset_handler

    os.makedirs(run_dir, exist_ok=True)
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()

    args = SimpleNamespace(
        dataset=cfg["dataset"], model_name=cfg["model"],
        train_data_path=cfg.get("train_data_path"), test_data_path=cfg.get("test_data_path"),
        train_samples=cfg.get("train_samples", 64), test_samples=cfg.get("test_samples", 96),
        num_engines=int(cfg.get("num_engines", 1)),
    )
    handler = get_dataset_handler(args.dataset)
    max_tokens = cfg.get("max_tokens") or handler.default_max_tokens
    runtime_env = {"env_vars": {"PYTHONPATH": REPO + os.pathsep + os.environ.get("PYTHONPATH", "")}}
    if os.environ.get("RAY_ADDRESS"):
        ray.init(address="auto", ignore_reinit_error=True, runtime_env=runtime_env)
    else:
        ray.init(address="local", ignore_reinit_error=True, runtime_env=runtime_env)

    train_datas, test_datas = randopt.load_data(handler, args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    test_prompts = [_fmt_prompt(tokenizer, args.model_name, d["messages"]) for d in test_datas]
    train_prompts = [_fmt_prompt(tokenizer, args.model_name, d["messages"]) for d in train_datas]
    sp = SamplingParams(temperature=0.0, seed=int(cfg.get("global_seed", 42)), max_tokens=max_tokens)

    t = time.perf_counter()
    engines, pgs = launch_engines(
        args.num_engines, args.model_name, precision=cfg.get("precision", "bfloat16"),
        tensor_parallel_size=cfg.get("tensor_parallel_size", 1),
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.55),
        enforce_eager=cfg.get("enforce_eager", True),
        noise=cfg.get("noise", "rademacher"), kernel=cfg.get("kernel", "auto"),
        max_num_seqs=cfg.get("max_num_seqs"), max_model_len=cfg.get("max_model_len"),
    )
    timings["launch"] = time.perf_counter() - t

    try:
        t = time.perf_counter()
        ray.get(engines[0].collective_rpc.remote("reset_to_base_weights", args=()))
        base_train_out = ray.get(engines[0].generate.remote(train_prompts, sp, use_tqdm=False))
        base_test_out = ray.get(engines[0].generate.remote(test_prompts, sp, use_tqdm=False))
        base_train_reward = handler.postprocess_outputs(base_train_out, train_datas)
        base_test_correct = _single_model_accuracy(handler, base_test_out, test_datas)
        base_test_acc = 100.0 * base_test_correct / len(test_datas)
        timings["base_eval"] = time.perf_counter() - t
        print(f"\n>>> BASE(re-eval) train_reward={base_train_reward:.4f} "
              f"test_acc={base_test_acc:.2f}% ({base_test_correct}/{len(test_datas)})", flush=True)

        base_for_seed_families = logged_base_test_acc if logged_base_test_acc is not None else base_test_acc
        merge_cfg = cfg.get("merge", {}) or {}
        top_n = int(merge_cfg.get("top_n", 12))
        pairs_per_family = int(merge_cfg.get("pairs_per_family", 8))
        ops = list(merge_cfg.get("ops", ["avg", "normsum", "sum"]))
        py_rng = random.Random(int(cfg.get("global_seed", 42)) + 1729)

        sorted_by_train = sorted(seed_rows, key=lambda r: (r["train_reward"], r["test_acc"]), reverse=True)
        train_top = sorted_by_train[:top_n]
        test_good = [r for r in sorted(seed_rows, key=lambda r: r["test_acc"], reverse=True)
                     if r["test_acc"] > base_for_seed_families][:top_n]
        if len(test_good) < 2:
            test_good = sorted(seed_rows, key=lambda r: r["test_acc"], reverse=True)[:top_n]
        test_bad = [r for r in sorted(seed_rows, key=lambda r: r["test_acc"])
                    if r["test_acc"] <= base_for_seed_families][:top_n]
        if len(test_bad) < 2:
            test_bad = sorted(seed_rows, key=lambda r: r["test_acc"])[:top_n]

        families = [
            ("train_top_train_top", train_top),
            ("test_good_test_good", test_good),
            ("random_random", seed_rows),
            ("test_bad_test_bad", test_bad),
        ]
        family_pairs = {
            name: _sample_same_pairs(items, pairs_per_family, py_rng)
            for name, items in families if len(items) >= 2
        }

        merge_rows: List[Dict] = []
        merges_path = os.path.join(run_dir, "merges.jsonl")
        t = time.perf_counter()
        with open(merges_path, "w") as mf:
            total = sum(len(pairs) for pairs in family_pairs.values()) * len(ops)
            done = 0
            for family, pairs in family_pairs.items():
                for left, right in pairs:
                    seed_sigmas = [(int(left["seed"]), float(left["sigma"])),
                                   (int(right["seed"]), float(right["sigma"]))]
                    for op in ops:
                        coeffs = _coeffs_for_op(op)
                        ray.get(engines[0].collective_rpc.remote(
                            "apply_linear_combined_perturbations", args=(seed_sigmas, coeffs)))
                        outs = ray.get(engines[0].generate.remote(test_prompts, sp, use_tqdm=False))
                        correct = _single_model_accuracy(handler, outs, test_datas)
                        acc = 100.0 * correct / len(test_datas)
                        parent_avg = 0.5 * (left["test_acc"] + right["test_acc"])
                        parent_best = max(left["test_acc"], right["test_acc"])
                        row = {
                            "family": family, "op": op, "coeffs": coeffs,
                            "left": _row_brief(left), "right": _row_brief(right),
                            "test_acc": acc, "test_correct": correct, "n_test": len(test_datas),
                            "delta_vs_base": acc - base_test_acc,
                            "parent_avg_test_acc": parent_avg,
                            "parent_best_test_acc": parent_best,
                            "delta_vs_parent_avg": acc - parent_avg,
                            "delta_vs_best_parent": acc - parent_best,
                        }
                        merge_rows.append(row)
                        mf.write(json.dumps(row) + "\n")
                        mf.flush()
                        done += 1
                        if done % 8 == 0 or done == total:
                            print(f"  merge {done}/{total} | {family}/{op} "
                                  f"test={acc:.1f}% base={base_test_acc:.1f}%", flush=True)
        ray.get(engines[0].collective_rpc.remote("reset_to_base_weights", args=()))
        timings["merge_eval"] = time.perf_counter() - t

        timings["total"] = time.perf_counter() - t0
        merge_summary = _summarize_merges(merge_rows, base_test_acc)
        best_seed = max(seed_rows, key=lambda r: r["test_acc"])
        by_train = max(seed_rows, key=lambda r: (r["train_reward"], r["test_acc"]))
        record = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "config_name": cfg.get("name", "?"),
            "config_path": cfg.get("_path"),
            "config_sha256": cfg.get("_sha256"),
            "hardware": detect_hardware(),
            "model": cfg["model"],
            "dataset": args.dataset,
            "noise": cfg.get("noise", "rademacher"),
            "sigma_values": cfg.get("sigma_values"),
            "population_size": len(seed_rows),
            "population_source": "precomputed_seed_rows",
            "logged_base_test_accuracy": logged_base_test_acc,
            "base_train_reward": base_train_reward,
            "base_test_accuracy": base_test_acc,
            "best_k1_test": {
                "test_acc": best_seed["test_acc"], "seed": best_seed["seed"],
                "sigma": best_seed["sigma"], "train_reward": best_seed["train_reward"],
                "delta_vs_logged_base": (
                    best_seed["test_acc"] - logged_base_test_acc
                    if logged_base_test_acc is not None else None
                ),
            },
            "k1_selected_by_train": {
                "test_acc": by_train["test_acc"], "seed": by_train["seed"],
                "sigma": by_train["sigma"], "train_reward": by_train["train_reward"],
            },
            "merge_config": {"top_n": top_n, "pairs_per_family": pairs_per_family, "ops": ops},
            "merge_summary": merge_summary,
            "timings_s": timings,
            "seed_rows_jsonl": "seeds.jsonl",
            "merges_jsonl": "merges.jsonl",
        }
        with open(os.path.join(run_dir, "record.json"), "w") as f:
            json.dump(record, f, indent=2)
        with open(os.path.join(run_dir, "seeds.jsonl"), "w") as f:
            for row in seed_rows:
                f.write(json.dumps(row) + "\n")

        print(f"\n=== MERGE-FROM-ROWS DONE ({run_dir}) ===")
        print(json.dumps(merge_summary, indent=2, default=float))
        return record, merge_rows
    finally:
        cleanup_engines(engines, pgs)


def parse_args():
    p = argparse.ArgumentParser(description="RandOpt pairwise merge check from precomputed rows")
    p.add_argument("--config", required=True)
    p.add_argument("--seed-rows-jsonl", required=True)
    p.add_argument("--run-name", default=None)
    p.add_argument("--logged-base-test-acc", type=float, default=None)
    p.add_argument("--merge-pairs", type=int, default=None)
    p.add_argument("--merge-top-n", type=int, default=None)
    p.add_argument("--merge-ops", default=None)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    cfg = load_config(a.config)
    cfg.setdefault("merge", {})
    if a.merge_pairs is not None:
        cfg["merge"]["pairs_per_family"] = a.merge_pairs
    if a.merge_top_n is not None:
        cfg["merge"]["top_n"] = a.merge_top_n
    if a.merge_ops is not None:
        cfg["merge"]["ops"] = [s.strip() for s in a.merge_ops.split(",") if s.strip()]
    rows = _load_rows(a.seed_rows_jsonl)
    name = a.run_name or f"{cfg['name']}_from_rows_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    main_from_rows(cfg, rows, os.path.join(RUNS_DIR, name), a.logged_base_test_acc)

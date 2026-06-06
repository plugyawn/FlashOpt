#!/usr/bin/env python3
"""
RandOpt 128-population reproduction plus pairwise seed-merge check.

This runner keeps the K=1 hotpath artifact (per-seed train reward and held-out
test accuracy), then asks whether weight-space combinations of two individually
good seeds are enriched for held-out-good models relative to random pairs.

The tested merge operations are:
  avg     : W0 + 0.5*d1 + 0.5*d2
  normsum : W0 + (d1 + d2) / sqrt(2), preserving RMS perturbation scale
  sum     : W0 + d1 + d2

Run locally on a suitable GPU:
  python merge_run.py --config configs/merge_7b_l40s.yaml
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import statistics
import sys
import time
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from hotpath import spearman
from speedrun import load_config, git_commit, detect_hardware, _single_model_accuracy

RUNS_DIR = os.path.join(REPO, "merge-runs")


def _fmt_prompt(tokenizer, model_name: str, messages):
    is_instruct = any(x in model_name.lower() for x in ["instruct", "chat", "it"])
    if is_instruct and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return "\n".join(m["content"] for m in messages) + "\n"


def _coeffs_for_op(op: str) -> List[float]:
    if op == "avg":
        return [0.5, 0.5]
    if op == "normsum":
        c = 1.0 / math.sqrt(2.0)
        return [c, c]
    if op == "sum":
        return [1.0, 1.0]
    raise ValueError(f"unknown merge op: {op}")


def _sample_same_pairs(rows: List[Dict], max_pairs: int, rng: random.Random):
    pairs = [(rows[i], rows[j]) for i in range(len(rows)) for j in range(i + 1, len(rows))]
    rng.shuffle(pairs)
    return pairs[:max_pairs]


def _pct(values: List[bool]) -> float:
    return sum(1 for v in values if v) / len(values) if values else float("nan")


def _mean(values: List[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _median(values: List[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def _summarize_merges(rows: List[Dict], base_test_acc: float) -> Dict[str, Dict[str, Dict]]:
    grouped: Dict[Tuple[str, str], List[Dict]] = {}
    for row in rows:
        grouped.setdefault((row["family"], row["op"]), []).append(row)

    summary: Dict[str, Dict[str, Dict]] = {}
    for (family, op), group in sorted(grouped.items()):
        vals = [r["test_acc"] for r in group]
        out = {
            "n": len(group),
            "mean_test_acc": _mean(vals),
            "median_test_acc": _median(vals),
            "best_test_acc": max(vals) if vals else float("nan"),
            "mean_delta_vs_base": _mean([v - base_test_acc for v in vals]),
            "frac_beats_base": _pct([r["test_acc"] > base_test_acc for r in group]),
            "frac_beats_parent_avg": _pct([r["test_acc"] > r["parent_avg_test_acc"] for r in group]),
            "frac_beats_best_parent": _pct([r["test_acc"] > r["parent_best_test_acc"] for r in group]),
        }
        summary.setdefault(family, {})[op] = out

    random_by_op = summary.get("random_random", {})
    for family, ops in summary.items():
        for op, out in ops.items():
            rand = random_by_op.get(op, {}).get("frac_beats_base")
            out["enrichment_vs_random_frac_beats_base"] = (
                out["frac_beats_base"] / rand if rand and rand == rand else None
            )
    return summary


def _row_brief(row: Dict) -> Dict:
    return {
        "idx": row["idx"],
        "seed": row["seed"],
        "sigma": row["sigma"],
        "train_reward": row["train_reward"],
        "test_acc": row["test_acc"],
    }


def main(cfg: Dict[str, Any], run_dir: str):
    import ray
    import torch  # noqa
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    import randopt
    from core import launch_engines, cleanup_engines
    from data_handlers import get_dataset_handler

    os.makedirs(run_dir, exist_ok=True)
    t0 = time.perf_counter()
    timings: Dict[str, float] = {}

    population = int(cfg["population_size"])
    sigma_list = [float(s) for s in cfg["sigma_values"]]
    global_seed = int(cfg.get("global_seed", 42))
    noise = cfg.get("noise", "rademacher")
    if noise != "rademacher":
        raise ValueError("merge_run currently supports rademacher noise only")

    args = SimpleNamespace(
        dataset=cfg["dataset"], model_name=cfg["model"],
        train_data_path=cfg.get("train_data_path"), test_data_path=cfg.get("test_data_path"),
        train_samples=cfg.get("train_samples", 64), test_samples=cfg.get("test_samples", 96),
        num_engines=int(cfg.get("num_engines", 1)),
        top_k_list=sorted({max(1, int(r * population)) for r in cfg.get("top_k_ratios", [0.04, 0.10])}, reverse=True),
    )
    args.max_top_k = args.top_k_list[0]

    handler = get_dataset_handler(args.dataset)
    max_tokens = cfg.get("max_tokens") or handler.default_max_tokens
    runtime_env = {"env_vars": {"PYTHONPATH": REPO + os.pathsep + os.environ.get("PYTHONPATH", "")}}
    if os.environ.get("RAY_ADDRESS"):
        ray.init(address="auto", ignore_reinit_error=True, runtime_env=runtime_env)
    else:
        ray.init(address="local", ignore_reinit_error=True, runtime_env=runtime_env)

    train_datas, test_datas = randopt.load_data(handler, args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_prompts = [_fmt_prompt(tokenizer, args.model_name, d["messages"]) for d in train_datas]
    test_prompts = [_fmt_prompt(tokenizer, args.model_name, d["messages"]) for d in test_datas]
    sp = SamplingParams(temperature=0.0, seed=global_seed, max_tokens=max_tokens)

    t = time.perf_counter()
    engines, pgs = launch_engines(
        args.num_engines, args.model_name, precision=cfg.get("precision", "bfloat16"),
        tensor_parallel_size=cfg.get("tensor_parallel_size", 1),
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.55),
        enforce_eager=cfg.get("enforce_eager", True),
        noise=noise, kernel=cfg.get("kernel", "auto"),
        max_num_seqs=cfg.get("max_num_seqs"), max_model_len=cfg.get("max_model_len"),
    )
    timings["launch"] = time.perf_counter() - t

    seeds_path = os.path.join(run_dir, "seeds.jsonl")
    merges_path = os.path.join(run_dir, "merges.jsonl")

    try:
        t = time.perf_counter()
        ray.get(engines[0].collective_rpc.remote("reset_to_base_weights", args=()))
        base_train_out = ray.get(engines[0].generate.remote(train_prompts, sp, use_tqdm=False))
        base_test_out = ray.get(engines[0].generate.remote(test_prompts, sp, use_tqdm=False))
        base_train_reward = handler.postprocess_outputs(base_train_out, train_datas)
        base_test_correct = _single_model_accuracy(handler, base_test_out, test_datas)
        base_test_acc = 100.0 * base_test_correct / len(test_datas)
        timings["base_eval"] = time.perf_counter() - t
        print(f"\n>>> BASE train_reward={base_train_reward:.4f} "
              f"test_acc={base_test_acc:.2f}% ({base_test_correct}/{len(test_datas)})", flush=True)

        rng_np = np.random.default_rng(seed=global_seed)
        all_seeds = rng_np.choice(2**31, size=population, replace=False).tolist()
        all_sigmas = rng_np.choice(sigma_list, size=population).tolist()

        t = time.perf_counter()
        rows: List[Dict] = []
        best_test = {"test_acc": -1.0}
        best_train = {"train_reward": -1.0}
        with open(seeds_path, "w") as sf:
            i = 0
            while i < population:
                batch = [(all_seeds[i + j], all_sigmas[i + j])
                         for j in range(min(args.num_engines, population - i))]
                ray.get([engines[k].collective_rpc.remote("perturb_self_weights", args=(int(s), sg, False))
                         for k, (s, sg) in enumerate(batch)])
                train_outs = ray.get([engines[k].generate.remote(train_prompts, sp, use_tqdm=False)
                                      for k in range(len(batch))])
                test_outs = ray.get([engines[k].generate.remote(test_prompts, sp, use_tqdm=False)
                                     for k in range(len(batch))])
                for k, (seed, sigma) in enumerate(batch):
                    tr = handler.postprocess_outputs(train_outs[k], train_datas)
                    tc = _single_model_accuracy(handler, test_outs[k], test_datas)
                    ta = 100.0 * tc / len(test_datas)
                    row = {"idx": i + k, "seed": int(seed), "sigma": float(sigma),
                           "train_reward": float(tr), "test_acc": ta, "test_correct": tc,
                           "n_test": len(test_datas)}
                    rows.append(row)
                    sf.write(json.dumps(row) + "\n")
                    sf.flush()
                    if ta > best_test["test_acc"]:
                        best_test = dict(row)
                    if tr > best_train["train_reward"]:
                        best_train = dict(row)
                i += len(batch)
                print(f"  seed {i}/{population} | last train={rows[-1]['train_reward']:.3f} "
                      f"test={rows[-1]['test_acc']:.1f}% | best-test={best_test['test_acc']:.1f}% "
                      f"[base {base_test_acc:.1f}%]", flush=True)
        timings["population_eval"] = time.perf_counter() - t

        trs = [r["train_reward"] for r in rows]
        tas = [r["test_acc"] for r in rows]
        by_train = max(rows, key=lambda r: (r["train_reward"], r["test_acc"]))
        rho = spearman(trs, tas)

        t = time.perf_counter()
        sorted_by_train = sorted(rows, key=lambda r: (r["train_reward"], r["test_acc"]), reverse=True)
        top_k_perturbs = [(r["seed"], r["sigma"]) for r in sorted_by_train[:args.max_top_k]]
        ensemble_results = randopt.run_ensemble_evaluation(
            args, engines, handler, test_prompts, test_datas, top_k_perturbs, sp,
            base_test=base_test_acc / 100.0)
        timings["ensemble_eval"] = time.perf_counter() - t

        merge_cfg = cfg.get("merge", {}) or {}
        top_n = int(merge_cfg.get("top_n", 16))
        pairs_per_family = int(merge_cfg.get("pairs_per_family", 24))
        ops = list(merge_cfg.get("ops", ["avg", "normsum", "sum"]))
        py_rng = random.Random(global_seed + 1729)

        train_top = sorted_by_train[:top_n]
        test_good = [r for r in sorted(rows, key=lambda r: r["test_acc"], reverse=True)
                     if r["test_acc"] > base_test_acc][:top_n]
        if len(test_good) < 2:
            test_good = sorted(rows, key=lambda r: r["test_acc"], reverse=True)[:top_n]
        test_bad = [r for r in sorted(rows, key=lambda r: r["test_acc"])
                    if r["test_acc"] <= base_test_acc][:top_n]
        if len(test_bad) < 2:
            test_bad = sorted(rows, key=lambda r: r["test_acc"])[:top_n]

        families = [
            ("train_top_train_top", train_top),
            ("test_good_test_good", test_good),
            ("random_random", rows),
            ("test_bad_test_bad", test_bad),
        ]
        family_pairs = {
            name: _sample_same_pairs(items, pairs_per_family, py_rng)
            for name, items in families if len(items) >= 2
        }

        t = time.perf_counter()
        merge_rows: List[Dict] = []
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
                        if done % 12 == 0 or done == total:
                            print(f"  merge {done}/{total} | {family}/{op} "
                                  f"test={acc:.1f}% base={base_test_acc:.1f}%", flush=True)
        ray.get(engines[0].collective_rpc.remote("reset_to_base_weights", args=()))
        timings["merge_eval"] = time.perf_counter() - t

        merge_summary = _summarize_merges(merge_rows, base_test_acc)
        timings["total"] = time.perf_counter() - t0
        record = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "config_name": cfg.get("name", "?"),
            "config_path": cfg.get("_path"),
            "config_sha256": cfg.get("_sha256"),
            "hardware": detect_hardware(),
            "model": cfg["model"],
            "dataset": args.dataset,
            "noise": noise,
            "sigma_values": sigma_list,
            "population_size": population,
            "train_samples": len(train_datas),
            "test_samples": len(test_datas),
            "base_train_reward": base_train_reward,
            "base_test_accuracy": base_test_acc,
            "best_k1_test": {"test_acc": best_test["test_acc"], "seed": best_test["seed"],
                             "sigma": best_test["sigma"], "train_reward": best_test["train_reward"],
                             "delta_vs_base": best_test["test_acc"] - base_test_acc},
            "k1_selected_by_train": {"test_acc": by_train["test_acc"], "seed": by_train["seed"],
                                     "sigma": by_train["sigma"], "train_reward": by_train["train_reward"],
                                     "delta_vs_base": by_train["test_acc"] - base_test_acc},
            "train_test_spearman": rho,
            "frac_seeds_beating_base_test": sum(t > base_test_acc for t in tas) / len(tas),
            "ensemble_results": ensemble_results,
            "merge_config": {"top_n": top_n, "pairs_per_family": pairs_per_family, "ops": ops},
            "merge_summary": merge_summary,
            "timings_s": timings,
            "seeds_jsonl": "seeds.jsonl",
            "merges_jsonl": "merges.jsonl",
        }
        with open(os.path.join(run_dir, "record.json"), "w") as f:
            json.dump(record, f, indent=2)

        print(f"\n=== MERGE RUN DONE ({run_dir}) ===")
        print(f"  base test {base_test_acc:.1f}%")
        print(f"  best K=1 test {best_test['test_acc']:.1f}% "
              f"(+{best_test['test_acc'] - base_test_acc:.1f})")
        print(f"  K=1 by train {by_train['test_acc']:.1f}% "
              f"(+{by_train['test_acc'] - base_test_acc:.1f}); rho={rho:.3f}")
        print(json.dumps(merge_summary, indent=2, default=float))
        return record, rows, merge_rows
    finally:
        cleanup_engines(engines, pgs)


def parse_args():
    p = argparse.ArgumentParser(description="RandOpt pairwise merge check")
    p.add_argument("--config", required=True)
    p.add_argument("--run-name", default=None)
    p.add_argument("--population", type=int, default=None)
    p.add_argument("--sigma", default=None, help="comma-separated, overrides config")
    p.add_argument("--merge-pairs", type=int, default=None)
    p.add_argument("--merge-top-n", type=int, default=None)
    p.add_argument("--merge-ops", default=None, help="comma-separated ops: avg,normsum,sum")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    cfg = load_config(a.config)
    if a.population is not None:
        cfg["population_size"] = a.population
    if a.sigma is not None:
        cfg["sigma_values"] = [float(s) for s in a.sigma.split(",")]
    cfg.setdefault("merge", {})
    if a.merge_pairs is not None:
        cfg["merge"]["pairs_per_family"] = a.merge_pairs
    if a.merge_top_n is not None:
        cfg["merge"]["top_n"] = a.merge_top_n
    if a.merge_ops is not None:
        cfg["merge"]["ops"] = [s.strip() for s in a.merge_ops.split(",") if s.strip()]
    name = a.run_name or f"{cfg['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    main(cfg, os.path.join(RUNS_DIR, name))

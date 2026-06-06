#!/usr/bin/env python3
"""
RandOpt research HOTPATH — single-perturbation (K=1) on held-out test.

Where speedrun.py is the full pipeline (sample -> select top-k -> ENSEMBLE), this
hotpath is the research instrument for the *individual* perturbation question:

  "Does random search find a single seed whose perturbed model beats base on a
   held-out test set, and does selecting by TRAIN reward actually pick it?"

For every seed it evaluates the perturbed model on BOTH the train set (the
selection signal) and the test set (the held-out truth), logs a row per seed, and
tracks the running best-K=1-on-test. The train-vs-test scatter it produces is the
core transfer diagnostic (does train reward predict test accuracy?).

Ensemble is intentionally out of scope here (speedrun.py does that). This is for
fast iteration on a small model: which perturbation scheme / train-set design
makes K=1 transfer best.

  python hotpath.py --config configs/hotpath_1b.yaml
  python hotpath.py --config configs/hotpath_1b.yaml --population 128 --noise gaussian
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

# reuse the speedrun helpers (pure; no GPU import at module load)
from speedrun import (load_config, git_commit, detect_hardware, count_tokens,
                      TokenMeter, _single_model_accuracy, _fineweb_prepare,
                      _bpb_for_current_weights)

RUNS_DIR = os.path.join(REPO, "hotpath-runs")


def spearman(x: List[float], y: List[float]) -> float:
    """Rank correlation (no scipy). Returns nan if degenerate."""
    n = len(x)
    if n < 3:
        return float("nan")
    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


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

    args = SimpleNamespace(
        dataset=cfg["dataset"], model_name=cfg["model"],
        train_data_path=cfg.get("train_data_path"), test_data_path=cfg.get("test_data_path"),
        train_samples=cfg.get("train_samples", 64), test_samples=cfg.get("test_samples", 192),
    )
    handler = get_dataset_handler(args.dataset)
    max_tokens = cfg.get("max_tokens") or handler.default_max_tokens

    _rt = {"env_vars": {"PYTHONPATH": REPO + os.pathsep + os.environ.get("PYTHONPATH", "")}}
    if os.environ.get("RAY_ADDRESS"):
        ray.init(address="auto", ignore_reinit_error=True, runtime_env=_rt)
    else:
        ray.init(address="local", ignore_reinit_error=True, runtime_env=_rt)

    train_datas, test_datas = randopt.load_data(handler, args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    is_instruct = any(x in args.model_name.lower() for x in ["instruct", "chat", "it"])

    def fmt(messages):
        if is_instruct and tokenizer.chat_template:
            return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return "\n".join(m["content"] for m in messages) + "\n"

    train_prompts = [fmt(d["messages"]) for d in train_datas]
    test_prompts = [fmt(d["messages"]) for d in test_datas]
    sp = SamplingParams(temperature=0.0, seed=global_seed, max_tokens=max_tokens)

    n_engines = int(cfg.get("num_engines", 1))
    t = time.perf_counter()
    engines, pgs = launch_engines(
        n_engines, args.model_name, precision=cfg.get("precision", "bfloat16"),
        tensor_parallel_size=cfg.get("tensor_parallel_size", 1),
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.75),
        enforce_eager=cfg.get("enforce_eager", True),
        noise=noise, kernel=cfg.get("kernel", "auto"),
        max_num_seqs=cfg.get("max_num_seqs"), max_model_len=cfg.get("max_model_len"),
    )
    timings["launch"] = time.perf_counter() - t
    meter = TokenMeter()

    def gen(eng, prompts):
        return ray.get(eng.generate.remote(prompts, sp, use_tqdm=False))

    seeds_path = os.path.join(run_dir, "seeds.jsonl")
    fw_chunks = fw_bytes = None
    want_bpb = bool(cfg.get("fineweb", {}).get("on_probe"))  # optional, slow

    try:
        # ---- base eval (unperturbed) on train + test ----
        t = time.perf_counter()
        ray.get(engines[0].collective_rpc.remote("reset_to_base_weights", args=()))
        base_train_out = gen(engines[0], train_prompts)
        base_test_out = gen(engines[0], test_prompts)
        base_train_reward = handler.postprocess_outputs(base_train_out, train_datas)
        base_test_correct = _single_model_accuracy(handler, base_test_out, test_datas)
        base_test_acc = 100.0 * base_test_correct / len(test_datas)
        timings["base_eval"] = time.perf_counter() - t
        meter.add("base_eval", [base_train_out, base_test_out], timings["base_eval"])
        print(f"\n>>> BASE: train_reward={base_train_reward:.4f}  "
              f"test_acc={base_test_acc:.2f}% ({base_test_correct}/{len(test_datas)})", flush=True)

        if want_bpb:
            fw_chunks, fw_bytes, _ = _fineweb_prepare(cfg, tokenizer)

        # ---- pre-generate seeds + sigmas (deterministic) ----
        rng = np.random.default_rng(seed=global_seed)
        all_seeds = rng.choice(2**31, size=population, replace=False).tolist()
        all_sigmas = rng.choice(sigma_list, size=population).tolist()

        rows: List[Dict] = []
        best_test = {"test_acc": -1.0}
        best_train = {"train_reward": -1.0}
        t_sample = time.perf_counter()

        with open(seeds_path, "w") as sf:
            i = 0
            while i < population:
                batch = [(all_seeds[i + j], all_sigmas[i + j])
                         for j in range(min(n_engines, population - i))]
                # perturb each engine to its seed (reconstruct from base)
                ray.get([engines[k].collective_rpc.remote("perturb_self_weights", args=(int(s), sg, False))
                         for k, (s, sg) in enumerate(batch)])
                train_outs = ray.get([engines[k].generate.remote(train_prompts, sp, use_tqdm=False)
                                      for k in range(len(batch))])
                test_outs = ray.get([engines[k].generate.remote(test_prompts, sp, use_tqdm=False)
                                     for k in range(len(batch))])
                meter.add("sampling", train_outs + test_outs, 0.0)  # time added at end

                for k, (seed, sigma) in enumerate(batch):
                    tr = handler.postprocess_outputs(train_outs[k], train_datas)
                    tc = _single_model_accuracy(handler, test_outs[k], test_datas)
                    ta = 100.0 * tc / len(test_datas)
                    row = {"idx": i + k, "seed": int(seed), "sigma": float(sigma),
                           "train_reward": float(tr), "test_acc": ta, "test_correct": tc,
                           "n_test": len(test_datas)}
                    if want_bpb and fw_chunks is not None:
                        # perturbed weights are already live on engine k
                        b = _bpb_for_current_weights(engines[k], fw_chunks, fw_bytes)
                        row["fineweb_bpb"] = b.bpb
                    rows.append(row)
                    sf.write(json.dumps(row) + "\n"); sf.flush()
                    if ta > best_test["test_acc"]:
                        best_test = dict(row)
                    if tr > best_train["train_reward"]:
                        best_train = dict(row)

                i += len(batch)
                bt = best_test["test_acc"]
                print(f"  seed {i}/{population} | last train={rows[-1]['train_reward']:.3f} "
                      f"test={rows[-1]['test_acc']:.1f}% | BEST-K1-test={bt:.1f}% "
                      f"[base {base_test_acc:.1f}%]", flush=True)

        timings["sampling"] = time.perf_counter() - t_sample
        # fold the real sampling time into the meter's last phase
        meter.per_phase["sampling"]["seconds"] = timings["sampling"]
        meter.gen_seconds += timings["sampling"]

        # ---- analysis ----
        trs = [r["train_reward"] for r in rows]
        tas = [r["test_acc"] for r in rows]
        rho = spearman(trs, tas)
        # what would TRAIN-based selection have given on test (K=1 by train)?
        by_train = max(rows, key=lambda r: (r["train_reward"], r["test_acc"]))
        timings["total"] = time.perf_counter() - t0
        thr = meter.summary()

        record = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "config_name": cfg.get("name", "?"),
            "model": cfg["model"], "dataset": args.dataset,
            "hardware": detect_hardware(),
            "noise": noise, "sigma_values": sigma_list,
            "population_size": population,
            "train_samples": len(train_datas), "test_samples": len(test_datas),
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
            "throughput": thr, "timings_s": timings,
            "seeds_jsonl": "seeds.jsonl",
        }
        with open(os.path.join(run_dir, "record.json"), "w") as f:
            json.dump(record, f, indent=2)

        print(f"\n=== HOTPATH DONE ({run_dir}) ===")
        print(f"  base test {base_test_acc:.1f}%")
        print(f"  best K=1 on TEST (oracle): {best_test['test_acc']:.1f}% "
              f"(+{best_test['test_acc']-base_test_acc:.1f}) seed={best_test['seed']} σ={best_test['sigma']}")
        print(f"  K=1 selected by TRAIN reward: {by_train['test_acc']:.1f}% "
              f"(+{by_train['test_acc']-base_test_acc:.1f}) [the realistic number]")
        print(f"  train->test Spearman rho = {rho:.3f}  "
              f"(how well train reward predicts test acc)")
        print(f"  frac seeds beating base on test = {record['frac_seeds_beating_base_test']:.2%}")
        return record
    finally:
        cleanup_engines(engines, pgs)


def parse_args():
    p = argparse.ArgumentParser(description="RandOpt K=1 hotpath")
    p.add_argument("--config", required=True)
    p.add_argument("--run-name", default=None)
    p.add_argument("--population", type=int, default=None)
    p.add_argument("--noise", default=None, choices=["rademacher", "gaussian"])
    p.add_argument("--sigma", default=None, help="comma-separated, overrides config")
    p.add_argument("--train-samples", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    cfg = load_config(a.config)
    if a.population is not None: cfg["population_size"] = a.population
    if a.noise is not None: cfg["noise"] = a.noise
    if a.sigma is not None: cfg["sigma_values"] = [float(s) for s in a.sigma.split(",")]
    if a.train_samples is not None: cfg["train_samples"] = a.train_samples
    name = a.run_name or f"{cfg['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    main(cfg, os.path.join(RUNS_DIR, name))

#!/usr/bin/env python3
"""
RandOpt Speedrun harness — nanoGPT-speedrun-style fixed standard.

Runs the RandOpt pipeline on the **fast** dense-Rademacher runtime, times each
phase, reports throughput (**seeds/sec**) and quality (ensemble task accuracy +
held-out **FineWeb bits-per-byte**), and appends a reproducible record.

    python speedrun.py --config configs/standard_8xh100_qwen72b.yaml
    python speedrun.py --config configs/smoke_1gpu_small.yaml         # tiny smoke

The pure helpers (config / record / hardware) import no GPU deps and are
CPU-tested in tests/test_speedrun.py; ray/vllm/randopt are imported inside
``main`` (GPU only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import yaml

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from eval import fineweb  # pure math/data

RECORDS_MD = os.path.join(REPO, "RECORDS.md")
RUNS_DIR = os.path.join(REPO, "speedrun-runs")


# --------------------------------------------------------------------------- #
# Pure helpers (CPU-testable)
# --------------------------------------------------------------------------- #
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = path
    cfg["_sha256"] = _sha256_file(path)
    return cfg


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def derive_topk(population_size: int, top_k_ratios: List[float]) -> Tuple[List[int], int]:
    """Mirror randopt.parse_args: ratios -> sorted unique top-k sizes (desc)."""
    top_k_list = sorted({max(1, int(r * population_size)) for r in top_k_ratios}, reverse=True)
    return top_k_list, top_k_list[0]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def detect_hardware() -> Dict[str, Any]:
    try:
        import torch
        if torch.cuda.is_available():
            return {"gpu": torch.cuda.get_device_name(0),
                    "gpu_count": torch.cuda.device_count()}
    except Exception:
        pass
    return {"gpu": "cpu", "gpu_count": 0}


def build_record(cfg: Dict[str, Any], *, hardware: Dict, timings: Dict, seeds_per_sec: float,
                 base_train_reward: float, base_test_acc: float, ensemble_results: Dict,
                 best_sigma: float, top_k_perturbs: List[Tuple[int, float]],
                 fineweb_base: Optional[Dict], fineweb_ensemble: Optional[Dict],
                 fineweb_manifest_sha: Optional[str]) -> Dict[str, Any]:
    ens_acc = {str(k): v["accuracy"] for k, v in ensemble_results.items()}
    best_ens_acc = max(ens_acc.values()) if ens_acc else float("nan")
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "config_name": cfg.get("name", "?"),
        "config_sha256": cfg.get("_sha256"),
        "hardware": hardware,
        "model": cfg["model"],
        "precision": cfg.get("precision", "bfloat16"),
        "tensor_parallel_size": cfg.get("tensor_parallel_size", 1),
        "num_engines": cfg.get("num_engines", 1),
        "enforce_eager": cfg.get("enforce_eager", True),
        "noise": cfg.get("noise", "rademacher"),
        "kernel": cfg.get("kernel", "auto"),
        "dataset": cfg["dataset"],
        "train_samples": cfg.get("train_samples"),
        "test_samples": cfg.get("test_samples"),
        "population_size": cfg["population_size"],
        "sigma_values": cfg["sigma_values"],
        "best_sigma": best_sigma,
        "timings_s": timings,
        "seeds_per_sec": seeds_per_sec,
        "base_train_reward": base_train_reward,
        "base_test_accuracy": base_test_acc,
        "ensemble_accuracy": ens_acc,
        "best_ensemble_accuracy": best_ens_acc,
        "fineweb": {
            "base": fineweb_base,
            "ensemble": fineweb_ensemble,
            "manifest_sha256": fineweb_manifest_sha,
        },
        "top_k_perturbs": [[int(s), float(sg)] for s, sg in top_k_perturbs],
    }


def _fmt_pct(x: Optional[float]) -> str:
    return f"{x:.1f}%" if isinstance(x, (int, float)) and x == x else "—"


def _fmt_bpb(d: Optional[Dict]) -> str:
    return f"{d['bpb']:.3f}" if d and d.get("bpb") == d.get("bpb") else "—"


def write_record(record: Dict[str, Any], run_dir: str, records_md: str = RECORDS_MD) -> str:
    os.makedirs(run_dir, exist_ok=True)
    rec_path = os.path.join(run_dir, "record.json")
    with open(rec_path, "w") as f:
        json.dump(record, f, indent=2)

    header = (
        "# RandOpt Speedrun Records\n\n"
        "Each row is one run of the fixed standard. Reproduce with:\n"
        "`python speedrun.py --config configs/standard_8xh100_qwen72b.yaml`\n\n"
        "| date | commit | config | model | hardware | pop | seeds/s | base acc | ens acc | base bpb | ens bpb | total |\n"
        "|------|--------|--------|-------|----------|-----|---------|----------|---------|----------|---------|-------|\n"
    )
    if not os.path.exists(records_md):
        with open(records_md, "w") as f:
            f.write(header)
    hw = record["hardware"]
    hw_str = f"{hw.get('gpu_count', '?')}x{hw.get('gpu', '?')}"
    row = (f"| {record['timestamp'][:10]} | {record['git_commit']} | {record['config_name']} "
           f"| {record['model'].split('/')[-1]} | {hw_str} | {record['population_size']} "
           f"| {record['seeds_per_sec']:.2f} | {_fmt_pct(record['base_test_accuracy']*100 if record['base_test_accuracy'] else None)} "
           f"| {_fmt_pct(record['best_ensemble_accuracy'])} "
           f"| {_fmt_bpb(record['fineweb']['base'])} | {_fmt_bpb(record['fineweb']['ensemble'])} "
           f"| {record['timings_s'].get('total', 0):.0f}s |\n")
    with open(records_md, "a") as f:
        f.write(row)
    return rec_path


# --------------------------------------------------------------------------- #
# FineWeb driving (engine generate callable injected; keeps eval.fineweb pure)
# --------------------------------------------------------------------------- #
def _collect_logprobs(generate_remote, chunks, sampling_params, use_tqdm=False):
    """generate_remote(prompts, sp, use_tqdm) -> list[RequestOutput]. Returns
    per-chunk lists of actual-token logprobs (positions >=1)."""
    prompts = [{"prompt_token_ids": list(c.token_ids)} for c in chunks]
    outputs = generate_remote(prompts, sampling_params, use_tqdm)
    return [fineweb.extract_actual_token_logprobs(o) for o in outputs]


# --------------------------------------------------------------------------- #
# Main (GPU). Heavy imports are local.
# --------------------------------------------------------------------------- #
def main(cfg: Dict[str, Any], run_dir: str):
    import ray
    import torch  # noqa
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    import randopt  # reuse load_data / evaluate_base_model / run_sampling / run_ensemble_evaluation
    from core import launch_engines, cleanup_engines
    from data_handlers import get_dataset_handler

    from types import SimpleNamespace

    t0 = time.perf_counter()
    timings: Dict[str, float] = {}

    # ---- args namespace expected by randopt.* ----
    top_k_list, max_top_k = derive_topk(cfg["population_size"], cfg["top_k_ratios"])
    args = SimpleNamespace(
        dataset=cfg["dataset"],
        model_name=cfg["model"],
        train_data_path=cfg.get("train_data_path"),
        test_data_path=cfg.get("test_data_path"),
        train_samples=cfg.get("train_samples", 200),
        test_samples=cfg.get("test_samples"),
        num_engines=cfg.get("num_engines", 1),
        tp=cfg.get("tensor_parallel_size", 1),
        population_size=cfg["population_size"],
        sigma_list=[float(s) for s in cfg["sigma_values"]],
        top_k_list=top_k_list,
        max_top_k=max_top_k,
        global_seed=cfg.get("global_seed", 42),
        precision=cfg.get("precision", "bfloat16"),
        max_tokens=cfg.get("max_tokens", 1024),
    )

    handler = get_dataset_handler(args.dataset)
    max_tokens = args.max_tokens or handler.default_max_tokens

    # Propagate the repo root to Ray workers so vLLM can import the
    # worker_extension_cls (utils.worker_extn -> core.perturb) on every worker.
    _rt = {"env_vars": {"PYTHONPATH": REPO + os.pathsep + os.environ.get("PYTHONPATH", "")}}
    if os.environ.get("RAY_ADDRESS"):
        ray.init(address="auto", ignore_reinit_error=True, runtime_env=_rt)
    else:
        ray.init(address="local", ignore_reinit_error=True, runtime_env=_rt)

    # ---- data + prompts ----
    train_datas, test_datas = randopt.load_data(handler, args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    is_instruct = any(x in args.model_name.lower() for x in ["instruct", "chat", "it"])

    def format_prompt(messages):
        if is_instruct and tokenizer.chat_template:
            return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return "\n".join(m["content"] for m in messages) + "\n"

    train_prompts = [format_prompt(d["messages"]) for d in train_datas]
    test_prompts = [format_prompt(d["messages"]) for d in test_datas]
    sampling_params = SamplingParams(temperature=0.0, seed=args.global_seed, max_tokens=max_tokens)

    # ---- launch engines (fast runtime) ----
    t = time.perf_counter()
    engines, pgs = launch_engines(
        args.num_engines, args.model_name, precision=args.precision,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.75),
        enforce_eager=cfg.get("enforce_eager", True),
        noise=cfg.get("noise", "rademacher"), kernel=cfg.get("kernel", "auto"),
    )
    timings["launch"] = time.perf_counter() - t

    fineweb_base = fineweb_ensemble = None
    fw_manifest_sha = None
    try:
        # ---- base eval ----
        t = time.perf_counter()
        base_train_reward, base_test_acc = randopt.evaluate_base_model(
            engines, handler, train_prompts, test_prompts, train_datas, test_datas, sampling_params)
        timings["base_eval"] = time.perf_counter() - t

        # ---- sampling (the hot loop; seeds/sec) ----
        t = time.perf_counter()
        perf, best_sigma = randopt.run_sampling(
            args, engines, handler, train_prompts, train_datas, sampling_params)
        timings["sampling"] = time.perf_counter() - t
        seeds_per_sec = args.population_size / max(timings["sampling"], 1e-9)
        print(f"\n>>> THROUGHPUT: {seeds_per_sec:.2f} seeds/sec "
              f"({args.population_size} seeds in {timings['sampling']:.1f}s)")

        # ---- selection ----
        sorted_perturbs = sorted(perf.items(), key=lambda x: x[1], reverse=True)
        top_k_perturbs = [(s, sg) for (s, sg), _ in sorted_perturbs[:max_top_k]]

        # ---- ensemble eval (task acc) ----
        t = time.perf_counter()
        ensemble_results = randopt.run_ensemble_evaluation(
            args, engines, handler, test_prompts, test_datas, top_k_perturbs, sampling_params, base_test_acc)
        timings["ensemble_eval"] = time.perf_counter() - t

        # ---- FineWeb held-out bpb ----
        t = time.perf_counter()
        fineweb_base, fineweb_ensemble, fw_manifest_sha = _run_fineweb(
            cfg, engines, tokenizer, top_k_perturbs)
        timings["fineweb"] = time.perf_counter() - t

        timings["total"] = time.perf_counter() - t0

        hardware = detect_hardware()
        record = build_record(
            cfg, hardware=hardware, timings=timings, seeds_per_sec=seeds_per_sec,
            base_train_reward=base_train_reward, base_test_acc=base_test_acc,
            ensemble_results=ensemble_results, best_sigma=best_sigma, top_k_perturbs=top_k_perturbs,
            fineweb_base=fineweb_base, fineweb_ensemble=fineweb_ensemble,
            fineweb_manifest_sha=fw_manifest_sha)
        rec_path = write_record(record, run_dir)
        print(f"\n=== RECORD written: {rec_path} ===")
        print(json.dumps({k: record[k] for k in
                          ["seeds_per_sec", "base_test_accuracy", "best_ensemble_accuracy", "fineweb", "timings_s"]},
                         indent=2))
    finally:
        cleanup_engines(engines, pgs)


def _run_fineweb(cfg, engines, tokenizer, top_k_perturbs):
    """Base + ensemble FineWeb bpb on engines[0] (sequential; correct for the
    1-engine standard). Returns (base_dict, ensemble_dict, manifest_sha)."""
    from vllm import SamplingParams
    import ray

    fw = cfg.get("fineweb")
    if not fw:
        return None, None, None
    path = fw.get("path", fineweb.DEFAULT_HELDOUT_PATH)
    if not os.path.exists(path):
        if fw.get("build_if_missing"):
            print(f"[fineweb] building held-out slice -> {path}")
            fineweb.build_heldout(path, num_docs=fw.get("build_num_docs", 256))
        else:
            print(f"[fineweb] {path} missing and build_if_missing=false; skipping bpb")
            return None, None, None

    manifest_sha = fineweb.file_sha256(path)
    docs = fineweb.load_heldout(path)
    chunks = fineweb.tokenize_docs_to_chunks(tokenizer, docs, max_len=fw.get("max_len", 2048))
    max_chunks = fw.get("max_chunks")
    if max_chunks:
        chunks = chunks[:max_chunks]
    total_bytes = fineweb.heldout_total_bytes(docs)
    sp = SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0.0, detokenize=False)
    eng = engines[0]

    def gen(prompts, sampling_params, use_tqdm=False):
        return ray.get(eng.generate.remote(prompts, sampling_params, use_tqdm))

    # base
    ray.get(eng.collective_rpc.remote("reset_to_base_weights", args=()))
    base_lps = _collect_logprobs(gen, chunks, sp)
    base_res = fineweb.bpb_from_token_logprobs(base_lps, total_bytes)
    print(f"[fineweb] base bpb={base_res.bpb:.4f} (nats/tok={base_res.nats_per_token:.4f}, "
          f"{base_res.n_scored_tokens} toks / {total_bytes} bytes)")

    ens_res = None
    if fw.get("ensemble") and top_k_perturbs:
        acc = fineweb.EnsembleBpbAccumulator(n_chunks=len(chunks))
        for i, (seed, sigma) in enumerate(top_k_perturbs):
            ray.get(eng.collective_rpc.remote("apply_perturbation", args=(int(seed), float(sigma))))
            acc.add_model(_collect_logprobs(gen, chunks, sp))
        ray.get(eng.collective_rpc.remote("reset_to_base_weights", args=()))
        ens_res = acc.result(total_bytes)
        print(f"[fineweb] ensemble(K={len(top_k_perturbs)}) bpb={ens_res.bpb:.4f} "
              f"(nats/tok={ens_res.nats_per_token:.4f})")

    return base_res.as_dict(), (ens_res.as_dict() if ens_res else None), manifest_sha


def parse_args():
    p = argparse.ArgumentParser(description="RandOpt speedrun")
    p.add_argument("--config", required=True, help="Path to a configs/*.yaml")
    p.add_argument("--run-name", default=None, help="Subdir name under speedrun-runs/")
    p.add_argument("--build-fineweb", action="store_true",
                   help="Only build the held-out FineWeb slice, then exit")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    cfg = load_config(a.config)
    if a.build_fineweb:
        fw = cfg.get("fineweb", {})
        man = fineweb.build_heldout(fw.get("path", fineweb.DEFAULT_HELDOUT_PATH),
                                    num_docs=fw.get("build_num_docs", 256))
        print(json.dumps(man.__dict__, indent=2))
        sys.exit(0)
    run_name = a.run_name or f"{cfg['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(RUNS_DIR, run_name)
    main(cfg, run_dir)

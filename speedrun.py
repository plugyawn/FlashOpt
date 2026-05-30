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
    # Allow callers (e.g. the Modal e2e where git isn't installed) to inject the
    # host commit via env so the record's provenance is preserved.
    env_commit = os.environ.get("RANDOPT_GIT_COMMIT")
    if env_commit:
        return env_commit
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


def count_tokens(outputs) -> Tuple[int, int, int]:
    """Sum (generated, prompt, prompts) tokens from a list of vLLM RequestOutputs.

    generated = decode tokens (the throughput-relevant work); prompt = prefill
    tokens; prompts = number of completions. Robust to missing token-id lists.
    """
    gen = prompt = n = 0
    for o in outputs:
        n += 1
        pti = getattr(o, "prompt_token_ids", None)
        if pti is not None:
            prompt += len(pti)
        for comp in getattr(o, "outputs", []) or []:
            tids = getattr(comp, "token_ids", None)
            gen += len(tids) if tids is not None else 0
    return gen, prompt, n


class TokenMeter:
    """Accumulates generation work across phases for tokens/sec + prompts/sec."""

    def __init__(self):
        self.gen_tokens = 0
        self.prompt_tokens = 0
        self.prompts = 0
        self.gen_seconds = 0.0          # wall-clock spent in counted generate phases
        self.per_phase: Dict[str, Dict[str, float]] = {}

    def add(self, phase: str, outputs_iter, seconds: float):
        g = p = n = 0
        for outputs in outputs_iter:
            gg, pp, nn = count_tokens(outputs)
            g += gg; p += pp; n += nn
        self.gen_tokens += g
        self.prompt_tokens += p
        self.prompts += n
        self.gen_seconds += seconds
        self.per_phase[phase] = {
            "gen_tokens": g, "prompt_tokens": p, "prompts": n, "seconds": seconds,
            "gen_tok_per_sec": g / seconds if seconds else 0.0,
            "prompts_per_sec": n / seconds if seconds else 0.0,
        }

    def summary(self) -> Dict[str, Any]:
        s = max(self.gen_seconds, 1e-9)
        return {
            "gen_tokens": self.gen_tokens,
            "prompt_tokens": self.prompt_tokens,
            "prompts": self.prompts,
            "gen_seconds": self.gen_seconds,
            "gen_tokens_per_sec": self.gen_tokens / s,
            "total_tokens_per_sec": (self.gen_tokens + self.prompt_tokens) / s,
            "prompts_per_sec": self.prompts / s,
            "per_phase": self.per_phase,
        }


def build_record(cfg: Dict[str, Any], *, hardware: Dict, timings: Dict, seeds_per_sec: float,
                 base_train_reward: float, base_test_acc: float, ensemble_results: Dict,
                 best_sigma: float, top_k_perturbs: List[Tuple[int, float]],
                 fineweb_base: Optional[Dict], fineweb_ensemble: Optional[Dict],
                 fineweb_manifest_sha: Optional[str], throughput: Optional[Dict] = None,
                 probes: Optional[List[Dict]] = None) -> Dict[str, Any]:
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
        "throughput": throughput or {},
        "base_train_reward": base_train_reward,
        "base_test_accuracy": base_test_acc,
        "ensemble_accuracy": ens_acc,
        "best_ensemble_accuracy": best_ens_acc,
        "fineweb": {
            "base": fineweb_base,
            "ensemble": fineweb_ensemble,
            "manifest_sha256": fineweb_manifest_sha,
        },
        "probes": probes or [],
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
        "| date | commit | config | model | hardware | pop | seeds/s | gen-tok/s | prompts/s | base acc | ens acc | base bpb | ens bpb | total |\n"
        "|------|--------|--------|-------|----------|-----|---------|-----------|-----------|----------|---------|----------|---------|-------|\n"
    )
    if not os.path.exists(records_md):
        with open(records_md, "w") as f:
            f.write(header)
    hw = record["hardware"]
    hw_str = f"{hw.get('gpu_count', '?')}x{hw.get('gpu', '?')}"
    tp = record.get("throughput", {}) or {}
    row = (f"| {record['timestamp'][:10]} | {record['git_commit']} | {record['config_name']} "
           f"| {record['model'].split('/')[-1]} | {hw_str} | {record['population_size']} "
           f"| {record['seeds_per_sec']:.2f} | {tp.get('gen_tokens_per_sec', 0):.0f} | {tp.get('prompts_per_sec', 0):.2f} "
           f"| {_fmt_pct(record['base_test_accuracy']*100 if record['base_test_accuracy'] else None)} "
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
        max_num_seqs=cfg.get("max_num_seqs"), max_model_len=cfg.get("max_model_len"),
    )
    timings["launch"] = time.perf_counter() - t

    meter = TokenMeter()
    fineweb_base = fineweb_ensemble = None
    fw_manifest_sha = None
    probes: List[Dict] = []
    try:
        # ---- base eval ----
        t = time.perf_counter()
        base_outs: List = []
        base_train_reward, base_test_acc = randopt.evaluate_base_model(
            engines, handler, train_prompts, test_prompts, train_datas, test_datas, sampling_params,
            on_outputs=base_outs.append)
        timings["base_eval"] = time.perf_counter() - t
        meter.add("base_eval", base_outs, timings["base_eval"])

        # ---- sampling (the hot loop; seeds/sec) ----
        t = time.perf_counter()
        samp_outs: List = []
        perf, best_sigma = randopt.run_sampling(
            args, engines, handler, train_prompts, train_datas, sampling_params,
            on_outputs=samp_outs.append)
        timings["sampling"] = time.perf_counter() - t
        seeds_per_sec = args.population_size / max(timings["sampling"], 1e-9)
        meter.add("sampling", samp_outs, timings["sampling"])
        sp = meter.per_phase["sampling"]
        print(f"\n>>> THROUGHPUT: {seeds_per_sec:.2f} seeds/sec | "
              f"{sp['gen_tok_per_sec']:.0f} gen-tok/sec | {sp['prompts_per_sec']:.2f} prompts/sec "
              f"({args.population_size} seeds, {sp['gen_tokens']} gen-tok in {timings['sampling']:.1f}s)")

        # ---- selection ----
        sorted_perturbs = sorted(perf.items(), key=lambda x: x[1], reverse=True)
        top_k_perturbs = [(s, sg) for (s, sg), _ in sorted_perturbs[:max_top_k]]
        ranked = [((s, sg), r) for (s, sg), r in sorted_perturbs]   # all, best-first

        # ---- ensemble eval (task acc) ----
        t = time.perf_counter()
        ens_outs: List = []
        ensemble_results = randopt.run_ensemble_evaluation(
            args, engines, handler, test_prompts, test_datas, top_k_perturbs, sampling_params, base_test_acc,
            on_outputs=ens_outs.append)
        timings["ensemble_eval"] = time.perf_counter() - t
        meter.add("ensemble_eval", ens_outs, timings["ensemble_eval"])

        # ---- per-seed probes (top-N individual seeds, incl. FineWeb) ----
        probe_n = int(cfg.get("probe_top", 0) or 0)
        if probe_n > 0:
            t = time.perf_counter()
            probes = _run_probes(cfg, engines, handler, tokenizer, test_prompts, test_datas,
                                 ranked[:probe_n], sampling_params, base_test_acc)
            timings["probes"] = time.perf_counter() - t

        # ---- FineWeb held-out bpb (base + ensemble) ----
        t = time.perf_counter()
        fineweb_base, fineweb_ensemble, fw_manifest_sha = _run_fineweb(
            cfg, engines, tokenizer, top_k_perturbs)
        timings["fineweb"] = time.perf_counter() - t

        timings["total"] = time.perf_counter() - t0
        throughput = meter.summary()

        hardware = detect_hardware()
        record = build_record(
            cfg, hardware=hardware, timings=timings, seeds_per_sec=seeds_per_sec,
            base_train_reward=base_train_reward, base_test_acc=base_test_acc,
            ensemble_results=ensemble_results, best_sigma=best_sigma, top_k_perturbs=top_k_perturbs,
            fineweb_base=fineweb_base, fineweb_ensemble=fineweb_ensemble,
            fineweb_manifest_sha=fw_manifest_sha, throughput=throughput, probes=probes)
        rec_path = write_record(record, run_dir)
        print(f"\n=== RECORD written: {rec_path} ===")
        print(f"  seeds/sec={seeds_per_sec:.2f}  gen-tok/sec={throughput['gen_tokens_per_sec']:.0f}  "
              f"prompts/sec(amortized over generation)={throughput['prompts_per_sec']:.2f}")
        print(json.dumps({k: record[k] for k in
                          ["seeds_per_sec", "throughput", "base_test_accuracy",
                           "best_ensemble_accuracy", "fineweb", "timings_s"]},
                         indent=2, default=float))
    finally:
        cleanup_engines(engines, pgs)


def _fineweb_prepare(cfg, tokenizer):
    """Resolve + tokenize the held-out FineWeb slice. Returns
    (chunks, total_bytes, manifest_sha) or (None, None, None) if unavailable."""
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
    if fw.get("max_chunks"):
        chunks = chunks[:fw["max_chunks"]]
    return chunks, fineweb.heldout_total_bytes(docs), manifest_sha


def _bpb_for_current_weights(eng, chunks, total_bytes):
    """FineWeb bpb of whatever weights are currently live on engine `eng`."""
    from vllm import SamplingParams
    import ray
    sp = SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0.0, detokenize=False)

    def gen(prompts, sampling_params, use_tqdm=False):
        return ray.get(eng.generate.remote(prompts, sampling_params, use_tqdm=use_tqdm))

    lps = _collect_logprobs(gen, chunks, sp)
    return fineweb.bpb_from_token_logprobs(lps, total_bytes)


def _run_fineweb(cfg, engines, tokenizer, top_k_perturbs):
    """Base + ensemble FineWeb bpb on engines[0] (sequential; correct for the
    1-engine standard). Returns (base_dict, ensemble_dict, manifest_sha)."""
    from vllm import SamplingParams
    import ray

    chunks, total_bytes, manifest_sha = _fineweb_prepare(cfg, tokenizer)
    if chunks is None:
        return None, None, None
    fw = cfg["fineweb"]
    eng = engines[0]
    sp = SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0.0, detokenize=False)

    def gen(prompts, sampling_params, use_tqdm=False):
        # vLLM's LLM.generate has use_tqdm as keyword-only (cf. randopt.py).
        return ray.get(eng.generate.remote(prompts, sampling_params, use_tqdm=use_tqdm))

    # base
    ray.get(eng.collective_rpc.remote("reset_to_base_weights", args=()))
    base_res = _bpb_for_current_weights(eng, chunks, total_bytes)
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


def _run_probes(cfg, engines, handler, tokenizer, test_prompts, test_datas,
                ranked, sampling_params, base_test_acc):
    """Evaluate the top-N *individual* seeds downstream, each on its own:
    single-model GSM8K test accuracy AND held-out FineWeb bpb. This answers
    "are the best individual perturbations good on their own, before voting?"
    Returns a list of per-seed dicts (also embedded in the record)."""
    import ray
    eng = engines[0]
    chunks, total_bytes, _ = _fineweb_prepare(cfg, tokenizer)
    out: List[Dict] = []
    print(f"\n{'='*60}\nPER-SEED PROBES (top {len(ranked)} individual seeds)\n{'='*60}")
    for rank, ((seed, sigma), train_reward) in enumerate(ranked, 1):
        # Apply this seed's perturbation from base (single model).
        ray.get(eng.collective_rpc.remote("apply_perturbation", args=(int(seed), float(sigma))))

        # single-model test accuracy (reuse the handler's voting/correctness path).
        # Probe generations are diagnostic and intentionally NOT folded into the
        # headline throughput meter (which measures the main base/sampling/ensemble
        # pipeline).
        outputs = ray.get(eng.generate.remote(test_prompts, sampling_params, use_tqdm=False))
        correct = _single_model_accuracy(handler, outputs, test_datas)
        acc = 100.0 * correct / len(test_datas) if test_datas else 0.0

        rec = {"rank": rank, "seed": int(seed), "sigma": float(sigma),
               "train_reward": float(train_reward),
               "test_accuracy": acc, "test_correct": correct, "n_test": len(test_datas)}

        # single-model FineWeb bpb
        if chunks is not None:
            bpb = _bpb_for_current_weights(eng, chunks, total_bytes)
            rec["fineweb_bpb"] = bpb.bpb
            rec["fineweb_nats_per_token"] = bpb.nats_per_token

        out.append(rec)
        print(f"  #{rank} seed={seed} σ={sigma}: train_reward={train_reward:.3f} "
              f"test_acc={acc:.1f}% [base {base_test_acc*100:.1f}%]"
              + (f" bpb={rec.get('fineweb_bpb', float('nan')):.4f}" if chunks is not None else ""))

    ray.get(eng.collective_rpc.remote("reset_to_base_weights", args=()))
    return out


def _single_model_accuracy(handler, outputs, test_datas) -> int:
    """Count correct answers for ONE model's outputs (mirrors the per-model
    correctness used inside evaluate_base_model / ensemble voting)."""
    correct = 0
    for i, data in enumerate(test_datas):
        response_text = outputs[i].outputs[0].text
        if handler.name == "countdown":
            numbers = test_datas[i].get("numbers")
            answer, is_valid, _ = handler.extract_answer_for_voting(response_text, numbers=numbers)
            answer = answer if is_valid else ""
        elif hasattr(handler, "extract_answer_for_voting"):
            answer = handler.extract_answer_for_voting(response_text) or ""
        else:
            answer = handler.extract_answer(response_text) or ""
        if not answer:
            continue
        if hasattr(handler, "is_voted_answer_correct"):
            ok = handler.is_voted_answer_correct(answer, data["ground_truth"])
        else:
            ok = handler.is_answer_correct(handler.format_answer_for_check(answer), data["ground_truth"])
        correct += int(bool(ok))
    return correct


def parse_args():
    p = argparse.ArgumentParser(description="RandOpt speedrun")
    p.add_argument("--config", required=True, help="Path to a configs/*.yaml")
    p.add_argument("--run-name", default=None, help="Subdir name under speedrun-runs/")
    p.add_argument("--build-fineweb", action="store_true",
                   help="Only build the held-out FineWeb slice, then exit")
    p.add_argument("--probe_top", type=int, default=None,
                   help="Evaluate the top-N individual seeds downstream on their own "
                        "(single-model GSM8K test acc + FineWeb bpb each). Overrides "
                        "config's probe_top. 0 disables.")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    cfg = load_config(a.config)
    if a.probe_top is not None:           # CLI overrides config
        cfg["probe_top"] = a.probe_top
    if a.build_fineweb:
        fw = cfg.get("fineweb", {})
        man = fineweb.build_heldout(fw.get("path", fineweb.DEFAULT_HELDOUT_PATH),
                                    num_docs=fw.get("build_num_docs", 256))
        print(json.dumps(man.__dict__, indent=2))
        sys.exit(0)
    run_name = a.run_name or f"{cfg['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(RUNS_DIR, run_name)
    main(cfg, run_dir)

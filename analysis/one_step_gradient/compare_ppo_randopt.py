#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from verl.utils.reward_score.gsm8k import compute_score as gsm8k_compute_score

DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument(
        "--ppo_ckpt",
        type=str,
        default="",
        help="Path to PPO-step-1 actor saved in HF format (must contain model.safetensors).",
    )
    p.add_argument(
        "--data_path",
        type=str,
        default="",
    )
    p.add_argument("--out_dir", type=str, default="outputs/compare_grpo_randopt")
    p.add_argument("--num_random", type=int, default=300, help="Number of random directions to evaluate.")
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--eval_max_len", type=int, default=512)
    p.add_argument("--eval_max_new_tokens", type=int, default=256)
    p.add_argument("--gsm8k_score_method", type=str, default="strict", choices=["strict", "flexible"])
    p.add_argument(
        "--ppo_log_path",
        type=str,
        default="",
        help="Optional path to run_vision trainer log (e.g. logs/verl_<jobid>.out) for extracting grad_norm.",
    )
    p.add_argument("--sigma", type=float, default=0.001, help="RandOpt sigma for the standalone noise model.")
    p.add_argument("--randopt_seed", type=int, default=42, help="Seed for RandOpt noise.")
    p.add_argument("--random_dir_seed_offset", type=int, default=10_000)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return p.parse_args()


def setup_distributed(device_arg: str) -> tuple[torch.device, int, int]:
    # Support both single-process and torchrun-style distributed execution.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return torch.device(device_arg), 0, 1

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed CUDA run requested, but CUDA is not available.")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
    else:
        device = torch.device(device_arg)
        backend = "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    rank = dist.get_rank()
    return device, rank, world_size


def _extract_last_float(pattern: str, text: str) -> Optional[float]:
    vals = re.findall(pattern, text)
    if not vals:
        return None
    try:
        return float(vals[-1])
    except ValueError:
        return None


def post_clip_grad_norm(pre_clip_norm: Optional[float], clip_grad: Optional[float]) -> float:
    """Return L2 norm after global grad clipping."""
    if pre_clip_norm is None:
        return float("nan")
    if clip_grad is None or clip_grad <= 0:
        return float(pre_clip_norm)
    return float(min(pre_clip_norm, clip_grad))


def try_extract_grad_norms(log_path: Path) -> Dict[str, float]:
    # Parse the latest scalar metrics from trainer logs for reporting only.
    if not log_path.exists():
        return {}
    text = log_path.read_text(errors="ignore")
    actor = _extract_last_float(r"actor/grad_norm:\s*(?:np\.float64\()?([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\)?", text)
    critic = _extract_last_float(r"critic/grad_norm:\s*(?:np\.float64\()?([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\)?", text)
    actor_lr = _extract_last_float(r"actor/lr:\s*(?:np\.float64\()?([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\)?", text)
    clip_grad = _extract_last_float(r"'clip_grad':\s*([+-]?\d+(?:\.\d+)?)", text)
    ppo_step = _extract_last_float(r"(?:^|\s)step:(\d+)\s-\s", text)
    train_global_step = _extract_last_float(r"train/global_step:\s*(\d+)", text)
    out: Dict[str, float] = {}
    if actor is not None:
        out["actor_grad_norm"] = actor
    if critic is not None:
        out["critic_grad_norm"] = critic
    if actor_lr is not None:
        out["actor_lr"] = actor_lr
    if clip_grad is not None:
        out["clip_grad"] = clip_grad
    if ppo_step is not None:
        out["ppo_step"] = ppo_step
    if train_global_step is not None:
        out["train_global_step"] = train_global_step
    return out


def resolve_ppo_log_path(arg_path: str, ckpt_path: str) -> Optional[Path]:
    if arg_path:
        p = Path(arg_path).expanduser()
        if not p.is_absolute():
            p = Path.cwd() / p
        return p

    ckpt_mtime = 0.0
    ckpt = Path(ckpt_path)
    if ckpt.exists():
        ckpt_mtime = ckpt.stat().st_mtime

    logs_dir = Path.cwd() / "logs"
    candidates = sorted(logs_dir.glob("verl_*.out"), key=lambda x: x.stat().st_mtime, reverse=True)
    for p in candidates:
        if p.stat().st_mtime + 1.0 >= ckpt_mtime:
            return p
    return candidates[0] if candidates else None


def build_eval_batch(
    tokenizer,
    data_path: str,
    batch_size: int,
    max_len: int,
    device: torch.device,
):
    # Build one fixed eval batch so all model variants are compared on identical prompts.
    df = pd.read_parquet(data_path)
    if len(df) < batch_size:
        raise RuntimeError(f"Need >= {batch_size} rows in {data_path}, got {len(df)}.")
    df = df.iloc[:batch_size].reset_index(drop=True)

    input_ids_list: List[List[int]] = []
    ground_truths: List[str] = []
    for row in df.to_dict("records"):
        messages = list(row["prompt"])
        answer = str(row["reward_model"]["ground_truth"])
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        if len(prompt_ids) > max_len:
            prompt_ids = prompt_ids[-max_len:]
        input_ids_list.append(prompt_ids)
        ground_truths.append(answer)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    max_t = max(len(x) for x in input_ids_list)
    B = len(input_ids_list)
    input_ids = torch.full((B, max_t), pad_id, dtype=torch.long)
    attn = torch.zeros((B, max_t), dtype=torch.long)
    for i, ids in enumerate(input_ids_list):
        start = max_t - len(ids)
        input_ids[i, start:] = torch.tensor(ids, dtype=torch.long)
        attn[i, start:] = 1
    print(f"[eval] B={B} max_prompt_t={max_t}")
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attn.to(device),
        "ground_truths": ground_truths,
    }


@torch.no_grad()
def compute_accuracy(model, batch, tokenizer, max_new_tokens: int, score_method: str) -> float:
    """Mean GSM8K score over the eval batch using greedy generation."""
    gen_out = model.generate(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_lens = batch["attention_mask"].sum(dim=1).tolist()
    scores: List[float] = []
    for i, prompt_len in enumerate(prompt_lens):
        gen_tokens = gen_out[i, int(prompt_len) :]
        pred_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        gt = batch["ground_truths"][i]
        score = gsm8k_compute_score(pred_text, gt, method=score_method)
        scores.append(float(score))
    return float(np.mean(scores)) if scores else 0.0


@torch.no_grad()
def l2_norm_of_state(state: Dict[str, torch.Tensor]) -> float:
    s = 0.0
    for t in state.values():
        s += float(t.detach().float().pow(2).sum().item())
    return math.sqrt(s)


@torch.no_grad()
def l2_diff_of_states(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    s = 0.0
    keys = a.keys() & b.keys()
    for k in keys:
        s += float((a[k].detach().float() - b[k].detach().float()).pow(2).sum().item())
    return math.sqrt(s)


@torch.no_grad()
def random_direction_norm_sq(model, seed: int) -> float:
    gen = torch.Generator(device=next(model.parameters()).device).manual_seed(int(seed))
    sq = 0.0
    for p in model.parameters():
        eps = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
        sq += float(eps.float().pow(2).sum().item())
    return sq


@torch.no_grad()
def apply_unit_random_direction_(model, base_state: Dict[str, torch.Tensor], seed: int, scale: float):
    gen = torch.Generator(device=next(model.parameters()).device).manual_seed(int(seed))
    for name, p in model.named_parameters():
        eps = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
        p.copy_(base_state[name] + scale * eps)


@torch.no_grad()
def restore_state_(model, state: Dict[str, torch.Tensor]):
    for name, p in model.named_parameters():
        p.copy_(state[name])

@torch.no_grad()
def make_randopt_state(base_state: Dict[str, torch.Tensor], seed: int, sigma: float) -> Dict[str, torch.Tensor]:
    new_state: Dict[str, torch.Tensor] = {}
    for name, base in base_state.items():
        gen = torch.Generator(device=base.device).manual_seed(int(seed))
        noise = torch.randn(base.shape, dtype=base.dtype, device=base.device, generator=gen)
        new_state[name] = base + sigma * noise
    return new_state

def main():
    args = parse_args()
    device, rank, world_size = setup_distributed(args.device)
    is_rank0 = rank == 0
    out_dir = Path(args.out_dir)
    if is_rank0:
        out_dir.mkdir(parents=True, exist_ok=True)
    dtype = DTYPE_MAP[args.dtype]
    if is_rank0:
        print("=" * 78)
        print("compare_grpo_randopt.py")
        print(f"  base_model = {args.base_model}")
        print(f"  ppo_ckpt   = {args.ppo_ckpt}")
        print(f"  data_path  = {args.data_path}")
        print(f"  out_dir    = {out_dir}")
        print(f"  world_size = {world_size}")
        print(f"  device/dtype = {device} / {dtype}")
        print("=" * 78)

    ppo_log_path = resolve_ppo_log_path(args.ppo_log_path, args.ppo_ckpt)
    grad_meta: Dict[str, float] = {}
    if ppo_log_path is not None:
        grad_meta = try_extract_grad_norms(ppo_log_path)
        if grad_meta and is_rank0:
            print(f"[meta] extracted grad norms from {ppo_log_path}")
            print(f"[meta] actor_grad_norm={grad_meta.get('actor_grad_norm')} critic_grad_norm={grad_meta.get('critic_grad_norm')}")
        elif is_rank0:
            print(f"[meta] no grad_norm found in {ppo_log_path}")
    elif is_rank0:
        print("[meta] no ppo trainer log found under logs/verl_*.out")

    if is_rank0:
        print("\n[1/5] loading tokenizer + base model ...")
    # Base model is the reference point for all delta computations.
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype).to(device)
    model.eval()
    base_state = {n: p.detach().clone() for n, p in model.named_parameters()}
    n_params = sum(p.numel() for p in model.parameters())
    if is_rank0:
        print(f"  total params = {n_params:,}")

    if is_rank0:
        print("\n[2/5] building eval batch ...")
    eval_batch = build_eval_batch(
        tokenizer, args.data_path, args.eval_batch_size, args.eval_max_len, device
    )

    base_acc = compute_accuracy(
        model,
        eval_batch,
        tokenizer,
        max_new_tokens=args.eval_max_new_tokens,
        score_method=args.gsm8k_score_method,
    )
    if is_rank0:
        print(f"  base_acc = {base_acc:.4f}")

    if is_rank0:
        print("\n[3/5] loading GRPO/PPO step-1 model + computing gradient_acc ...")
    # This checkpoint represents the actual one-step policy update.
    grpo_model = AutoModelForCausalLM.from_pretrained(args.ppo_ckpt, torch_dtype=dtype).to(device)
    grpo_model.eval()
    grad_acc = compute_accuracy(
        grpo_model,
        eval_batch,
        tokenizer,
        max_new_tokens=args.eval_max_new_tokens,
        score_method=args.gsm8k_score_method,
    )
    grpo_norm = l2_norm_of_state(dict(grpo_model.named_parameters()))
    base_norm = l2_norm_of_state(base_state)
    delta_grpo = l2_diff_of_states(dict(grpo_model.named_parameters()), base_state)
    if is_rank0:
        print(f"  grpo_acc = {grad_acc:.4f}")
    del grpo_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if is_rank0:
        print("\n[4/5] making RandOpt(sigma) model + L2 norms ...")
    # RandOpt baseline: add Gaussian noise directly to base weights.
    randopt_state = make_randopt_state(base_state, seed=args.randopt_seed, sigma=args.sigma)
    randopt_norm = l2_norm_of_state(randopt_state)
    delta_randopt = l2_diff_of_states(randopt_state, base_state)
    restore_state_(model, randopt_state)
    randopt_acc = compute_accuracy(
        model,
        eval_batch,
        tokenizer,
        max_new_tokens=args.eval_max_new_tokens,
        score_method=args.gsm8k_score_method,
    )
    del randopt_state
    restore_state_(model, base_state)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    summary = {
        "base": {"||theta||_2": base_norm, "acc": base_acc, "||delta theta||_2": 0.0},
        "grpo_step1": {"||theta||_2": grpo_norm, "acc": grad_acc, "||delta theta||_2": delta_grpo},
        "randopt_sigma": {
            "||theta||_2": randopt_norm,
            "acc": randopt_acc,
            "||delta theta||_2": delta_randopt,
            "sigma": args.sigma,
            "seed": args.randopt_seed,
        },
    }
    if is_rank0:
        print("\n=== L2 norm summary ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    if is_rank0:
        print("\n[5/5] random-direction acc landscape (matches randopt_one_grad_step.py) ...")
    # Match random-step radius to the true PPO update norm for fair comparison.
    step_radius = delta_grpo
    if is_rank0:
        print(f"  step_radius = ||theta_grpo - theta_base||_2 = {step_radius:.4g}")

    local_indices = list(range(rank, args.num_random, world_size))
    local_random_accs = np.zeros(len(local_indices), dtype=np.float64)
    t0 = time.time()
    for local_i, global_i in enumerate(local_indices):
        seed_i = args.random_dir_seed_offset + global_i
        # Sample random direction then rescale it to the same L2 step radius.
        norm_sq = random_direction_norm_sq(model, seed_i)
        scale = step_radius / math.sqrt(norm_sq)
        apply_unit_random_direction_(model, base_state, seed_i, scale)
        local_random_accs[local_i] = compute_accuracy(
            model,
            eval_batch,
            tokenizer,
            max_new_tokens=args.eval_max_new_tokens,
            score_method=args.gsm8k_score_method,
        )
        if (local_i + 1) % 25 == 0 or local_i == len(local_indices) - 1:
            dt = time.time() - t0
            print(
                f"[rank {rank}] random {local_i + 1:4d}/{len(local_indices)} "
                f"(global idx {global_i:4d}) acc={local_random_accs[local_i]:.4f} "
                f"elapsed={dt:.1f}s ({dt / (local_i + 1):.2f}s/sample)"
            )
        if world_size == 1 and ((global_i + 1) % 25 == 0 or global_i == args.num_random - 1):
            np.savez(
                out_dir / "accs.npz",
                base_acc=base_acc,
                gradient_acc=grad_acc,
                random_accs=local_random_accs[: local_i + 1],
                step_radius=step_radius,
                randopt_acc=randopt_acc,
                delta_randopt=delta_randopt,
            )
    restore_state_(model, base_state)

    if world_size > 1:
        vals = torch.zeros(args.num_random, dtype=torch.float64, device=device)
        mask = torch.zeros(args.num_random, dtype=torch.float64, device=device)
        for local_i, global_i in enumerate(local_indices):
            vals[global_i] = local_random_accs[local_i]
            mask[global_i] = 1.0
        dist.all_reduce(vals, op=dist.ReduceOp.SUM)
        dist.all_reduce(mask, op=dist.ReduceOp.SUM)
        random_accs = vals.cpu().numpy()
        if not np.allclose(mask.cpu().numpy(), 1.0):
            raise RuntimeError("Distributed random_acc aggregation failed: some indices are missing.")
    else:
        random_accs = local_random_accs

    if not is_rank0:
        return

    random_mean = float(random_accs.mean())
    random_std = float(random_accs.std(ddof=1))
    if random_std > 0:
        sigma_event = (grad_acc - random_mean) / random_std
    else:
        sigma_event = float("nan")
    gap = float(grad_acc - random_mean)
    print(
        f"\nresult: base_acc={base_acc:.4f}  gradient_acc={grad_acc:.4f}  "
        f"random_mean={random_mean:.4f}  random_std={random_std:.6f}  "
        f"gap(grad-random)={gap:.4f}  sigma={sigma_event:.2f}"
    )

    # Persist machine-readable metrics for later analysis.
    np.savez(
        out_dir / "accs.npz",
        base_acc=base_acc,
        gradient_acc=grad_acc,
        random_accs=random_accs,
        step_radius=step_radius,
        randopt_acc=randopt_acc,
        delta_randopt=delta_randopt,
        sigma_event=sigma_event,
    )

    actor_update_equiv_grad_l2 = float("nan")
    actor_lr = grad_meta.get("actor_lr")
    if actor_lr is not None and actor_lr > 0:
        actor_update_equiv_grad_l2 = step_radius / actor_lr
    actor_grad_post_clip = post_clip_grad_norm(
        grad_meta.get("actor_grad_norm"), grad_meta.get("clip_grad")
    )
    critic_grad_post_clip = post_clip_grad_norm(
        grad_meta.get("critic_grad_norm"), grad_meta.get("clip_grad")
    )

    with open(out_dir / "summary.txt", "w") as f:
        # Persist a human-readable report with both accuracy and update-scale metadata.
        f.write("Compare GRPO/PPO one-step update vs RandOpt noise vs random directions\n")
        f.write("=" * 70 + "\n")
        f.write(f"base_model = {args.base_model}\n")
        f.write(f"ppo_ckpt   = {args.ppo_ckpt}\n")
        f.write(f"sigma      = {args.sigma}\n")
        f.write(f"num_random = {args.num_random}\n")
        f.write(f"world_size = {world_size}\n\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
        if ppo_log_path is not None:
            f.write(f"ppo_log_path = {ppo_log_path}\n")
        if grad_meta:
            f.write(
                "grad_norms: "
                f"actor_pre_clip={grad_meta.get('actor_grad_norm', float('nan')):.6g}, "
                f"actor_post_clip={actor_grad_post_clip:.6g}, "
                f"critic_pre_clip={grad_meta.get('critic_grad_norm', float('nan')):.6g}, "
                f"critic_post_clip={critic_grad_post_clip:.6g}, "
                f"clip_grad={grad_meta.get('clip_grad', float('nan')):.6g}, "
                f"ppo_step={int(grad_meta.get('ppo_step', -1))}, "
                f"train_global_step={int(grad_meta.get('train_global_step', -1))}\n"
            )
        f.write(
            "actual_update: "
            f"actor_delta_theta_l2={step_radius:.6g}, "
            f"actor_lr={grad_meta.get('actor_lr', float('nan')):.6g}, "
            f"actor_equiv_grad_l2(delta/lr)={actor_update_equiv_grad_l2:.6g}\n"
        )
        f.write("\n")
        f.write(f"step_radius (||delta theta_grpo||) = {step_radius:.6g}\n")
        f.write(f"random_accs: mean={random_mean:.4f}, std={random_std:.6f}\n")
        f.write(f"gradient_acc = {grad_acc:.4f}\n")
        f.write(f"sigma_event = {sigma_event:.2f}\n")

    print(f"saved: {out_dir / 'accs.npz'}")
    print(f"saved: {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()

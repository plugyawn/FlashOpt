#!/usr/bin/env python3
"""
Modal smoke test for the RandOpt speedrun runtime — serverless GPU, automatic
teardown (no pod to leak; the container is reclaimed when the function returns).

  modal run scripts/modal_smoke.py                # tier1: Triton kernel == torch ref on CUDA
  modal run scripts/modal_smoke.py --tier all     # + tier2 end-to-end speedrun (vLLM)

Tier 1 (critical, cheap): validates the fused Rademacher kernel against the
pure-torch reference BIT-FOR-BIT on real CUDA + the GPU throughput micro-bench.
Tier 2 (best-effort): full pipeline on Qwen2.5-1.5B-Instruct (base eval -> seed
sampling -> top-k ensemble -> FineWeb bpb -> record). Needs `modal` auth.
"""
import os
import modal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Exclude the big/irrelevant trees from the code mount.
IGNORE = [
    ".git", ".git/**", ".venv", ".venv/**", "baselines", "baselines/**",
    "speedrun-runs", "speedrun-runs/**", "**/__pycache__", "**/__pycache__/**",
    "env.local", "*.pyc", "data/**",
]

app = modal.App("randopt-smoke")

kernel_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "torch", "triton", "pytest")
    .add_local_dir(REPO, "/root/RandOpt", ignore=IGNORE)
)

e2e_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm", "ray", "pandas", "pyarrow", "datasets", "transformers",
                 "huggingface-hub", "pyyaml", "numpy")
    # debian_slim has no CUDA toolkit (nvcc); vLLM's default FlashInfer sampler
    # JIT-compiles a kernel at runtime and needs nvcc. Route sampling to the
    # native PyTorch top-k/top-p path instead (greedy decode is unaffected).
    # On Hopper (H100) vLLM probes an FP8 DeepGEMM path during CUDA-graph warmup
    # that needs the `deep_gemm` package (absent here). Our weights are bf16, so
    # disable that path rather than ship the heavy build.
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0",
          "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
          "VLLM_USE_DEEP_GEMM": "0"})
    .add_local_dir(REPO, "/root/RandOpt", ignore=IGNORE)
)


@app.function(gpu="L4", image=kernel_image, timeout=1200)
def kernel_test():
    import os, sys, subprocess
    sys.path.insert(0, "/root/RandOpt")
    os.chdir("/root/RandOpt")
    import torch
    from core import perturb
    info = (f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
            f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'} "
            f"triton={perturb.triton_available()}")
    print(">>>", info)
    assert torch.cuda.is_available() and perturb.triton_available(), "need CUDA+Triton in image"
    # Capture pytest output so the per-test results + bench come back to the caller.
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_kernel_gpu.py", "-v", "-s"],
                       cwd="/root/RandOpt", capture_output=True, text=True)
    print(r.stdout[-4000:]); print(r.stderr[-2000:])
    return {"tier": "kernel", "returncode": r.returncode, "device_info": info,
            "pytest_tail": r.stdout[-4000:]}


@app.function(gpu="L4", image=kernel_image, timeout=1200)
def throughput():
    """Measure the real GPU speedup: fused single-pass reconstruct (no sign
    materialisation) vs the upstream perturb+restore (2x full randn) at model
    scale, in bf16. Returns measured ms + speedup."""
    import os, sys, time
    sys.path.insert(0, "/root/RandOpt")
    os.chdir("/root/RandOpt")
    import torch
    from core import perturb as P

    dev = "cuda"
    info = f"{torch.cuda.get_device_name(0)} torch={torch.__version__}"
    out = {"device": info, "runs": []}
    for n in [50_000_000, 200_000_000]:
        base = torch.randn(n, device=dev, dtype=torch.bfloat16)
        buf = torch.empty_like(base)
        sigma = 1e-3

        def old():  # upstream: perturb (randn+add) then restore (randn+sub)
            g = torch.Generator(device=dev); g.manual_seed(5)
            base.add_(sigma * torch.randn(n, device=dev, dtype=torch.bfloat16, generator=g))
            g2 = torch.Generator(device=dev); g2.manual_seed(5)
            base.add_(-sigma * torch.randn(n, device=dev, dtype=torch.bfloat16, generator=g2))
            torch.cuda.synchronize()

        def new():  # fused reconstruct from base, single pass, no materialisation
            P.reconstruct_into(buf, base, 5, sigma, 0, kernel="triton"); torch.cuda.synchronize()

        for fn in (old, new):  # warmup
            fn()

        def med(fn, reps=7):
            ts = []
            for _ in range(reps):
                t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
            return sorted(ts)[len(ts) // 2]

        t_old, t_new = med(old), med(new)
        out["runs"].append({"n": n, "old_ms": t_old * 1e3, "new_ms": t_new * 1e3,
                            "speedup": t_old / max(t_new, 1e-9)})
    return out


def _run_e2e(config: str, git_commit: str, probe_top: int, n_train: int, n_test: int,
             data_cmd: list = None):
    """Shared e2e body: prep dataset, run the speedrun for `config`, return record.

    data_cmd: explicit data-prep argv (default = GSM8K). MATH passes its own.
    """
    import os, sys, subprocess, glob, json
    os.chdir("/root/RandOpt")
    # git isn't installed in the container; pass the host commit through so the
    # record's provenance is real. PYTHONUNBUFFERED=1 so progress prints stream
    # live to `modal app logs` (block buffering on a pipe makes a working run look
    # stalled — ~200 batch lines buffer before becoming visible).
    env = dict(os.environ, PYTHONPATH="/root/RandOpt", RANDOPT_GIT_COMMIT=git_commit,
               PYTHONUNBUFFERED="1")
    if data_cmd is None:
        data_cmd = [sys.executable, "scripts/make_gsm8k_smoke.py",
                    "--n-train", str(n_train), "--n-test", str(n_test)]
    subprocess.run(data_cmd, check=True, env=env)
    cmd = [sys.executable, "speedrun.py", "--config", config]
    if probe_top:
        cmd += ["--probe_top", str(probe_top)]
    r = subprocess.run(cmd, env=env)
    recs = sorted(glob.glob("speedrun-runs/*/record.json"))
    rec = json.load(open(recs[-1])) if recs else {}
    return {"tier": "e2e", "config": config, "returncode": r.returncode, "record": rec}


@app.function(gpu="L4", image=e2e_image, timeout=3600)
def e2e_smoke(git_commit: str = "unknown", probe_top: int = 0):
    return _run_e2e("configs/smoke_1gpu_small.yaml", git_commit, probe_top, 32, 32)


# 512-seed run: more data, on L4 (24GB, already validated). Memory budget in the
# config leaves room for the resident base copy + CUDA-graph pools.
@app.function(gpu="L4", image=e2e_image, timeout=5400)
def e2e_512(git_commit: str = "unknown", probe_top: int = 5):
    return _run_e2e("configs/smoke_512seed.yaml", git_commit, probe_top, 80, 80)


# Larger Qwen on a single H100-80GB, CUDA graphs ON. 7B = ~15GB weights + ~15GB
# resident base copy = ~30GB, leaving ~50GB for KV + graph pools — comfortable.
@app.function(gpu="H100", image=e2e_image, timeout=5400)
def e2e_7b(git_commit: str = "unknown", probe_top: int = 3):
    return _run_e2e("configs/tuned_1xh100.yaml", git_commit, probe_top, 64, 64)


# 512-seed run: 7B on H100, larger GSM8K slice (128 train / 256 test) + PROPER
# FineWeb bpb (real 256-doc slice). Long (~30-45min) -> run with `modal run --detach`.
@app.function(gpu="H100", image=e2e_image, timeout=10800)
def e2e_run512(git_commit: str = "unknown", probe_top: int = 5):
    return _run_e2e("configs/run_512_7b_h100.yaml", git_commit, probe_top, 128, 256)


# 512-seed run on a HARDER task: MATH-500 levels 4-5 (base has real headroom vs
# GSM8K's ~89% ceiling). Same 7B/H100/graphs/population — only difficulty changes.
@app.function(gpu="H100", image=e2e_image, timeout=14400)
def e2e_math512(git_commit: str = "unknown", probe_top: int = 5):
    import sys
    data_cmd = [sys.executable, "scripts/make_math500_hard.py",
                "--levels", "4", "5", "--n-train", "64", "--n-test", "192"]
    return _run_e2e("configs/run_512_7b_math500hard.yaml", git_commit, probe_top, 64, 192,
                    data_cmd=data_cmd)


def _host_commit():
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def _report_e2e(res2, save_name):
    import json
    print("\n================ E2E RESULT ================")
    rec = res2.get("record") or {}
    print("config:", res2.get("config"), "| returncode:", res2.get("returncode"))
    print(json.dumps({k: rec.get(k) for k in
                      ["seeds_per_sec", "throughput", "base_test_accuracy",
                       "best_ensemble_accuracy", "fineweb", "probes", "timings_s"]},
                     indent=2, default=float))
    if rec:
        d = os.path.join(REPO, "speedrun-runs", save_name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "record.json"), "w") as f:
            json.dump(rec, f, indent=2)
        print(f"wrote speedrun-runs/{save_name}/record.json")


@app.local_entrypoint()
def main(tier: str = "1", probe_top: int = 0):
    import json
    if tier == "throughput":
        res = throughput.remote()
        print("\n================ GPU THROUGHPUT ================")
        print("device:", res["device"])
        for r in res["runs"]:
            print(f"  n={r['n']:>12,}  old(perturb+restore)={r['old_ms']:7.2f}ms  "
                  f"fused reconstruct={r['new_ms']:7.2f}ms  speedup={r['speedup']:.2f}x")
        return
    if tier == "512":
        # 512-seed real-population run (L4). probe_top defaults to 5 here.
        res2 = e2e_512.remote(git_commit=_host_commit(), probe_top=probe_top or 5)
        _report_e2e(res2, "modal_512seed")
        return
    if tier == "3b":
        res2 = e2e_3b.remote(git_commit=_host_commit(), probe_top=probe_top or 3)
        _report_e2e(res2, "modal_3b")
        return
    if tier == "7b":
        res2 = e2e_7b.remote(git_commit=_host_commit(), probe_top=probe_top or 3)
        _report_e2e(res2, "modal_7b")
        return
    if tier == "run512":
        res2 = e2e_run512.remote(git_commit=_host_commit(), probe_top=probe_top or 5)
        _report_e2e(res2, "modal_run512")
        return
    if tier == "math512":
        res2 = e2e_math512.remote(git_commit=_host_commit(), probe_top=probe_top or 5)
        _report_e2e(res2, "modal_math512")
        return
    # tier "2" runs ONLY the e2e (kernel already validated); "1"/"all" run kernel.
    if tier != "2":
        res = kernel_test.remote()
        print("\n================ KERNEL RESULT ================")
        print(json.dumps({k: v for k, v in res.items() if k != "pytest_tail"}, indent=2))
        print(res.get("pytest_tail", ""))
        if res.get("returncode") != 0:
            print("!! kernel test FAILED")
    if tier in ("2", "all"):
        res2 = e2e_smoke.remote(git_commit=_host_commit(), probe_top=probe_top)
        _report_e2e(res2, "modal_smoke")

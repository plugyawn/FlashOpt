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


@app.function(gpu="L4", image=e2e_image, timeout=3600)
def e2e_smoke():
    import os, sys, subprocess, glob, json
    os.chdir("/root/RandOpt")
    env = dict(os.environ, PYTHONPATH="/root/RandOpt")
    subprocess.run([sys.executable, "scripts/make_gsm8k_smoke.py", "--n-train", "32", "--n-test", "32"],
                   check=True, env=env)
    r = subprocess.run([sys.executable, "speedrun.py", "--config", "configs/smoke_1gpu_small.yaml"],
                       env=env)
    recs = sorted(glob.glob("speedrun-runs/*/record.json"))
    rec = json.load(open(recs[-1])) if recs else {}
    return {"tier": "e2e", "returncode": r.returncode, "record": rec}


@app.local_entrypoint()
def main(tier: str = "1"):
    import json
    if tier == "throughput":
        res = throughput.remote()
        print("\n================ GPU THROUGHPUT ================")
        print("device:", res["device"])
        for r in res["runs"]:
            print(f"  n={r['n']:>12,}  old(perturb+restore)={r['old_ms']:7.2f}ms  "
                  f"fused reconstruct={r['new_ms']:7.2f}ms  speedup={r['speedup']:.2f}x")
        return
    res = kernel_test.remote()
    print("\n================ KERNEL RESULT ================")
    print(json.dumps({k: v for k, v in res.items() if k != "pytest_tail"}, indent=2))
    print(res.get("pytest_tail", ""))
    if res.get("returncode") != 0:
        print("!! kernel test FAILED")
    if tier in ("2", "all"):
        res2 = e2e_smoke.remote()
        print("\n================ E2E RESULT ================")
        rec = res2.get("record") or {}
        print("returncode:", res2.get("returncode"))
        print(json.dumps({k: rec.get(k) for k in
                          ["seeds_per_sec", "base_test_accuracy", "best_ensemble_accuracy",
                           "fineweb", "timings_s"]}, indent=2))
        if rec:
            d = os.path.join(REPO, "speedrun-runs", "modal_smoke")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "record.json"), "w") as f:
                json.dump(rec, f, indent=2)
            print("wrote speedrun-runs/modal_smoke/record.json")

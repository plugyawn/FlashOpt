"""
GPU numerics test: the Triton fused kernel must equal the pure-torch reference
BIT-FOR-BIT (that is the whole point of defining our own RNG instead of using
Triton's Philox). Runs only where CUDA + Triton are available; skips otherwise.

Run on the smoke pod:  python -m pytest tests/test_kernel_gpu.py -q -s
"""
import os
import sys
import time

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import perturb as P

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and P.triton_available()),
    reason="needs CUDA + Triton",
)

DEV = "cuda"
SHAPES = [(1,), (2, 3), (127,), (1024,), (255, 511), (4096, 8192)]
SEEDS = [0, 1, 42, 123456, 2**31 - 1]
SIGMAS = [5e-4, 1e-3, 2e-3]


def _ref(base, seed, sigma):
    out = torch.empty_like(base)
    P.reconstruct_into(out, base, seed, sigma, 0, kernel="torch")
    return out


def _triton(base, seed, sigma):
    out = torch.empty_like(base)
    P.reconstruct_into(out, base, seed, sigma, 0, kernel="triton")
    return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_triton_equals_reference(dtype):
    torch.manual_seed(0)
    for shape in SHAPES:
        base = torch.randn(shape, device=DEV).to(dtype)
        for seed in SEEDS:
            for sigma in SIGMAS:
                t = _triton(base, seed, sigma)
                r = _ref(base, seed, sigma)
                assert torch.equal(t, r), (
                    f"triton != torch ref @ shape={shape} dtype={dtype} seed={seed} sigma={sigma}; "
                    f"max|d|={ (t.float()-r.float()).abs().max().item() }")


def test_triton_signs_match_cpu_reference():
    # The same seeds must give the same signs on GPU (triton) and CPU (torch ref).
    n = 100_003
    base = torch.zeros(n, device=DEV, dtype=torch.float32)
    t = _triton(base, 12345, 1.0).cpu()                 # = sign(seed, idx)
    cpu = P.rademacher_signs(12345, 0, n, "cpu", torch.float32)
    assert torch.equal(t, cpu), "GPU triton signs differ from CPU reference"


def test_switch_matches_reconstruct_gpu():
    base = torch.randn(4096, device=DEV, dtype=torch.float32)
    sigma = 1e-3
    live = _triton(base, 11, sigma)
    P.switch_inplace(live, 11, 22, sigma, 0, kernel="triton")
    target = _triton(base, 22, sigma)
    assert torch.allclose(live, target, atol=1e-6)


def test_reconstruct_drift_free_gpu():
    base = torch.randn(8192, device=DEV, dtype=torch.bfloat16)
    sigma = 1e-3
    a1 = _triton(base, 7, sigma)
    _ = _triton(base, 8, sigma)
    a2 = _triton(base, 7, sigma)
    assert torch.equal(a1, a2)


def test_kernel_throughput_report():
    """Informational: fused reconstruct (no materialisation) vs old perturb+restore
    (2x full randn). Prints the GPU speedup — the real headline number."""
    n = 200_000_000
    base = torch.randn(n, device=DEV, dtype=torch.bfloat16)
    out = torch.empty_like(base)
    sigma = 1e-3

    def old_perturb_restore():
        g = torch.Generator(device=DEV); g.manual_seed(5)
        noise = torch.randn(n, device=DEV, dtype=torch.bfloat16, generator=g)
        base.add_(sigma * noise)
        g2 = torch.Generator(device=DEV); g2.manual_seed(5)
        noise2 = torch.randn(n, device=DEV, dtype=torch.bfloat16, generator=g2)
        base.add_(-sigma * noise2)
        torch.cuda.synchronize()

    def new_reconstruct():
        P.reconstruct_into(out, base, 5, sigma, 0, kernel="triton")
        torch.cuda.synchronize()

    for fn in (old_perturb_restore, new_reconstruct):  # warmup
        fn()

    def med(fn, reps=5):
        ts = []
        for _ in range(reps):
            t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
        return sorted(ts)[len(ts) // 2]

    t_old, t_new = med(old_perturb_restore), med(new_reconstruct)
    print(f"\n[GPU n={n:,}] old perturb+restore={t_old*1e3:.1f}ms  "
          f"fused reconstruct={t_new*1e3:.1f}ms  speedup={t_old/max(t_new,1e-9):.2f}x")

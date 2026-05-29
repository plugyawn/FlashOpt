"""
CPU correctness tests for the fused dense-Rademacher perturbation math
(core/perturb.py).  These run on any box with torch (no GPU/Triton needed) and
pin down the contract that the Triton kernel must also satisfy on GPU
(tests/test_kernel_gpu.py).

Run:  .venv/bin/python -m pytest tests/test_perturb.py -q
  or: .venv/bin/python tests/test_perturb.py
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import perturb as P

torch.manual_seed(0)
DEVICE = "cpu"


# --------------------------------------------------------------------------- #
# 1. RNG correctness: vectorised fmix32 == scalar Python oracle (incl. overflow)
# --------------------------------------------------------------------------- #
def test_fmix32_torch_matches_python():
    # Cover values that force int64 overflow in the multiply (h near 2**32).
    vals = [0, 1, 2, 255, 0xDEADBEEF, 0xFFFFFFFF, 0x7FFFFFFF, 0x85EBCA6B, 0xC2B2AE35, 123456789]
    h = torch.tensor(vals, dtype=torch.int64)
    got = P._fmix32_torch(h.clone()).tolist()
    exp = [P._fmix32_py(v) for v in vals]
    assert got == exp, f"fmix32 mismatch:\n got={got}\n exp={exp}"


def test_signs_match_scalar_oracle():
    # rademacher_signs (vectorised) must equal sign_for_py element-by-element,
    # including across the 32-bit position boundary (pos > 2**32).
    for seed in [0, 1, 42, 2**31 - 1, 123456]:
        for base_off in [0, 7, 2**31, 2**32 - 3, 2**32 + 5, 5 * 10**9]:
            n = 37
            vec = P.rademacher_signs(seed, base_off, n, DEVICE).tolist()
            exp = [P.sign_for_py(seed, base_off + i) for i in range(n)]
            assert vec == exp, f"sign mismatch seed={seed} off={base_off}\n{vec}\n{exp}"


def test_signs_are_pm_one():
    s = P.rademacher_signs(7, 0, 100000, DEVICE)
    assert torch.all((s == 1.0) | (s == -1.0))


# --------------------------------------------------------------------------- #
# 2. Distribution: zero-mean, unit |value|, unit variance
# --------------------------------------------------------------------------- #
def test_distribution():
    n = 1_000_000
    s = P.rademacher_signs(2024, 0, n, DEVICE)
    assert abs(s.mean().item()) < 5e-3, s.mean().item()          # balanced
    assert torch.allclose(s.abs(), torch.ones_like(s))           # |value| == 1
    assert abs(s.var(unbiased=False).item() - 1.0) < 1e-3        # var == 1
    # different seeds are nearly uncorrelated
    s2 = P.rademacher_signs(2025, 0, n, DEVICE)
    corr = (s * s2).mean().item()
    assert abs(corr) < 5e-3, corr


# --------------------------------------------------------------------------- #
# 3. Determinism
# --------------------------------------------------------------------------- #
def test_determinism():
    a = P.rademacher_signs(99, 1000, 5000, DEVICE)
    b = P.rademacher_signs(99, 1000, 5000, DEVICE)
    assert torch.equal(a, b)


# --------------------------------------------------------------------------- #
# 4. Drift-free reconstruction (the core win over in-place perturb/restore)
# --------------------------------------------------------------------------- #
def _reconstruct_tensor(base, seed, sigma, dtype):
    out = torch.empty_like(base, dtype=dtype)
    P.reconstruct_into(out, base.to(dtype), seed, sigma, 0, kernel="torch")
    return out


def test_reconstruct_drift_free_fp32():
    base = torch.randn(4096, dtype=torch.float32)
    sigma = 1e-3
    a1 = _reconstruct_tensor(base, seedA := 11, sigma, torch.float32)
    _ = _reconstruct_tensor(base, 22, sigma, torch.float32)        # perturb to B
    _ = _reconstruct_tensor(base, 33, sigma, torch.float32)        # perturb to C
    a2 = _reconstruct_tensor(base, seedA, sigma, torch.float32)     # back to A
    assert torch.equal(a1, a2), "reconstruct-from-base must be bit-identical regardless of history"
    # and it equals base + sigma*signs exactly
    signs = P.rademacher_signs(seedA, 0, base.numel(), DEVICE).reshape(base.shape)
    assert torch.allclose(a1, base + sigma * signs, atol=0, rtol=0)


def test_reconstruct_drift_free_bf16():
    # In production the base copy is stored in the model dtype (bf16), so the
    # helper feeds a bf16 base; the parity contract is computed from that.
    base_f32 = torch.randn(8192, dtype=torch.float32)
    base_bf16 = base_f32.to(torch.bfloat16)
    sigma = 2e-3
    a1 = _reconstruct_tensor(base_bf16, 11, sigma, torch.bfloat16)
    _ = _reconstruct_tensor(base_bf16, 22, sigma, torch.bfloat16)
    a2 = _reconstruct_tensor(base_bf16, 11, sigma, torch.bfloat16)
    assert torch.equal(a1, a2), "bf16 reconstruct must be bit-identical (computed from base, not delta)"
    # bf16 parity contract: (base_bf16->f32 + sigma*sign)->bf16  (matches the Triton kernel)
    signs = P.rademacher_signs(11, 0, base_bf16.numel(), DEVICE).reshape(base_bf16.shape)
    expected = (base_bf16.to(torch.float32) + sigma * signs).to(torch.bfloat16)
    assert torch.equal(a1, expected)


# --------------------------------------------------------------------------- #
# 5. switch_inplace == reconstruct in fp32; document bf16 drift
# --------------------------------------------------------------------------- #
def test_switch_equals_reconstruct_fp32():
    base = torch.randn(4096, dtype=torch.float32)
    sigma = 1e-3
    live = _reconstruct_tensor(base, 11, sigma, torch.float32)     # at A
    P.switch_inplace(live, 11, 22, sigma, 0, kernel="torch")        # A -> B
    target = _reconstruct_tensor(base, 22, sigma, torch.float32)    # B
    assert torch.allclose(live, target, atol=1e-6), (live - target).abs().max().item()


def test_switch_bf16_drift_motivates_reconstruct():
    # bf16 in-place switching accumulates rounding error: after a loop of hops
    # returning to the start seed, `live` does NOT match a fresh reconstruct.
    # This is exactly why the runtime reconstructs absolutely from a resident
    # base (drift-free) instead of delta-switching for long runs.
    base = torch.randn(8192, dtype=torch.float32).to(torch.bfloat16)
    sigma = 1e-3
    live = _reconstruct_tensor(base, 1, sigma, torch.bfloat16)
    prev = 1
    for s in [2, 3, 4, 5, 1]:
        P.switch_inplace(live, prev, s, sigma, 0, kernel="torch")
        prev = s
    target = _reconstruct_tensor(base, 1, sigma, torch.bfloat16)     # fresh reconstruct of seed 1
    drift = (live.to(torch.float32) - target.to(torch.float32)).abs().max().item()
    # Drift is real (non-zero) but bounded by a few bf16 ULPs (~2^-7 at |w|~1).
    assert drift > 0.0, "expected measurable bf16 drift from delta-switching"
    assert drift < 0.05, f"drift {drift} larger than a few bf16 ULPs — unexpected"


# --------------------------------------------------------------------------- #
# 6. Global-offset coherence (engine/layout independence)
# --------------------------------------------------------------------------- #
def test_global_offset_coherence():
    # Two params keyed with a running offset must equal one concatenated tensor.
    n1, n2 = 1000, 1500
    seed = 7
    big = P.rademacher_signs(seed, 0, n1 + n2, DEVICE)
    p1 = P.rademacher_signs(seed, 0, n1, DEVICE)
    p2 = P.rademacher_signs(seed, n1, n2, DEVICE)
    assert torch.equal(big, torch.cat([p1, p2]))


def test_iter_perturb_params_offsets():
    params = [("b.weight", torch.zeros(10)), ("a.weight", torch.zeros(20)),
              ("visual.x", torch.zeros(5)), ("c.weight", torch.zeros(7))]
    should = lambda name: not name.startswith("visual.")
    plan = P.iter_perturb_params(params, should)
    names = [n for n, _, _ in plan]
    offs = [o for _, _, o in plan]
    assert names == ["a.weight", "b.weight", "c.weight"]   # sorted; visual skipped
    assert offs == [0, 20, 30]                              # 20, then +10, then +7


# --------------------------------------------------------------------------- #
# 7. Model-level reconstruction on a fake model
# --------------------------------------------------------------------------- #
def test_reconstruct_model_from_base():
    torch.manual_seed(1)
    base = {f"layer{i}.weight": torch.randn(64, 32) for i in range(4)}
    base["visual.enc"] = torch.randn(16)                  # should NOT be perturbed
    live = {k: v.clone() for k, v in base.items()}
    named = list(live.items())
    should = lambda name: not name.startswith("visual.")
    sigma = 5e-3
    touched = P.reconstruct_model_from_base(
        named, lambda n: base[n], seed=123, sigma=sigma, should_perturb=should, kernel="torch")
    assert touched == 4
    # visual untouched
    assert torch.equal(live["visual.enc"], base["visual.enc"])
    # each perturbed layer == base + sigma*signs with the right running offset
    plan = P.iter_perturb_params(list(base.items()), should)
    for name, _, off in plan:
        signs = P.rademacher_signs(123, off, base[name].numel(), DEVICE).reshape(base[name].shape)
        assert torch.allclose(live[name], base[name] + sigma * signs, atol=0)


# --------------------------------------------------------------------------- #
# 8. Micro-benchmark: new single-pass reconstruct vs old perturb+restore
# --------------------------------------------------------------------------- #
def _old_perturb_restore(live, seed, sigma):
    """Mimics utils/worker_extn.py original: 2 full randn materialisations + 2 passes."""
    gen = torch.Generator(device=live.device); gen.manual_seed(seed)
    noise = torch.randn(live.shape, dtype=live.dtype, device=live.device, generator=gen)
    live.add_(sigma * noise)
    gen2 = torch.Generator(device=live.device); gen2.manual_seed(seed)
    noise2 = torch.randn(live.shape, dtype=live.dtype, device=live.device, generator=gen2)
    live.add_(-sigma * noise2)


def _median(fn, reps=3):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2]


def test_microbench_restore_elimination():
    """The hardware-independent win is **pass-count**: the old loop does a
    perturb pass AND a restore pass per seed; the new loop reconstructs once
    (the next seed's reconstruct overwrites from base, so restore is free).

    With matched RNG, 1 pass must beat 2 passes. We also print the raw
    upstream comparison (2x torch.randn) for context. NOTE: on this CPU
    fallback the int64 hash is slower per-pass than torch.randn, so the raw
    number can favour upstream here — the real win (zero sign materialisation +
    inline RNG) is on the GPU Triton path, measured in the smoke test.
    """
    n = 20_000_000
    base = torch.randn(n, dtype=torch.float32)
    out = torch.empty_like(base)
    sigma = 1e-3

    one_pass = lambda: P.reconstruct_into(out, base, 5, sigma, 0, kernel="torch")
    two_pass = lambda: (P.reconstruct_into(out, base, 5, sigma, 0, kernel="torch"),
                        P.reconstruct_into(out, base, 6, sigma, 0, kernel="torch"))
    t1 = _median(one_pass)
    t2 = _median(two_pass)

    live = base.clone()
    t_old_raw = _median(lambda: _old_perturb_restore(live, 5, sigma))

    print(f"\n[microbench n={n:,}] matched-RNG: 1-pass(new)={t1*1e3:.1f}ms  "
          f"2-pass(old perturb+restore)={t2*1e3:.1f}ms  -> restore-elim {t2/t1:.2f}x")
    print(f"[microbench] raw upstream 2x-randn perturb+restore={t_old_raw*1e3:.1f}ms "
          f"(CPU torch.randn is faster per-pass than the int64 hash fallback; "
          f"GPU Triton path materialises NO signs and is the real fast path)")

    # Correctness: one reconstruct yields exactly base + sigma*sign (no restore needed).
    signs = P.rademacher_signs(5, 0, n, DEVICE)
    P.reconstruct_into(out, base, 5, sigma, 0, kernel="torch")
    assert torch.equal(out, base + sigma * signs)
    # Pass-count win is hardware-independent.
    assert t1 < t2, "removing the restore pass must save time with matched RNG"


# --------------------------------------------------------------------------- #
# Standalone runner (so we can run without pytest)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

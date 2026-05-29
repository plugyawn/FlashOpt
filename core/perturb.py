"""
Fused dense-Rademacher weight perturbation for RandOpt speedrunning.

The original RandOpt switch (``utils/worker_extn.py``) materialised a full
model-sized ``torch.randn`` tensor *per parameter, twice per seed* (apply +
restore), with a ``synchronize()`` + ``empty_cache()`` on every call. For a
~100B model that is ~hundreds of GB of RNG traffic per seed evaluated.

This module replaces that with a **fused dense-Rademacher** scheme:

    W = W0 + sigma * R(seed, global_index)        R in {-1, +1}

* RNG is a **counter-based hash** (murmur3 ``fmix32`` over a folded 64-bit
  position), computed *inline* — no noise tensor is materialised on the fast
  (Triton) path.
* Because the hash is a pure function of ``(seed, global_index)``, we always
  reconstruct *absolutely from a resident base copy* (``W = W0 + sigma*R``),
  which is **drift-free** regardless of history and identical across engines.
* The "restore" pass disappears: the next seed's reconstruct overwrites the
  live weights from the base.

The RNG is defined here (not delegated to Triton's Philox) so the pure-torch
reference and the Triton kernel produce **bit-identical** signs — seeds are
reproducible across backends. The torch reference doubles as the CPU/no-Triton
fallback.

Backend selection: env ``RANDOPT_KERNEL`` in {``auto``, ``triton``, ``torch``}
(default ``auto`` -> Triton on CUDA when available, else torch).
"""
from __future__ import annotations

import os
from typing import Callable, Iterable, List, Sequence, Tuple

import torch

# ---------------------------------------------------------------------------
# Counter-based RNG: murmur3 fmix32 over a folded 64-bit position.
# Defined identically for torch (below) and Triton (further down) so that
# reference == kernel bit-for-bit.  All arithmetic is mod 2**32.
# ---------------------------------------------------------------------------
_MASK32 = 0xFFFFFFFF
_C1 = 0x85EBCA6B
_C2 = 0xC2B2AE35


def _fmix32_py(h: int) -> int:
    """Scalar murmur3 finalizer on a uint32 (pure-Python oracle for tests)."""
    h &= _MASK32
    h ^= h >> 16
    h = (h * _C1) & _MASK32
    h ^= h >> 13
    h = (h * _C2) & _MASK32
    h ^= h >> 16
    return h & _MASK32


def sign_for_py(seed: int, pos: int) -> float:
    """Scalar +/-1 sign for one (seed, position).  Reference oracle for tests.

    bit0 == 0 -> +1, bit0 == 1 -> -1.
    """
    lo = pos & _MASK32
    hi = (pos >> 32) & _MASK32
    h = _fmix32_py(lo)
    h = _fmix32_py(hi ^ h)
    h = _fmix32_py((seed & _MASK32) ^ h)
    return 1.0 - 2.0 * float(h & 1)


def _fmix32_torch(h: torch.Tensor) -> torch.Tensor:
    """Vectorised murmur3 finalizer.

    ``h`` is an int64 tensor whose values are in ``[0, 2**32)``.  We mask after
    every multiply, so all intermediates stay non-negative (logical shifts) and
    int64 multiply overflow wraps mod 2**64 — the low 32 bits we keep are
    exactly the uint32 result.  Verified bit-for-bit against ``_fmix32_py`` in
    tests/test_perturb.py.
    """
    h = h ^ (h >> 16)
    h = (h * _C1) & _MASK32
    h = h ^ (h >> 13)
    h = (h * _C2) & _MASK32
    h = h ^ (h >> 16)
    return h & _MASK32


def rademacher_signs(seed: int, offset: int, n: int, device, dtype=torch.float32) -> torch.Tensor:
    """Reference / fallback: dense +/-1 vector for positions ``[offset, offset+n)``.

    Returns a tensor of shape ``(n,)`` with values in {-1, +1}.  This DOES
    materialise the sign vector (the Triton path does not); it is the
    correctness oracle and the no-Triton fallback.
    """
    pos = torch.arange(n, device=device, dtype=torch.int64) + int(offset)
    lo = pos & _MASK32
    hi = (pos >> 32) & _MASK32
    h = _fmix32_torch(lo)
    h = _fmix32_torch(hi ^ h)
    h = _fmix32_torch((int(seed) & _MASK32) ^ h)
    bit = (h & 1).to(dtype)
    return 1.0 - 2.0 * bit


# ---------------------------------------------------------------------------
# Triton kernels (GPU fast path; no sign materialisation).  Guarded import.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised on GPU only
    import triton
    import triton.language as tl

    _HAS_TRITON = True

    @triton.jit
    def _tl_fmix32(h):
        # murmur3 fmix32; constants inlined as literals because Triton's JIT
        # cannot access module-level Python globals (must be constexpr/literal).
        # These MUST equal _C1/_C2 above so kernel == torch reference bit-for-bit.
        h = h ^ (h >> 16)
        h = h * tl.full((), 0x85EBCA6B, tl.uint32)  # _C1
        h = h ^ (h >> 13)
        h = h * tl.full((), 0xC2B2AE35, tl.uint32)  # _C2
        h = h ^ (h >> 16)
        return h

    @triton.jit
    def _reconstruct_kernel(out_ptr, base_ptr, seed, sigma, offset, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        idx = pid * BLOCK + tl.arange(0, BLOCK)
        mask = idx < n
        pos = offset + idx.to(tl.int64)
        lo = (pos & 0xFFFFFFFF).to(tl.uint32)
        hi = ((pos >> 32) & 0xFFFFFFFF).to(tl.uint32)
        s = tl.full((), seed, tl.uint32)
        h = _tl_fmix32(lo)
        h = _tl_fmix32(hi ^ h)
        h = _tl_fmix32(s ^ h)
        sign = 1.0 - 2.0 * (h & 1).to(tl.float32)
        base = tl.load(base_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        val = base + sigma * sign
        tl.store(out_ptr + idx, val.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _switch_kernel(live_ptr, seed_from, seed_to, sigma, offset, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        idx = pid * BLOCK + tl.arange(0, BLOCK)
        mask = idx < n
        pos = offset + idx.to(tl.int64)
        lo = (pos & 0xFFFFFFFF).to(tl.uint32)
        hi = ((pos >> 32) & 0xFFFFFFFF).to(tl.uint32)
        # sign(seed_to) - sign(seed_from)  in {-2, 0, +2}
        sf = tl.full((), seed_from, tl.uint32)
        st = tl.full((), seed_to, tl.uint32)
        hf = _tl_fmix32(_tl_fmix32(hi ^ _tl_fmix32(lo)) ^ sf)
        ht = _tl_fmix32(_tl_fmix32(hi ^ _tl_fmix32(lo)) ^ st)
        sign_f = 1.0 - 2.0 * (hf & 1).to(tl.float32)
        sign_t = 1.0 - 2.0 * (ht & 1).to(tl.float32)
        live = tl.load(live_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        val = live + sigma * (sign_t - sign_f)
        tl.store(live_ptr + idx, val.to(live_ptr.dtype.element_ty), mask=mask)

except Exception:  # ImportError or any Triton init failure
    _HAS_TRITON = False
    triton = None
    tl = None


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------
def triton_available() -> bool:
    return _HAS_TRITON


def resolve_backend(tensor: torch.Tensor, kernel: str | None = None) -> str:
    """Return 'triton' or 'torch' for this tensor + requested mode."""
    mode = (kernel or os.environ.get("RANDOPT_KERNEL", "auto")).lower()
    if mode == "torch":
        return "torch"
    if mode == "triton":
        if not _HAS_TRITON:
            raise RuntimeError("RANDOPT_KERNEL=triton but Triton is not available")
        return "triton"
    # auto
    if _HAS_TRITON and tensor.is_cuda:
        return "triton"
    return "torch"


def describe_backend() -> str:
    mode = os.environ.get("RANDOPT_KERNEL", "auto").lower()
    return f"kernel={mode} (triton_available={_HAS_TRITON})"


def _grid_blocks(n: int, block: int) -> int:
    return (n + block - 1) // block


# ---------------------------------------------------------------------------
# Per-tensor ops
# ---------------------------------------------------------------------------
def reconstruct_into(
    out: torch.Tensor,
    base: torch.Tensor,
    seed: int,
    sigma: float,
    offset: int,
    *,
    noise: str = "rademacher",
    kernel: str | None = None,
) -> None:
    """Write ``out = base + sigma * R(seed, offset..)`` for one tensor.

    ``out`` and ``base`` must be contiguous and the same shape. ``out`` may
    alias ``base`` only for the torch path (Triton reads & writes separate
    buffers in general, but aliasing is also fine since each element is
    independent).
    """
    n = out.numel()
    if noise == "gaussian":
        _gaussian_reconstruct(out, base, seed, sigma)
        return
    if noise != "rademacher":
        raise ValueError(f"unknown noise '{noise}'")

    backend = resolve_backend(out, kernel)
    if backend == "triton":
        BLOCK = 1024
        _reconstruct_kernel[(_grid_blocks(n, BLOCK),)](
            out.view(-1), base.view(-1), int(seed) & _MASK32, float(sigma), int(offset), n, BLOCK=BLOCK
        )
        return
    # torch reference / fallback (materialises signs; compute in fp32 then cast)
    signs = rademacher_signs(seed, offset, n, out.device, dtype=torch.float32).reshape(out.shape)
    val = base.to(torch.float32) + float(sigma) * signs
    out.copy_(val.to(out.dtype))


def switch_inplace(
    live: torch.Tensor,
    seed_from: int,
    seed_to: int,
    sigma: float,
    offset: int,
    *,
    noise: str = "rademacher",
    kernel: str | None = None,
) -> None:
    """In-place ``live += sigma * (R(seed_to) - R(seed_from))``.

    Drift-prone over many switches (bf16 rounding accumulates); prefer
    ``reconstruct_into`` from a resident base for long runs. Provided for the
    no-base / memory-constrained (e.g. MoE) regime.
    """
    if noise != "rademacher":
        raise ValueError("switch_inplace only supports rademacher noise")
    n = live.numel()
    backend = resolve_backend(live, kernel)
    if backend == "triton":
        BLOCK = 1024
        _switch_kernel[(_grid_blocks(n, BLOCK),)](
            live.view(-1), int(seed_from) & _MASK32, int(seed_to) & _MASK32,
            float(sigma), int(offset), n, BLOCK=BLOCK,
        )
        return
    s_from = rademacher_signs(seed_from, offset, n, live.device, torch.float32).reshape(live.shape)
    s_to = rademacher_signs(seed_to, offset, n, live.device, torch.float32).reshape(live.shape)
    val = live.to(torch.float32) + float(sigma) * (s_to - s_from)
    live.copy_(val.to(live.dtype))


def _gaussian_reconstruct(out: torch.Tensor, base: torch.Tensor, seed: int, sigma: float) -> None:
    """Legacy/research Gaussian path: one pass from base (still no restore), but
    materialises noise via a per-seed generator. Device-dependent RNG (not
    backend-portable) — Rademacher is the reproducible fast default."""
    gen = torch.Generator(device=out.device)
    gen.manual_seed(int(seed))
    noise = torch.randn(out.shape, dtype=out.dtype, device=out.device, generator=gen)
    val = base.to(torch.float32) + float(sigma) * noise.to(torch.float32)
    out.copy_(val.to(out.dtype))


# ---------------------------------------------------------------------------
# Model-level orchestration
# ---------------------------------------------------------------------------
def iter_perturb_params(
    named_params: Iterable[Tuple[str, torch.Tensor]],
    should_perturb: Callable[[str], bool],
) -> List[Tuple[str, torch.Tensor, int]]:
    """Return ``[(name, param, offset)]`` in a fixed sorted-by-name order.

    ``offset`` is the running element count over *perturbed* params only, so the
    RNG stream is a single coherent sequence across the whole model and is
    independent of engine/TP layout. Non-perturbed params are skipped and do
    not consume offset.
    """
    out: List[Tuple[str, torch.Tensor, int]] = []
    offset = 0
    for name, p in sorted(named_params, key=lambda x: x[0]):
        if should_perturb(name):
            out.append((name, p, offset))
            offset += p.numel()
    return out


def reconstruct_model_from_base(
    named_params: Iterable[Tuple[str, torch.Tensor]],
    base_lookup: Callable[[str], torch.Tensor],
    seed: int,
    sigma: float,
    should_perturb: Callable[[str], bool],
    *,
    noise: str = "rademacher",
    kernel: str | None = None,
) -> int:
    """Set every perturbable param to ``base + sigma*R(seed, .)``. Returns #params touched."""
    plan = iter_perturb_params(named_params, should_perturb)
    for name, p, off in plan:
        reconstruct_into(p.data, base_lookup(name), seed, sigma, off, noise=noise, kernel=kernel)
    return len(plan)


def switch_model(
    named_params: Iterable[Tuple[str, torch.Tensor]],
    seed_from: int,
    seed_to: int,
    sigma: float,
    should_perturb: Callable[[str], bool],
    *,
    kernel: str | None = None,
) -> int:
    """In-place delta-switch every perturbable param from seed_from to seed_to."""
    plan = iter_perturb_params(named_params, should_perturb)
    for name, p, off in plan:
        switch_inplace(p.data, seed_from, seed_to, sigma, off, kernel=kernel)
    return len(plan)


def perturbation_norm(seed: int, sigma: float, total_params: int, noise: str = "rademacher") -> float:
    """Expected L2 norm of the perturbation sigma*eps over `total_params` elements.

    For Rademacher and standard-normal alike, E||sigma*eps||_2 = sigma*sqrt(d),
    so a given sigma is comparable across the two noise types (the per-coordinate
    magnitude differs: Rademacher is exactly sigma, Gaussian is sigma*|N(0,1)|).
    """
    return float(sigma) * (float(total_params) ** 0.5)

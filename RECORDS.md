# RandOpt Speedrun Records

Each row is one run of the fixed standard. Reproduce with:
`python speedrun.py --config configs/standard_8xh100_qwen72b.yaml`

See [docs/SPEEDRUN.md](docs/SPEEDRUN.md) for the standard, the metric
(held-out FineWeb bits-per-byte), and the methodology. Rows are appended
automatically by `speedrun.py`; `seeds/s` is the optimization throughput.

## Kernel validation (GPU, measured)

Validated on **NVIDIA L4** (torch 2.12+cu130, triton 3.7) via Modal — full record
in `speedrun-runs/gpu_kernel_validation/record.json`:

- `tests/test_kernel_gpu.py`: **7 passed** — Triton kernel == pure-torch reference
  **bit-for-bit** (fp32/bf16/fp16), GPU signs == CPU signs, switch≡reconstruct,
  drift-free reconstruction.
- **Weight-switch throughput** (fused reconstruct vs. upstream perturb+restore),
  measured bf16 on the L4 (`modal run scripts/modal_smoke.py --tier throughput`):

  | elements | old perturb+restore | fused reconstruct | speedup |
  |----------|--------------------:|------------------:|--------:|
  | 50M      | 5.17 ms             | 0.91 ms           | **5.7×** |
  | 200M     | 20.66 ms            | 3.46 ms           | **6.0×** |

  Conservative: "old" is just upstream's 2×-`randn` perturb+restore; the real
  runtime additionally drops the per-call `empty_cache`/`synchronize` and the
  entire restore pass. Larger GPUs (H100) widen the gap (memory-bandwidth bound).

## End-to-end smoke (GPU, measured)

Full pipeline on **NVIDIA L4** via Modal (`modal run scripts/modal_smoke.py
--tier 2`), config `configs/smoke_1gpu_small.yaml`, **Qwen2.5-1.5B-Instruct**,
population 16, GSM8K (32 train / 32 test) — record in
`speedrun-runs/modal_smoke/record.json`:

| metric | base | ensemble (K=4) |
|--------|-----:|---------------:|
| GSM8K test accuracy | 28.1% (9/32) | **40.6% (13/32)**  (+12.5 pp) |
| held-out FineWeb **bpb** (lower=better) | 0.2665 | 0.2678 |

- **Ensemble lifts the task metric** by +12.5 pp — the majority vote over
  fused-Rademacher-perturbed models works end-to-end through the fast runtime.
- **FineWeb bpb is essentially flat** (ensemble marginally worse). Expected: at
  this tiny scale (1.5B, only 16 seeds, σ∈{1e-3,2e-3}) selection optimizes GSM8K
  reward, *not* language modeling, and held-out web text is far from the GSM8K
  distribution — there is no reason a 16-seed task-selected ensemble should
  improve generic LM bpb. The metric is wired through correctly; it simply
  doesn't move at smoke scale. Whether the 72B standard moves bpb is the open
  empirical question this harness exists to answer — not something to assume.
- Throughput **0.19 seeds/s** (16 seeds in 83 s) on a single L4, `enforce_eager`,
  greedy. This is a tiny smoke, not the 8×H100 standard.

| date | commit | config | model | hardware | pop | seeds/s | base acc | ens acc | base bpb | ens bpb | total |
|------|--------|--------|-------|----------|-----|---------|----------|---------|----------|---------|-------|
| 2026-05-30 | bf6e80b | smoke_1gpu_small | Qwen2.5-1.5B-Instruct | 1×NVIDIA L4 | 16 | 0.19 | 28.1% | 40.6% | 0.267 | 0.268 | 472s |

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

| date | commit | config | model | hardware | pop | seeds/s | base acc | ens acc | base bpb | ens bpb | total |
|------|--------|--------|-------|----------|-----|---------|----------|---------|----------|---------|-------|

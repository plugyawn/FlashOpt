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

### Throughput (measured, default-logged)

`seeds/sec` is the optimization rate; `gen-tok/sec` and `prompts/sec` are the
inference rates the harness now logs by default (amortized over the
base+sampling+ensemble generation phases). From the `--probe_top 3` L4 run:

- **0.20 seeds/sec**, **~1.3–1.4k generated-tokens/sec**, **~5.9 prompts/sec**
  (≈2.0k total tokens/sec incl. prefill) on a single L4, eager, greedy.
- Per-seed `--probe_top 3` (each top seed evaluated standalone, not voted): the
  best seed alone reached **37.5%** test acc (vs 28.1% base), but the #3 seed
  *dropped to 18.8%* despite equal train reward — i.e. high train reward doesn't
  guarantee held-out gains; the **K=4 ensemble (40.6%) beat every individual
  seed**. (bpb per seed ≈ 0.270, flat vs base 0.2665, as expected at this scale.)

### CUDA graphs at 7B (measured) — graphs + in-place perturbation are compatible

Qwen2.5-7B-Instruct on a single **H100-80GB**, `enforce_eager=false` (`configs/
tuned_1xh100.yaml`, 256 seeds), record `speedrun-runs/modal_7b/record.json`:

- **Graphs captured fine**: "Capturing CUDA graphs (decode, FULL) 51/51 … finished
  in 4 s, took 0.59 GiB" — one-time, amortized over the population.
- **Correctness under graphs**: sampling rewards *vary across seeds* (0.94/0.95/
  0.97/…), so replayed graphs read the freshly-perturbed weights — in-place
  reconstruction is compatible with captured graphs at 7B. ✓ (the key claim.)
- **Throughput**: **0.33 seeds/s, ~5.3k gen-tok/s, ~21 prompts/s** (≈7.8k
  total-tok/s incl. prefill) — **4× the gen-tok/s of the eager 1.5B run** despite
  a 4.7× larger model (graphs + H100 + `max_num_seqs=512`).
- **Quality caveat**: base is already **90.6%** on this 64-sample GSM8K slice and
  the K=25 ensemble is also 90.6% (+0.0) — *saturated at n=64*, not a method
  result. A probe seed hit 93.8% individually; the ensemble can't move a
  near-ceiling base on so few samples. FineWeb skipped (config `build_if_missing:
  false`, no slice prebuilt). Use a larger test slice to get a real ensemble Δ.

### 512-seed 7B/H100, larger slice + proper FineWeb (the headline run)

Qwen2.5-7B on **1×H100**, graphs ON, **512 seeds**, GSM8K **128 train / 256 test**,
held-out FineWeb = real 256-doc slice (sha `4682737b…`), `configs/run_512_7b_h100.yaml`,
record `speedrun-runs/modal_run512/record.json`:

- **Ensemble gain (both metrics improve):**
  - GSM8K: base **89.06%** (228/256) → K=25 **91.80%** (235/256, **+2.73 pp**),
    K=10 91.41% (+2.34 pp).
  - Held-out FineWeb **bpb**: base **0.5932** → ensemble **0.5918** (lower=better,
    over 205k scored tokens / 1.02 MB). Small but the *right direction* — the
    task-selected ensemble nudged generic LM held-out loss down, not up.
- **Throughput** (512 seeds): **0.28 seeds/s, 9,045 gen-tok/s, 36.1 prompts/s**
  (18.1M gen-tok; ~13.3k total-tok/s incl. prefill). Higher gen-tok/s than the
  256-seed run — wider batch over 128 train prompts saturates the H100 better.
  Sampling rewards varied 0.92–0.98 across all 512 seeds → graph+perturbation
  compatibility holds at full scale. ✓
- **Per-seed probes (top 5, evaluated standalone):** individual test acc ranges
  87.9–91.0% (vs 89.06% base) — i.e. some single seeds beat base, some don't, and
  none reliably beats the **91.8% ensemble**; per-seed bpb 0.596–0.649, all worse
  than the ensemble's 0.592. Voting helps on both metrics. Best σ=0.0005.
- Wall-clock 2,915 s total (sampling 1,861 s · ensemble 134 s · probes 375 s ·
  FineWeb 222 s · launch 306 s).

> The earlier `77e1977` attempt produced the accuracy/throughput above but
> **crashed in FineWeb** (a held-out chunk tokenized to exactly `max_model_len`,
> and the bpb pass requests 1 output token → `prompt_len+1 > max_model_len`).
> Fixed in `e724a9a` (`speedrun._fineweb_max_len` clamps chunk len to
> `max_model_len−2`, + regression test); this re-run completed the bpb.

| date | commit | config | model | hardware | pop | seeds/s | gen-tok/s | prompts/s | base acc | ens acc | base bpb | ens bpb | total |
|------|--------|--------|-------|----------|-----|---------|-----------|-----------|----------|---------|----------|---------|-------|
| 2026-05-30 | bf6e80b | smoke_1gpu_small | Qwen2.5-1.5B-Instruct | 1×NVIDIA L4 | 16 | 0.19 | — | — | 28.1% | 40.6% | 0.267 | 0.268 | 472s |
| 2026-05-30 | 36abea5 | smoke_1gpu_small (+probe) | Qwen2.5-1.5B-Instruct | 1×NVIDIA L4 | 16 | 0.20 | 1316 | 5.88 | 28.1% | 40.6% | 0.267 | 0.268 | — |
| 2026-05-30 | 27c5438 | tuned_1xh100 (graphs on) | Qwen2.5-7B-Instruct | 1×H100-80GB | 256 | 0.33 | 5296 | 20.9 | 90.6% | 90.6% | — | — | 1021s |
| 2026-05-30 | e724a9a | run_512_7b_h100 | Qwen2.5-7B-Instruct | 1×H100-80GB | 512 | 0.28 | 9045 | 36.1 | 89.1% | 91.8% | 0.593 | 0.592 | 2915s |

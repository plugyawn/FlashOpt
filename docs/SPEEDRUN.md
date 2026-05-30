# RandOpt Speedrun

A fixed standard for **gradient-free post-training by random optimization**, in
the spirit of the [nanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt):
fix the hardware, fix the model, fix the protocol, and race the wall clock — then
report quality on a held-out metric everyone shares.

## The method (RandOpt / "Neural Thickets")

Given pretrained weights `W₀`, draw perturbations `W(s,σ) = W₀ + σ·ε_s` for seeds
`s` and scales `σ`, score each on a small **train** set (greedy decode + task
reward), keep the **top-k**, and **majority-vote ensemble** them on the test set.
No gradients. The cost is dominated by *evaluating perturbations*, so the figure
of merit is **seeds evaluated per second**.

## What was slow, and the fix

The upstream switch (`utils/worker_extn.py`) did, **per seed**:

1. loop every parameter, allocate a full model-sized `torch.randn` noise tensor, add it;
2. *restore* by regenerating the **same** noise and subtracting it;
3. `torch.cuda.synchronize()` + `torch.cuda.empty_cache()` **every call**.

That is **two full-model RNG materializations per seed** plus cache churn. For a
72B model (144 GB bf16) it is ~288 GB of noise traffic per seed, ×population.

The fast runtime (`core/perturb.py`) switches to **dense Rademacher** noise
(`ε ∈ {−1,+1}`) and reconstructs **absolutely from a resident base copy**:

```
W = W₀ + σ · R(seed, global_index)
```

* `R` is a **counter-based hash** (murmur3 `fmix32` over a folded 64-bit position),
  evaluated **inline** in a Triton kernel — *no noise tensor is materialized*.
* Reconstruction is **absolute from base**, so it is **drift-free** regardless of
  history and identical across engines/TP layouts.
* The **restore pass disappears**: the next seed's reconstruct overwrites the live
  weights from the base. Per-call `synchronize`/`empty_cache` are removed (one
  sync per seed).

Net per seed: **2 full-model RNG materializations → 0**, **2 passes → 1**.

The RNG is defined by *us* (not Triton's Philox) so the Triton kernel and a
pure-torch reference produce **bit-identical** signs — seeds reproduce across
backends, and the CPU reference doubles as the no-Triton fallback. This is
asserted bit-for-bit in `tests/test_kernel_gpu.py`.

Why Rademacher is principled: it is zero-mean, unit-variance, symmetric, and the
maximum-entropy ±1 distribution; `E‖σε‖₂ = σ·√d` matches the Gaussian case, so a
given σ is comparable (Gaussian remains available via `noise: gaussian`, on the
slower materializing path).

## The standard

`configs/standard_8xh100_qwen72b.yaml`:

| | |
|---|---|
| Hardware | 8× H100-80GB, single node |
| Model | `Qwen/Qwen2.5-72B-Instruct` (dense, bf16) |
| Topology | TP=8, **1 engine** (see below) |
| Selection task | GSM8K, 200 train / 200 test, greedy, `max_tokens=1024` |
| Population | 512 seeds, σ ∈ {5e-4, 1e-3, 2e-3} |
| Ensemble | top 4% and 10% by train reward, majority vote |
| Held-out metric | FineWeb-edu `sample-10BT` bits-per-byte (fixed slice + sha256) |

### Why TP=8 / 1 engine (the base-resident trade-off)

The resident base copy gives drift-free reconstruction but **doubles weight
memory**: 72B → 288 GB. Across 8×H100-80GB that is ~36 GB/GPU at TP=8, leaving
~44 GB/GPU for KV — comfortable, but only **one** engine, so seeds are evaluated
serially (each seed's *inference* still uses all 8 GPUs, and the switch is one
cheap fused pass). To get data-parallel-over-seeds you scale **nodes** (16 GPUs →
TP=8 × 2 engines), not the single node. `gpu_memory_utilization` must leave room
for the base copy: `(1−util)·80GB ≥ per-GPU base shard` (≈18 GB at TP=8 → util ≤ ~0.7).

Memory-constrained alternative: drop the base copy and use the **in-place delta
switch** (`switch_to_seed`, `W += σ(R_to − R_from)`) for 1× memory and more
engines — at the cost of bf16 drift over long runs (documented and tested).

## The metric: held-out FineWeb bits-per-byte

nanoGPT reports raw token cross-entropy because its tokenizer is fixed (GPT-2 BPE).
Profile models have *different* tokenizers, so we report **bits-per-byte**, which
is tokenizer-invariant:

```
bpb = (Σ token NLL in nats) / (ln2 · total UTF-8 bytes)
```

Teacher-forced, obtained from vLLM `prompt_logprobs` (the logprob of each actual
token given its prefix) — reflecting the live (perturbed) weights with no custom
forward code. We report bpb for the **base** model and the **ensemble**, where the
ensemble distribution is the **probability average** of the top-k models'
next-token distributions (`p̄ = mean_m p_m`, `nll = −log p̄`), computed exactly
from per-model realized-token probabilities via a streaming accumulator.

FineWeb is *never* used for selection (selection is task reward), so any fixed
slice is a valid held-out language-modeling probe; the slice + its sha256 are
pinned in `eval/fineweb_heldout.jsonl(.manifest.json)`.

### bf16 resolution caveat — and why we reconstruct from base (measured)

A σ≈1e-3 perturbation on an O(1) weight is below bf16 resolution (ULP = 2⁻⁸ ≈
0.0039 at |w|≈1), so it rounds to ±1 ULP; perturbations survive mainly on
smaller-magnitude weights. Inherent to bf16 (upstream has it too) — choose σ
relative to the weight magnitudes and the dtype. Ensemble averaging is in fp32.

This is also the real reason for the **2× resident base copy** (absolute
reconstruction `W = W₀ + σ·R`) over the 1× in-place delta switch
(`W += σ(R_to − R_from)`). Measured drift (CPU, |w|≈1; `tests/test_perturb.py::
test_inplace_switch_drift`):

| dtype | σ | mean |Δ| vs fresh reconstruct | Δ/σ | grows with #hops? |
|-------|-----|---------------------------|------|--------|
| bf16  | 1e-3 | 7.7e-4 | **0.77** | **no — flat 16→2048 hops** |
| bf16  | 1e-2 | 2.0e-3 | 0.20 | no |
| fp16  | 1e-3 | 4.6e-5 | 0.046 | no |

Two findings: (1) the error does **not** accumulate over hops — `R∈{−1,+1}` is
exact in bf16, so each `+=` rounds at the local ULP and re-randomizes rather than
compounding. (2) But the per-switch rounding is a *large fraction of σ* at
production σ (0.77σ for bf16/σ=1e-3), which makes delta-switching **path-dependent**:
two different seed-chains reaching the same final seed differ by up to one ULP
(≈4σ), so a seed's weights aren't reproducible. Absolute reconstruction is
path-independent and bit-identical every run — that reproducibility (not
"preventing accumulation") is what the 2× copy buys. The 1× path
(`switch_model`) remains available when memory forces it and bit-exactness can be
traded away.

## "Batching profiles" — the honest version

You cannot cheaply co-batch many *dense* perturbations into one GEMM: a full sign
matrix `S` makes `Sx` a full matvec, so k profiles in one forward = k× the FLOPs,
and KV can't be shared across seeds (different weights → different KV). The real,
principled throughput levers — all used here — are:

1. the fused single-pass switch (no materialization);
2. deleting the restore pass + per-call cache churn;
3. CUDA graphs (`enforce_eager: false`) for inference throughput — compatible with
   in-place weight reconstruction (graphs reference weight *storage addresses*);
4. **data-parallel over seeds** across engines/GPUs (the existing structure);
5. maximizing per-engine tokens/sec (large continuous batch, high `gpu_memory_utilization`).

## Running

```bash
# CPU: verify the perturbation + bpb + harness math (no GPU needed)
.venv/bin/python -m pytest tests/test_perturb.py tests/test_worker.py \
                                tests/test_fineweb.py tests/test_speedrun.py -q

# GPU pod: verify Triton == reference, then the standard / a smoke
python -m pytest tests/test_kernel_gpu.py -q -s
python speedrun.py --config configs/smoke_1gpu_small.yaml          # tiny end-to-end
python speedrun.py --config configs/standard_8xh100_qwen72b.yaml   # THE standard

# build/refresh the fixed held-out FineWeb slice (records a sha256)
python speedrun.py --config configs/standard_8xh100_qwen72b.yaml --build-fineweb
```

`scripts/prime_smoke.sh` provisions the cheapest viable single-GPU Prime Intellect
pod, runs the smoke end-to-end, copies the record back, and tears the pod down.

## Records

Each run writes `speedrun-runs/<name>/record.json` (full repro blob: commit,
config sha, top-k seeds, timings, throughput, accuracies, bpb) and appends a row
to [`RECORDS.md`](../RECORDS.md). A run is reproducible from its commit + config +
FineWeb manifest sha + the recorded top-k seeds.

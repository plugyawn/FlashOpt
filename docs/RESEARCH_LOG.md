# RandOpt Research Log — how does this technique behave?

A running, honest log of what we measure about RandOpt (random weight-perturbation
post-training) as a research method. Focus for this phase: **K=1 on held-out test**
(does a *single* perturbation beat base, and does selecting by train reward find
it?) — ensemble is studied separately in [RECORDS.md](../RECORDS.md).

Instruments: `hotpath.py` (per-seed train+test eval, no ensemble),
`scripts/plot_hotpath.py` (K=1-vs-seeds curve + train→test transfer scatter),
`scripts/make_math500_hard.py` (level-filtered / stratified / sized splits).

Key metrics per run:
- **base_test** — unperturbed model accuracy on the held-out test slice.
- **best K=1 (oracle)** — best single seed's test acc (upper bound; cheats by
  looking at test).
- **K=1 by train** — test acc of the seed picked by *train* reward (the honest,
  deployable number; this is what selection actually gives you).
- **Spearman ρ(train, test)** — does train reward predict held-out gain? (the
  transfer quality; low ρ ⇒ selection is near-random).
- **frac>base** — fraction of seeds that beat base on test.

All MATH-500 = levels 4-5 unless noted; greedy decode; bf16; vLLM 0.22 on Modal.
Caveat throughout: vLLM is not bit-reproducible across runs under graphs+batching,
so absolute acc carries ~±1.5pp run-to-run noise.

---

## Baselines (unperturbed, MATH-500 lvl4-5 test, greedy)

| model | test slice | base acc |
|-------|-----------|---------:|
| Qwen2.5-1.5B-Instruct | lvl4-5, n=192 | **38.0%** (73/192) |
| Qwen2.5-1.5B-Instruct | lvl4-5, n=96 (sweep slice) | **35.4%** (34/96) |
| Qwen2.5-7B-Instruct   | lvl4-5, n=192 | ~60.4% (prior 512-runs; 61.98% on a re-run — ±1.5pp run-to-run) |

The 1.5B base differs by slice (38.0% on n=192 vs 35.4% on the n=96 sweep slice) —
small-n sampling, within noise. (7B lvl5-only/stratified baselines: TODO.)

---

## Experiment matrix (1B fast iteration, then 7B confirm)

The goal's questions → planned cells (all K=1, Qwen2.5-1.5B unless noted):

| axis | values | question |
|------|--------|----------|
| **train size** | 16 / 32 / 64 | does a *smaller* train set help or just add noise? |
| **grouping** | random vs stratified-by-level | does matching train/test level mix improve transfer (ρ)? |
| **train levels** | lvl4-5 vs lvl5-only | does selecting on *harder* problems transfer better to a hard test? |
| **σ** | 1e-4 / 5e-4 / 1e-3 | (confirm 5e-4 floor from RECORDS.md on the K=1 metric) |
| **noise** | rademacher vs gaussian | does dense-sign beat Gaussian for K=1? |
| **population** | 64 (1B) → 128 (7B final) | more seeds = better best-K=1 (diminishing) |

Each cell reports base / best-oracle / by-train / ρ / frac>base, plus a plot.

---

## Runs

### R0 — hotpath validation smoke (1.5B, pop=8, n=192)
`hp1b_smoke` — validated the K=1 instrument end-to-end on GPU (✓). base 38.0%,
oracle/by-train both 39.1%, ρ=0.64, frac>base 12.5%. n=8 is far too small to
conclude anything; recorded only as the instrument check.

### R1 — sweep1b: does train-set DESIGN improve K=1 transfer? (1.5B, pop=64, n=96)
4 cells, σ=5e-4, rademacher, L4. Same held-out n=96 test. **2/4 cells completed**
(`stratify`/`train16` OOM'd at gpu_mem_util=0.85 on the 22GB L4 — config fixed to
0.55, re-running).

| cell | train | base | best-K1 oracle | **K1 by train** | ρ(train,test) | frac>base |
|------|-------|-----:|---------------:|----------------:|--------------:|----------:|
| default   | 64 lvl4-5 random | 35.4% | 42.7% (+7.3) | **38.5% (+3.1)** | **−0.083** | 39% |
| lvl5train | 64 lvl5-only     | 35.4% | 42.7% (+7.3) | **36.5% (+1.0)** | **−0.153** | 39% |
| stratify  | — | (OOM, re-running) | | | | |
| train16   | — | (OOM, re-running) | | | | |

**The central K=1 finding (from the 2 valid cells):**
- **Good perturbations exist** — oracle-best single seed gains **+7.3pp** (35.4→42.7%),
  and 39% of seeds beat base on test. The *population* contains real winners.
- **But train reward cannot find them.** Spearman ρ(train, test) ≈ **0 to negative**
  (−0.08, −0.15). Selecting K=1 by train reward gets only +3.1 / +1.0pp — far below
  the +7.3 oracle. On a 64-problem train set, train reward is ~noise as a held-out
  predictor.
- **Training on level-5-only made transfer WORSE**, not better (ρ −0.15 vs −0.08,
  realistic gain +1.0 vs +3.1). Selecting on the hardest problems overfits harder.
- **Implication:** the K=1 bottleneck is *selection*, not *search*. This is exactly
  why the ensemble works (RECORDS.md: MATH ensemble +7.8pp) — majority-voting the
  top-k is robust to the broken train→test correlation that sinks any single pick.

Caveat: 2 cells, n=96, single σ — directional, not final. The re-run completes the
4-cell grid; the full axis grid (σ, noise, train-size 16/32/64, stratify) follows.

(appended as cells complete)

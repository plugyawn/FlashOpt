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

## Baselines (unperturbed, MATH-500 lvl4-5 test)

| model | test slice | base acc |
|-------|-----------|---------:|
| Qwen2.5-1.5B-Instruct | lvl4-5, n=192 (default split) | **53.1%** (102/192) |
| Qwen2.5-7B-Instruct   | lvl4-5, n=192 (default split) | ~60.4% (from prior 512-runs; 61.98% on a re-run — ±1.5pp run-to-run) |

(7B lvl5-only and stratified baselines: TODO as the sweep runs.)

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

### R0 — hotpath validation smoke (1.5B, pop=8)
`hp1b_smoke` · default lvl4-5 split · σ=5e-4 · rademacher · L4.
- base **53.1%** · best-K1 oracle 57.8% (+4.7) · K1-by-train 54.2% (+1.0) · ρ=0.29 · frac>base 50%.
- Purpose was to validate the K=1 instrument end-to-end on GPU (✓). n=8 is far too
  small to conclude anything — but already shows the central tension: the *oracle*
  best seed gains +4.7pp while the *train-selected* one gains only +1.0pp, i.e.
  train reward is a weak selector at this scale (ρ=0.29). The real sweep tests
  whether train-set design closes that oracle↔realistic gap.

(further runs appended as the sweep completes)

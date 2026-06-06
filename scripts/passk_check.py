#!/usr/bin/env python3
"""
Genuineness check for a "winning" RandOpt seed: is its greedy test-acc gain a real
model improvement, or selection noise (a lucky argmax on a few problems)?

Method: sample k completions per problem at temperature>0 for BOTH the base model
and the perturbed seed, on MATH-500 lvl4-5. Report, per model:
  - avg@1  : mean per-sample accuracy (UNBIASED estimate of the model's accuracy)
  - pass@k : fraction of problems solved by >=1 of k samples (capability ceiling)
  - maj@k  : majority-vote accuracy (self-consistency)

Crucially we split the problems into:
  - SELECTION slice: the exact problems the seed was chosen on -> gain here is
    selection-BIASED (upward). Reported but flagged.
  - FRESH slice: lvl4-5 problems NOT in the seed's train OR test set -> the seed
    never "saw" these for selection, so a gain here is GENUINE.

A seed whose avg@1 beats base on the FRESH slice is a real improvement; one that
only beats base on the SELECTION slice (and not fresh) was greedy/selection luck.

  modal run scripts/modal_smoke.py --tier passk   (driver wires model+seeds)
This module is the worker logic (pure-ish); it's imported and called by hotpath-
style harness code with engines provided.
"""
from __future__ import annotations
import json
import math
from typing import Dict, List


def avg_at_1(per_sample_correct: List[List[bool]]) -> float:
    """Mean over all (problem, sample) of correctness — unbiased model accuracy."""
    flat = [c for prob in per_sample_correct for c in prob]
    return 100.0 * sum(flat) / len(flat) if flat else float("nan")


def pass_at_k(per_sample_correct: List[List[bool]], k: int) -> float:
    """Fraction of problems with >=1 correct among the first k samples."""
    solved = sum(any(prob[:k]) for prob in per_sample_correct)
    return 100.0 * solved / len(per_sample_correct) if per_sample_correct else float("nan")


def maj_at_k(answers: List[List[str]], golds: List, handler, k: int) -> float:
    """Majority vote over k samples per problem."""
    from collections import Counter
    correct = 0
    for ans_list, gold in zip(answers, golds):
        votes = [a for a in ans_list[:k] if a]
        if not votes:
            continue
        top = Counter(votes).most_common(1)[0][0]
        ok = (handler.is_voted_answer_correct(top, gold)
              if hasattr(handler, "is_voted_answer_correct")
              else handler.is_answer_correct(handler.format_answer_for_check(top), gold))
        correct += int(bool(ok))
    return 100.0 * correct / len(answers) if answers else float("nan")


def wilson_ci(p_frac: float, n: int, z: float = 1.96):
    """95% Wilson interval for a proportion (p in [0,1]); returns (lo, hi) in %."""
    if n == 0:
        return (float("nan"), float("nan"))
    denom = 1 + z * z / n
    center = (p_frac + z * z / (2 * n)) / denom
    half = z * math.sqrt(p_frac * (1 - p_frac) / n + z * z / (4 * n * n)) / denom
    return (100 * (center - half), 100 * (center + half))


def summarize(tag: str, per_sample_correct, answers, golds, handler, k):
    a1 = avg_at_1(per_sample_correct)
    n_samples = sum(len(p) for p in per_sample_correct)
    lo, hi = wilson_ci(a1 / 100, n_samples)
    out = {"tag": tag, "n_problems": len(per_sample_correct), "k": k,
           "avg_at_1": a1, "avg_at_1_ci95": [lo, hi],
           "pass_at_k": pass_at_k(per_sample_correct, k),
           "maj_at_k": maj_at_k(answers, golds, handler, k)}
    return out

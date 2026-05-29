"""
Held-out FineWeb bits-per-byte (bpb) evaluation — a nanoGPT-speedrun-style
language-modeling metric, repurposed for the RandOpt speedrun.

WHY bits-per-byte (not raw cross-entropy)?  The nanoGPT speedrun reports raw
token cross-entropy because its tokenizer is fixed (GPT-2 BPE). The RandOpt
"profile" models have *different* tokenizers, so token-CE is not comparable
across models. **Bits-per-byte** normalizes by the UTF-8 byte length of the
text and is therefore tokenizer-invariant:

    bpb = ( sum of token NLL in nats ) / ( ln(2) * total UTF-8 bytes )

The metric is teacher-forced. We obtain per-token NLL from vLLM via
``SamplingParams(prompt_logprobs=...)`` — i.e. the logprob of each *actual*
prompt token given its prefix — so it reflects whatever (perturbed) weights are
live, with no custom forward code. See ``speedrun.py`` for the engine driving.

This module is intentionally free of ray/vllm imports: the data + tokenization +
bpb math are pure and unit-tested on CPU (tests/test_fineweb.py). Only
``build_heldout`` touches the network (``datasets``), lazily.

Protocol (documented so a record is reproducible):
* Fixed held-out slice of ``HuggingFaceFW/fineweb-edu`` ``sample-10BT`` written to
  ``eval/fineweb_heldout.jsonl`` with a recorded sha256.
* Each doc tokenized with the model's own tokenizer; a BOS is prepended if the
  tokenizer defines one (so the first real token is scored with context).
* Tokens split into non-overlapping chunks of ``max_len``; within a chunk the
  first token is unscored (no prefix). With long chunks this loss is ~1/max_len.
* bytes = total UTF-8 length of the held-out text (fixed, tokenizer-independent).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np

LN2 = math.log(2.0)

DEFAULT_HELDOUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fineweb_heldout.jsonl")
DEFAULT_DATASET = "HuggingFaceFW/fineweb-edu"
DEFAULT_CONFIG = "sample-10BT"


# --------------------------------------------------------------------------- #
# Data: build / load / hash a fixed held-out slice
# --------------------------------------------------------------------------- #
@dataclass
class HeldoutManifest:
    dataset: str
    config: str
    split: str
    offset: int
    num_docs: int
    min_chars: int
    total_bytes: int
    sha256: str
    path: str


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_heldout(
    out_path: str = DEFAULT_HELDOUT_PATH,
    dataset: str = DEFAULT_DATASET,
    config: str = DEFAULT_CONFIG,
    split: str = "train",
    offset: int = 1_000_000,
    num_docs: int = 256,
    min_chars: int = 1000,
    max_chars: int = 20_000,
) -> HeldoutManifest:
    """Materialize a fixed, reproducible held-out slice to ``out_path`` (jsonl of
    ``{"text": ...}``) and return its manifest. Requires ``datasets`` + network;
    run once (e.g. on the GPU pod). FineWeb is never used for selection, so any
    fixed slice is a valid held-out LM probe; we skip ``offset`` docs to avoid
    the head of the shard.
    """
    from datasets import load_dataset  # lazy: network + heavy dep
    from itertools import islice

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    ds = load_dataset(dataset, name=config, split=split, streaming=True)
    kept: List[str] = []
    for ex in islice(ds, offset, offset + num_docs * 8):  # over-scan, then filter
        txt = ex.get("text") or ""
        if min_chars <= len(txt) <= max_chars:
            kept.append(txt)
            if len(kept) >= num_docs:
                break
    if len(kept) < num_docs:
        raise RuntimeError(f"only found {len(kept)} docs in slice (wanted {num_docs}); widen the scan")

    total_bytes = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for txt in kept:
            total_bytes += len(txt.encode("utf-8"))
            f.write(json.dumps({"text": txt}, ensure_ascii=False) + "\n")

    man = HeldoutManifest(
        dataset=dataset, config=config, split=split, offset=offset, num_docs=len(kept),
        min_chars=min_chars, total_bytes=total_bytes, sha256=file_sha256(out_path), path=out_path,
    )
    with open(out_path + ".manifest.json", "w") as f:
        json.dump(asdict(man), f, indent=2)
    return man


def load_heldout(path: str = DEFAULT_HELDOUT_PATH) -> List[str]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line)["text"])
    return docs


def heldout_total_bytes(docs: Sequence[str]) -> int:
    return int(sum(len(d.encode("utf-8")) for d in docs))


# --------------------------------------------------------------------------- #
# Tokenization into chunks
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    token_ids: List[int]      # full chunk (length <= max_len); positions [1:] are scored
    doc_index: int


def tokenize_docs_to_chunks(tokenizer, docs: Sequence[str], max_len: int = 2048) -> List[Chunk]:
    """Tokenize each doc (no extra special tokens beyond an optional BOS) and
    split into non-overlapping chunks of ``max_len`` tokens. The model's BOS, if
    any, is prepended once per doc so the first real token has context."""
    bos = getattr(tokenizer, "bos_token_id", None)
    chunks: List[Chunk] = []
    for di, text in enumerate(docs):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if bos is not None:
            ids = [bos] + ids
        for start in range(0, len(ids), max_len):
            piece = ids[start:start + max_len]
            if len(piece) >= 2:   # need >=2 to score at least one token
                chunks.append(Chunk(token_ids=piece, doc_index=di))
    return chunks


# --------------------------------------------------------------------------- #
# NLL extraction from a vLLM RequestOutput (pure; tested with fakes)
# --------------------------------------------------------------------------- #
def extract_actual_token_logprobs(request_output) -> List[float]:
    """From one vLLM RequestOutput (produced with prompt_logprobs>=1 and a
    prompt of token ids), return the natural-log prob of each *actual* prompt
    token at positions >= 1 (position 0 has no prefix -> skipped).

    ``request_output.prompt_token_ids[i]`` is the actual token; its logprob is
    ``request_output.prompt_logprobs[i][actual_id].logprob``.
    """
    tok_ids = list(request_output.prompt_token_ids)
    plps = request_output.prompt_logprobs
    out: List[float] = []
    for i in range(1, len(tok_ids)):
        entry = plps[i]
        if not entry:
            continue
        actual = tok_ids[i]
        lp = entry.get(actual)
        if lp is None:
            # actual token fell outside the returned set; approximate with the
            # smallest returned logprob (rare with prompt_logprobs>=1).
            lp_val = min(v.logprob for v in entry.values())
        else:
            lp_val = lp.logprob if hasattr(lp, "logprob") else float(lp)
        out.append(float(lp_val))
    return out


# --------------------------------------------------------------------------- #
# bpb math (pure)
# --------------------------------------------------------------------------- #
@dataclass
class BpbResult:
    bpb: float
    nats_per_token: float
    n_scored_tokens: int
    n_bytes: int

    def as_dict(self) -> Dict:
        return asdict(self)


def bpb_from_token_logprobs(per_chunk_logprobs: Sequence[Sequence[float]], total_bytes: int) -> BpbResult:
    """Single-model bpb from per-chunk actual-token logprobs (nats)."""
    total_nll = 0.0
    n_tok = 0
    for lps in per_chunk_logprobs:
        for lp in lps:
            total_nll += -lp
            n_tok += 1
    bpb = total_nll / (LN2 * total_bytes) if total_bytes else float("nan")
    npt = total_nll / n_tok if n_tok else float("nan")
    return BpbResult(bpb=bpb, nats_per_token=npt, n_scored_tokens=n_tok, n_bytes=int(total_bytes))


def ensemble_nll_per_position(model_logprobs: Sequence[Sequence[float]]) -> List[float]:
    """Combine K models' per-position actual-token logprobs into the ensemble
    NLL (nats) of the **probability-averaged** next-token distribution:

        p_ens(t) = mean_m p_m(t)   ->   nll = -log p_ens = log K - logsumexp_m lp_m

    All models must share the same positions (same chunks, same token ids).
    """
    arr = np.asarray(model_logprobs, dtype=np.float64)   # (K, P)
    if arr.ndim != 2:
        raise ValueError("model_logprobs must be a (K, P) rectangular array")
    K = arr.shape[0]
    m = arr.max(axis=0)
    lse = m + np.log(np.exp(arr - m).sum(axis=0))         # logsumexp over models
    nll = math.log(K) - lse                               # = -log(mean prob)
    return nll.tolist()


def bpb_from_ensemble_logprobs(
    per_model_per_chunk_logprobs: Sequence[Sequence[Sequence[float]]],
    total_bytes: int,
) -> BpbResult:
    """Ensemble bpb. Input indexed [model][chunk][position]; every model must
    have identical chunk/position structure."""
    n_models = len(per_model_per_chunk_logprobs)
    if n_models == 0:
        raise ValueError("no models")
    n_chunks = len(per_model_per_chunk_logprobs[0])
    total_nll = 0.0
    n_tok = 0
    for c in range(n_chunks):
        stacked = [per_model_per_chunk_logprobs[m][c] for m in range(n_models)]
        nll = ensemble_nll_per_position(stacked)
        total_nll += float(sum(nll))
        n_tok += len(nll)
    bpb = total_nll / (LN2 * total_bytes) if total_bytes else float("nan")
    npt = total_nll / n_tok if n_tok else float("nan")
    return BpbResult(bpb=bpb, nats_per_token=npt, n_scored_tokens=n_tok, n_bytes=int(total_bytes))


class EnsembleBpbAccumulator:
    """Streaming ensemble bpb: feed one model's per-chunk logprobs at a time and
    keep only a running sum of probabilities per position (O(total_tokens) memory
    regardless of K). Equivalent to ``bpb_from_ensemble_logprobs`` but never
    stores all K models at once — important for the 72B standard with large K.
    """

    def __init__(self, n_chunks: int):
        self._prob_sum: List[Optional[np.ndarray]] = [None] * n_chunks
        self.n_models = 0

    def add_model(self, per_chunk_logprobs: Sequence[Sequence[float]]) -> None:
        if len(per_chunk_logprobs) != len(self._prob_sum):
            raise ValueError("chunk count mismatch across models")
        self.n_models += 1
        for c, lps in enumerate(per_chunk_logprobs):
            p = np.exp(np.asarray(lps, dtype=np.float64))
            self._prob_sum[c] = p if self._prob_sum[c] is None else self._prob_sum[c] + p

    def result(self, total_bytes: int) -> BpbResult:
        if self.n_models == 0:
            raise ValueError("no models added")
        total_nll = 0.0
        n_tok = 0
        for ps in self._prob_sum:
            if ps is None:
                continue
            mean = ps / self.n_models
            total_nll += float(-np.log(mean).sum())
            n_tok += int(ps.shape[0])
        bpb = total_nll / (LN2 * total_bytes) if total_bytes else float("nan")
        npt = total_nll / n_tok if n_tok else float("nan")
        return BpbResult(bpb=bpb, nats_per_token=npt, n_scored_tokens=n_tok, n_bytes=int(total_bytes))

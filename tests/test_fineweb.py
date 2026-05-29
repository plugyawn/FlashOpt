"""
CPU tests for the FineWeb bpb math (eval/fineweb.py) using fake vLLM outputs and
a fake tokenizer. No GPU/network.

Run:  .venv/bin/python tests/test_fineweb.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import fineweb as F


# ---- fakes ---------------------------------------------------------------- #
class _LP:
    def __init__(self, lp): self.logprob = lp


class _RO:
    def __init__(self, ids, plps):
        self.prompt_token_ids = ids
        self.prompt_logprobs = plps


class _FakeTok:
    bos_token_id = 1
    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 97 + 2 for c in text]   # 1 token/char, ids >= 2


# ---- extraction ----------------------------------------------------------- #
def test_extract_actual_token_logprobs():
    ids = [10, 11, 12, 13]
    plps = [None, {11: _LP(-0.5)}, {12: _LP(-1.0), 99: _LP(-3.0)}, {13: _LP(-2.0)}]
    got = F.extract_actual_token_logprobs(_RO(ids, plps))
    assert got == [-0.5, -1.0, -2.0], got


def test_extract_handles_missing_actual():
    ids = [10, 11]
    plps = [None, {99: _LP(-4.0), 98: _LP(-5.0)}]   # actual id 11 absent
    got = F.extract_actual_token_logprobs(_RO(ids, plps))
    assert got == [-5.0], got   # falls back to smallest logprob


# ---- single-model bpb ----------------------------------------------------- #
def test_bpb_from_token_logprobs():
    # two chunks; total NLL = 0.5+1.0 + 2.0 = 3.5 nats over 3 tokens, 7 bytes
    per_chunk = [[-0.5, -1.0], [-2.0]]
    r = F.bpb_from_token_logprobs(per_chunk, total_bytes=7)
    assert r.n_scored_tokens == 3
    assert abs(r.nats_per_token - 3.5 / 3) < 1e-12
    assert abs(r.bpb - 3.5 / (math.log(2) * 7)) < 1e-12


# ---- ensemble combine ----------------------------------------------------- #
def test_ensemble_nll_prob_average():
    # models with probs 0.2 and 0.8 -> mean 0.5 -> nll = ln2
    lp = [[math.log(0.2)], [math.log(0.8)]]
    nll = F.ensemble_nll_per_position(lp)
    assert abs(nll[0] - math.log(2)) < 1e-12

    # probs 0.9 and 0.1 -> mean 0.5 -> nll = ln2 (averaging disagreeing models)
    lp2 = [[math.log(0.9)], [math.log(0.1)]]
    assert abs(F.ensemble_nll_per_position(lp2)[0] - math.log(2)) < 1e-12


def test_ensemble_of_one_equals_single():
    per_chunk = [[-0.5, -1.0], [-2.0]]
    single = F.bpb_from_token_logprobs(per_chunk, total_bytes=9)
    ens = F.bpb_from_ensemble_logprobs([per_chunk], total_bytes=9)   # K=1
    assert abs(single.bpb - ens.bpb) < 1e-12
    assert single.n_scored_tokens == ens.n_scored_tokens


def test_ensemble_of_identical_equals_single():
    per_chunk = [[-0.3, -0.7, -1.2]]
    single = F.bpb_from_token_logprobs(per_chunk, total_bytes=12)
    ens = F.bpb_from_ensemble_logprobs([per_chunk, per_chunk, per_chunk], total_bytes=12)  # K=3 identical
    assert abs(single.bpb - ens.bpb) < 1e-12   # mean of identical probs = same prob


def test_streaming_accumulator_matches_batch():
    import math as _m
    m0 = [[-0.5, -1.0], [-2.0]]
    m1 = [[-0.7, -0.3], [-1.5]]
    m2 = [[-1.1, -0.9], [-0.8]]
    batch = F.bpb_from_ensemble_logprobs([m0, m1, m2], total_bytes=20)
    acc = F.EnsembleBpbAccumulator(n_chunks=2)
    for m in (m0, m1, m2):
        acc.add_model(m)
    streamed = acc.result(total_bytes=20)
    assert abs(batch.bpb - streamed.bpb) < 1e-12
    assert batch.n_scored_tokens == streamed.n_scored_tokens == 3


# ---- tokenization --------------------------------------------------------- #
def test_tokenize_docs_to_chunks():
    tok = _FakeTok()
    docs = ["abcdefghij", "xy"]   # 10 chars, 2 chars
    chunks = F.tokenize_docs_to_chunks(tok, docs, max_len=4)
    # doc0: bos + 10 tokens = 11 ids -> chunks of 4: [4,4,3]
    # doc1: bos + 2 = 3 ids -> one chunk of 3
    doc0 = [c for c in chunks if c.doc_index == 0]
    doc1 = [c for c in chunks if c.doc_index == 1]
    assert [len(c.token_ids) for c in doc0] == [4, 4, 3]
    assert [len(c.token_ids) for c in doc1] == [3]
    assert all(len(c.token_ids) >= 2 for c in chunks)
    # bos prepended
    assert doc0[0].token_ids[0] == tok.bos_token_id


def test_heldout_total_bytes():
    docs = ["héllo", "abc"]   # é is 2 UTF-8 bytes -> 6 + 3 = 9
    assert F.heldout_total_bytes(docs) == 9


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t(); print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa
            failed += 1
            import traceback; print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}"); traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

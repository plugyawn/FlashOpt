"""
CPU tests for speedrun.py pure helpers (config / top-k / record / RECORDS.md).
No GPU. Run:  .venv/bin/python tests/test_speedrun.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import speedrun as S


def test_load_configs():
    for name in ["configs/standard_8xh100_qwen72b.yaml", "configs/smoke_1gpu_small.yaml"]:
        cfg = S.load_config(os.path.join(S.REPO, name))
        assert cfg["model"] and cfg["population_size"] and cfg["sigma_values"]
        assert cfg["_sha256"] and cfg["dataset"]


def test_derive_topk():
    assert S.derive_topk(16, [0.25]) == ([4], 4)
    tk, mx = S.derive_topk(512, [0.04, 0.1])
    assert tk == [51, 20] and mx == 51       # int(.1*512)=51, int(.04*512)=20, desc


def test_detect_hardware_cpu():
    hw = S.detect_hardware()
    assert "gpu" in hw and "gpu_count" in hw   # cpu here


def _fake_record():
    cfg = {"name": "smoke", "_sha256": "abc123", "model": "org/Tiny-1.5B-Instruct",
           "precision": "bfloat16", "tensor_parallel_size": 1, "num_engines": 1,
           "enforce_eager": True, "noise": "rademacher", "kernel": "auto",
           "dataset": "gsm8k", "train_samples": 32, "test_samples": 32,
           "population_size": 16, "sigma_values": [0.001, 0.002]}
    return S.build_record(
        cfg,
        hardware={"gpu": "A100-40GB", "gpu_count": 1},
        timings={"launch": 5.0, "sampling": 8.0, "total": 30.0},
        seeds_per_sec=2.0,
        base_train_reward=0.5, base_test_acc=0.40,
        ensemble_results={4: {"accuracy": 53.0, "correct": 17}, 2: {"accuracy": 50.0, "correct": 16}},
        best_sigma=0.001, top_k_perturbs=[(123, 0.001), (456, 0.002)],
        fineweb_base={"bpb": 0.95, "nats_per_token": 2.1, "n_scored_tokens": 100, "n_bytes": 400},
        fineweb_ensemble={"bpb": 0.90, "nats_per_token": 2.0, "n_scored_tokens": 100, "n_bytes": 400},
        fineweb_manifest_sha="deadbeef")


def test_build_record_fields():
    r = _fake_record()
    assert r["seeds_per_sec"] == 2.0
    assert r["best_ensemble_accuracy"] == 53.0           # max over k
    assert r["ensemble_accuracy"] == {"4": 53.0, "2": 50.0}
    assert r["fineweb"]["base"]["bpb"] == 0.95
    assert r["top_k_perturbs"] == [[123, 0.001], [456, 0.002]]
    assert r["git_commit"]                                 # some string


def test_write_record_and_table():
    r = _fake_record()
    with tempfile.TemporaryDirectory() as d:
        run_dir = os.path.join(d, "run1")
        records_md = os.path.join(d, "RECORDS.md")
        rec_path = S.write_record(r, run_dir, records_md=records_md)
        assert os.path.exists(rec_path)
        loaded = json.load(open(rec_path))
        assert loaded["model"] == r["model"]
        md = open(records_md).read()
        assert "| date | commit |" in md                  # header
        assert "Tiny-1.5B-Instruct" in md                 # short model name
        assert "1xA100-40GB" in md                         # hardware string
        assert "0.95" in md and "0.90" in md               # bpb columns
        # appending a second run keeps one header, two rows
        S.write_record(r, os.path.join(d, "run2"), records_md=records_md)
        md2 = open(records_md).read()
        assert md2.count("| date | commit |") == 1
        assert md2.count("Tiny-1.5B-Instruct") == 2


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

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
    for name in ["configs/standard_8xh100_qwen72b.yaml", "configs/smoke_1gpu_small.yaml",
                 "configs/tuned_1xh100.yaml"]:
        cfg = S.load_config(os.path.join(S.REPO, name))
        assert cfg["model"] and cfg["population_size"] and cfg["sigma_values"]
        assert cfg["_sha256"] and cfg["dataset"]


def test_tuned_config_throughput_knobs():
    cfg = S.load_config(os.path.join(S.REPO, "configs/tuned_1xh100.yaml"))
    assert cfg["enforce_eager"] is False          # CUDA graphs on
    assert cfg["max_num_seqs"] >= 256             # wide continuous batch
    assert cfg.get("max_model_len")               # capped seq len
    assert "probe_top" in cfg                      # probe knob present


# ---- token accounting (fakes mimicking vLLM RequestOutput) ---------------- #
class _Comp:
    def __init__(self, n_tok): self.token_ids = list(range(n_tok))


class _RO:
    def __init__(self, n_prompt, n_gen):
        self.prompt_token_ids = list(range(n_prompt))
        self.outputs = [_Comp(n_gen)]


def test_fineweb_max_len_clamp():
    # bpb pass requests 1 output token, so chunk must be < max_model_len.
    assert S._fineweb_max_len(2048, 2048) == 2046     # clamped to mml-2
    assert S._fineweb_max_len(1024, 2048) == 1024     # already smaller, unchanged
    assert S._fineweb_max_len(2048, None) == 2048     # no engine cap -> unchanged
    assert S._fineweb_max_len(4096, 1280) == 1278     # clamp to the engine limit


def test_count_tokens():
    outs = [_RO(10, 5), _RO(8, 7), _RO(12, 0)]
    gen, prompt, n = S.count_tokens(outs)
    assert gen == 12 and prompt == 30 and n == 3


def test_token_meter_rates_and_summary():
    m = S.TokenMeter()
    # base_eval: one generate of 2 prompts (5 gen tok each) in 2s
    m.add("base_eval", [[_RO(10, 5), _RO(10, 5)]], seconds=2.0)
    # sampling: two generates (one per seed), 3 prompts x 4 gen tok, in 8s total
    m.add("sampling", [[_RO(6, 4)] * 3, [_RO(6, 4)] * 3], seconds=8.0)
    s = m.summary()
    assert s["gen_tokens"] == (2 * 5) + (6 * 4)      # 10 + 24 = 34
    assert s["prompts"] == 2 + 6                       # 8
    assert abs(s["gen_seconds"] - 10.0) < 1e-9        # 2 + 8
    assert abs(s["gen_tokens_per_sec"] - 34 / 10.0) < 1e-9
    assert abs(s["prompts_per_sec"] - 8 / 10.0) < 1e-9
    # per-phase sampling rate uses that phase's own seconds
    sp = s["per_phase"]["sampling"]
    assert sp["gen_tokens"] == 24 and abs(sp["gen_tok_per_sec"] - 24 / 8.0) < 1e-9


# ---- single-model accuracy (probe path) with a fake handler --------------- #
class _FakeHandler:
    name = "gsm8k"
    def extract_answer_for_voting(self, text): return text.strip()
    def is_voted_answer_correct(self, ans, gt): return str(ans) == str(gt)


class _Out:
    def __init__(self, text):
        self.outputs = [type("o", (), {"text": text})()]


def test_single_model_accuracy():
    handler = _FakeHandler()
    outputs = [_Out("42"), _Out("7"), _Out(""), _Out("13")]
    datas = [{"ground_truth": "42"}, {"ground_truth": "8"},
             {"ground_truth": "1"}, {"ground_truth": "13"}]
    # "42"==42 ✓, "7"!=8 ✗, ""->skip, "13"==13 ✓  -> 2 correct
    assert S._single_model_accuracy(handler, outputs, datas) == 2


def test_on_outputs_wired_in_randopt_source():
    # The throughput meter relies on randopt.run_* accepting on_outputs AND
    # actually invoking it. randopt imports ray/vllm (absent on this CPU box), so
    # we verify via AST instead of importing: each of the 3 functions must take
    # an `on_outputs` param and contain a call to it.
    import ast
    src = open(os.path.join(S.REPO, "randopt.py")).read()
    tree = ast.parse(src)
    targets = {"evaluate_base_model", "run_sampling", "run_ensemble_evaluation"}
    seen = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in targets:
            params = {a.arg for a in node.args.args} | {a.arg for a in node.args.kwonlyargs}
            assert "on_outputs" in params, f"{node.name} missing on_outputs param"
            # must actually call on_outputs(...) somewhere in the body
            calls_it = any(
                isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == "on_outputs"
                for c in ast.walk(node))
            assert calls_it, f"{node.name} never calls on_outputs()"
            seen[node.name] = True
    assert seen.keys() == targets, f"missing functions: {targets - seen.keys()}"


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
        fineweb_manifest_sha="deadbeef",
        throughput={"gen_tokens_per_sec": 1234.5, "prompts_per_sec": 6.1, "gen_tokens": 9999,
                    "prompts": 50, "gen_seconds": 8.1, "total_tokens_per_sec": 2000.0, "per_phase": {}},
        probes=[{"rank": 1, "seed": 123, "sigma": 0.001, "test_accuracy": 45.0,
                 "fineweb_bpb": 0.93}])


def test_build_record_fields():
    r = _fake_record()
    assert r["seeds_per_sec"] == 2.0
    assert r["best_ensemble_accuracy"] == 53.0           # max over k
    assert r["ensemble_accuracy"] == {"4": 53.0, "2": 50.0}
    assert r["fineweb"]["base"]["bpb"] == 0.95
    assert r["top_k_perturbs"] == [[123, 0.001], [456, 0.002]]
    assert r["git_commit"]                                 # some string
    assert r["throughput"]["gen_tokens_per_sec"] == 1234.5
    assert r["throughput"]["prompts_per_sec"] == 6.1
    assert r["probes"][0]["seed"] == 123 and r["probes"][0]["fineweb_bpb"] == 0.93


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
        assert "gen-tok/s" in md and "prompts/s" in md     # new throughput cols
        assert "Tiny-1.5B-Instruct" in md                 # short model name
        assert "1xA100-40GB" in md                         # hardware string
        assert "0.95" in md and "0.90" in md               # bpb columns
        assert "1234" in md and "6.10" in md               # throughput values
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

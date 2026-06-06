#!/usr/bin/env python3
"""
GPU runner for the pass@k genuineness check (scripts/passk_check.py is the pure
math). Evaluates base + a list of (seed, sigma) perturbations with k samples per
problem at temp>0, on a SELECTION slice (problems the seed was chosen on, biased)
and a FRESH slice (lvl4-5 problems the seed never saw — the honest test).

  python passk_run.py --model Qwen/Qwen2.5-1.5B-Instruct --k 24 \
      --seeds 1068999192:0.0005 --sel data/m/test.jsonl --fresh data/m/fresh.jsonl
"""
import argparse, json, os, sys, time
from typing import Dict, List, Tuple

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
import scripts.passk_check as pk


def _load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def _eval_current_weights(eng, handler, datas, prompts, sp, k, ray):
    """Generate k samples for each prompt; return (per_sample_correct, answers)."""
    # vLLM: n=k samples per prompt in one call
    outs = ray.get(eng.generate.remote(prompts, sp, use_tqdm=False))
    per_correct, answers = [], []
    for o, d in zip(outs, datas):
        gold = d["ground_truth"]
        corrects, ans = [], []
        for comp in o.outputs[:k]:
            txt = comp.text
            a = (handler.extract_answer_for_voting(txt) if hasattr(handler, "extract_answer_for_voting")
                 else handler.extract_answer(txt)) or ""
            ans.append(a)
            ok = False
            if a:
                ok = (handler.is_voted_answer_correct(a, gold)
                      if hasattr(handler, "is_voted_answer_correct")
                      else handler.is_answer_correct(handler.format_answer_for_check(a), gold))
            corrects.append(bool(ok))
        per_correct.append(corrects)
        answers.append(ans)
    return per_correct, answers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--seeds", nargs="+", required=True, help="seed:sigma pairs")
    ap.add_argument("--sel", required=True, help="selection-slice jsonl (biased)")
    ap.add_argument("--fresh", required=True, help="fresh-slice jsonl (honest)")
    ap.add_argument("--out", default="passk-runs/out.json")
    ap.add_argument("--gpu-mem-util", type=float, default=0.55)
    ap.add_argument("--max-model-len", type=int, default=2048)
    a = ap.parse_args()

    import ray
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    import randopt  # noqa
    from core import launch_engines, cleanup_engines
    from data_handlers import get_dataset_handler

    handler = get_dataset_handler("math500")
    sel = handler.load_data(a.sel, max_samples=None)
    fresh = handler.load_data(a.fresh, max_samples=None)
    tok = AutoTokenizer.from_pretrained(a.model)

    def fmt(messages):
        if tok.chat_template:
            return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return "\n".join(m["content"] for m in messages) + "\n"
    sel_p = [fmt(d["messages"]) for d in sel]
    fresh_p = [fmt(d["messages"]) for d in fresh]

    _rt = {"env_vars": {"PYTHONPATH": REPO + os.pathsep + os.environ.get("PYTHONPATH", "")}}
    ray.init(address="local", ignore_reinit_error=True, runtime_env=_rt)
    engines, pgs = launch_engines(1, a.model, precision="bfloat16", enforce_eager=True,
                                  gpu_memory_utilization=a.gpu_mem_util,
                                  max_num_seqs=256, max_model_len=a.max_model_len)
    eng = engines[0]
    # n=k sampling at temp>0; fixed seed for reproducibility of the SAMPLING (not the model)
    sp = SamplingParams(n=a.k, temperature=a.temperature, top_p=0.95,
                        max_tokens=a.max_tokens, seed=1234)

    results = {"model": a.model, "k": a.k, "temperature": a.temperature,
               "n_sel": len(sel), "n_fresh": len(fresh), "rows": []}
    try:
        def run_label(label, seed, sigma):
            if seed is None:
                ray.get(eng.collective_rpc.remote("reset_to_base_weights", args=()))
            else:
                ray.get(eng.collective_rpc.remote("perturb_self_weights", args=(int(seed), float(sigma), False)))
            for slice_name, datas, prompts in [("selection", sel, sel_p), ("fresh", fresh, fresh_p)]:
                pc, ans = _eval_current_weights(eng, handler, datas, prompts, sp, a.k, ray)
                golds = [d["ground_truth"] for d in datas]
                s = pk.summarize(f"{label}/{slice_name}", pc, ans, golds, handler, a.k)
                s.update({"label": label, "slice": slice_name, "seed": seed, "sigma": sigma})
                results["rows"].append(s)
                print(f"[{label}/{slice_name}] avg@1={s['avg_at_1']:.1f}% "
                      f"(95%CI {s['avg_at_1_ci95'][0]:.1f}-{s['avg_at_1_ci95'][1]:.1f}) "
                      f"pass@{a.k}={s['pass_at_k']:.1f}% maj@{a.k}={s['maj_at_k']:.1f}%", flush=True)

        run_label("base", None, None)
        for sp_pair in a.seeds:
            seed, sigma = sp_pair.split(":")
            run_label(f"seed{seed}", int(seed), float(sigma))
    finally:
        cleanup_engines(engines, pgs)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

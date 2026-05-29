import argparse
import json
import os
import time
from collections import defaultdict
from datetime import datetime

import pandas as pd
import ray
from transformers import AutoTokenizer
from vllm import SamplingParams

from data_handlers import get_dataset_handler, list_datasets
from core import launch_engines, cleanup_engines


def parse_args():
    parser = argparse.ArgumentParser(description="Distillation Pipeline: Generate and Filter")

    parser.add_argument("--dataset", type=str, default="gsm8k", choices=list_datasets())
    parser.add_argument("--data_path", type=str, default=None, help="Data path (default: handler's train path)")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--seeds_file", type=str, required=True, help="Path to top_k_seeds.json from randopt run")
    parser.add_argument("--precision", type=str, default="bfloat16")
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--num_engines", type=int, default=4)
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--cuda_devices", type=str, default="0,1,2,3")
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--all_traces", action="store_true", 
                        help="Keep ALL correct traces (not just shortest)")
    parser.add_argument("--no_format_check", action="store_true", 
                        help="Skip #### format check (not recommended for GSM8K)")
    parser.add_argument("--output_dir", type=str, default="distill_data")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name for output dir (default: dataset_timestamp)")
    parser.add_argument("--reuse_raw", type=str, default=None,
                        help="Path to existing output dir with raw_outputs.jsonl to skip generation")
    parser.add_argument("--force_regenerate", action="store_true",
                        help="Force regeneration even if raw outputs exist")
    
    return parser.parse_args()


def load_data_from_parquet(data_path, max_samples=None):
    """Load data from parquet file."""
    print(f"\n[Data] Loading from: {data_path}")
    
    df = pd.read_parquet(data_path)
    print(f"  Total rows in parquet: {len(df)}")
    
    task_datas = []
    for row in df.to_dict("records"):
        messages = row["prompt"]
        if hasattr(messages, 'tolist'):
            messages = messages.tolist()
        
        reward_model = row.get("reward_model", {})
        if isinstance(reward_model, dict):
            ground_truth = str(reward_model.get("ground_truth", ""))
        else:
            ground_truth = ""
        
        task_datas.append({
            "messages": messages,
            "ground_truth": ground_truth,
        })
        
        if max_samples and len(task_datas) >= max_samples:
            break
    
    print(f"  Loaded {len(task_datas)} samples")
    
    if task_datas:
        sample = task_datas[0]
        user_msg = next((m["content"] for m in sample["messages"] if m.get("role") == "user"), "")
        print(f"  Sample question: {user_msg[:100]}...")
        print(f"  Sample ground truth: {sample['ground_truth']}")
    
    return task_datas


def load_raw_outputs(raw_output_path):
    """Load existing raw outputs from JSONL file."""
    print(f"\n[Load] Loading existing raw outputs from: {raw_output_path}")
    
    sample_responses = defaultdict(list)
    total_records = 0
    
    with open(raw_output_path) as f:
        for line in f:
            record = json.loads(line)
            sample_idx = record["sample_idx"]
            response = record["response"]
            model_info = {
                "model_idx": record.get("model_idx"),
                "seed": record.get("seed"),
                "sigma": record.get("sigma"),
            }
            sample_responses[sample_idx].append((response, model_info))
            total_records += 1
    
    num_samples = len(sample_responses)
    avg_per_sample = total_records / num_samples if num_samples else 0.0
    print(f"  ✓ Loaded {total_records} records for {num_samples} samples")
    print(f"  Average responses per sample: {avg_per_sample:.1f}")
    
    return sample_responses


def check_existing_outputs(output_dir):
    """Check if output directory has existing raw outputs."""
    raw_path = f"{output_dir}/raw_outputs.jsonl"
    meta_path = f"{output_dir}/meta.json"
    prompts_path = f"{output_dir}/prompts.jsonl"
    
    if not (os.path.exists(raw_path) and os.path.exists(meta_path) and os.path.exists(prompts_path)):
        return False

    with open(raw_path) as f:
        return bool(f.readline())


def filter_correct_responses(responses, ground_truth, handler, require_format_marker=True, all_traces=False):
    """Filter responses where extracted answer matches ground truth.
    
    Args:
        responses: List of (response_text, model_info) tuples
        ground_truth: Ground truth answer string
        handler: Dataset handler for answer extraction and comparison
        require_format_marker: If True, require "####" in response
        all_traces: If True, return all correct traces; if False, return only shortest
    
    Returns:
        List of correct response dicts, or empty list if none correct
    """
    correct_traces = []
    
    for resp_text, model_info in responses:
        if require_format_marker and "####" not in resp_text:
            continue

        extracted_answer = handler.extract_answer(resp_text)
        if not extracted_answer:
            continue

        if handler.compute_reward(resp_text, ground_truth) > 0:
            correct_traces.append({
                "response": resp_text,
                "extracted_answer": extracted_answer,
                "response_length": len(resp_text),
                **model_info
            })
    
    if not correct_traces:
        return []
    
    if all_traces:
        return correct_traces
    else:
        correct_traces.sort(key=lambda x: x["response_length"])
        return [correct_traces[0]]


def save_sft_dataset(sft_data, output_dir, dataset_name):
    """Save in verl-compatible parquet format."""
    print(f"\n[Save] Saving SFT dataset to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    verl_data = []
    for item in sft_data:
        messages = item["messages"]
        if isinstance(messages, list):
            user_content = next((m["content"] for m in messages if m.get("role") == "user"), str(messages))
        else:
            user_content = str(messages)
        
        extra_info = {
            "sample_idx": item["sample_idx"],
            "ground_truth": item["ground_truth"],
        }
        for key in ["model_idx", "seed", "sigma", "extracted_answer"]:
            if key in item and item[key] is not None:
                extra_info[key] = item[key]
        
        verl_data.append({
            "data_source": f"distill_{dataset_name}",
            "prompt": user_content,
            "response": item["response"],
            "extra_info": extra_info
        })

    df = pd.DataFrame(verl_data)
    parquet_path = f"{output_dir}/train.parquet"
    df.to_parquet(parquet_path)
    print(f"  ✓ Parquet: {parquet_path} ({len(df)} rows)")

    jsonl_path = f"{output_dir}/train.jsonl"
    with open(jsonl_path, 'w') as f:
        for item in sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  ✓ JSONL: {jsonl_path}")
    
    return parquet_path, jsonl_path


def generate_responses(args, datas, prompts, top_k_perturbs, base_model_path, output_dir, handler):
    """Generate responses from all top-k models."""
    
    sampling_params = SamplingParams(
        temperature=0.0, 
        seed=args.global_seed, 
        max_tokens=args.max_tokens or handler.default_max_tokens
    )
    
    if os.environ.get("RAY_ADDRESS"):
        ray.init(address="auto", ignore_reinit_error=True)
    else:
        ray.init(address="local", ignore_reinit_error=True)

    print(f"\n[Generate] Launching {args.num_engines} vLLM engines")
    engines, pgs = launch_engines(args.num_engines, base_model_path, 
                                   precision=args.precision, tensor_parallel_size=args.tp)
    
    sample_responses = defaultdict(list)
    
    try:
        num_models = len(top_k_perturbs)
        total_batches = (num_models + args.num_engines - 1) // args.num_engines
        
        print(f"\n[Generate] Generating responses from {num_models} models")
        print(f"  Total batches: {total_batches}")
        print(f"  Total samples: {len(prompts)}")
        print(f"  Expected total responses: {num_models * len(prompts)}")

        raw_output_path = f"{output_dir}/raw_outputs.jsonl"
        raw_output_file = open(raw_output_path, 'w')
        
        total_start = time.time()
        
        for batch_idx in range(total_batches):
            start = batch_idx * args.num_engines
            batch_perturbs = top_k_perturbs[start:start + args.num_engines]
            
            t0 = time.time()

            ray.get([engines[i].collective_rpc.remote("perturb_self_weights", args=(int(s), sig, False)) 
                     for i, (s, sig) in enumerate(batch_perturbs)])

            batch_outputs = ray.get([engines[i].generate.remote(prompts, sampling_params, use_tqdm=False) 
                                     for i in range(len(batch_perturbs))])

            ray.get([engines[i].collective_rpc.remote("restore_self_weights", args=(int(s), sig, False)) 
                     for i, (s, sig) in enumerate(batch_perturbs)])

            for local_idx, (seed, sigma) in enumerate(batch_perturbs):
                model_idx = start + local_idx
                outputs = batch_outputs[local_idx]
                
                for sample_idx, output in enumerate(outputs):
                    response = output.outputs[0].text
                    model_info = {
                        "model_idx": model_idx,
                        "seed": int(seed),
                        "sigma": float(sigma),
                    }
                    sample_responses[sample_idx].append((response, model_info))

                    record = {
                        "sample_idx": sample_idx,
                        "model_idx": model_idx,
                        "seed": int(seed),
                        "sigma": float(sigma),
                        "response": response,
                        "ground_truth": datas[sample_idx]["ground_truth"],
                    }
                    raw_output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            raw_output_file.flush()
            
            elapsed = time.time() - t0
            progress = (batch_idx + 1) / total_batches * 100
            print(f"  Batch {batch_idx + 1}/{total_batches} done ({elapsed:.1f}s) [{progress:.1f}%]")
        
        raw_output_file.close()
        
        total_gen_time = time.time() - total_start
        print(f"\n  ✓ Generation complete in {total_gen_time:.1f}s")
        print(f"  ✓ Raw outputs saved: {raw_output_path}")
        
    finally:
        cleanup_engines(engines, pgs)
    
    return sample_responses


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    
    print("=" * 70)
    print("Distillation Pipeline: Generate + Filter (Correctness-based Selection)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Data path: {args.data_path or 'default'}")
    print(f"  Model: {args.model_name}")
    print(f"  Seeds file: {args.seeds_file}")
    print(f"  Num engines: {args.num_engines}")
    print(f"  All traces mode: {args.all_traces}")
    print(f"  Require #### format: {not args.no_format_check}")
    print(f"  Reuse raw from: {args.reuse_raw or 'None'}")
    print(f"  Force regenerate: {args.force_regenerate}")

    handler = get_dataset_handler(args.dataset)
    max_tokens = args.max_tokens or handler.default_max_tokens
    print(f"  Max tokens: {max_tokens}")

    print(f"\n[Step 1] Loading seeds from: {args.seeds_file}")
    with open(args.seeds_file) as f:
        seeds_info = json.load(f)
    base_model_path = seeds_info.get("base_model_path", args.model_name)
    top_k_perturbs = [(m["seed"], m["sigma"]) for m in seeds_info["top_k_models"]]
    print(f"  ✓ Loaded {len(top_k_perturbs)} perturbations")
    print(f"  Base model: {base_model_path}")

    if args.reuse_raw:
        output_dir = args.reuse_raw
        print(f"\n[Step 2] Reusing existing output directory: {output_dir}")
    else:
        if args.run_name:
            run_name = args.run_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{args.dataset}_{timestamp}"
        output_dir = f"{args.output_dir}/{run_name}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n[Step 2] Output directory: {output_dir}")

    data_path = args.data_path or handler.default_train_path
    datas = load_data_from_parquet(data_path, max_samples=args.max_samples)

    existing_output_dir = args.reuse_raw or output_dir
    skip_generation = (
        not args.force_regenerate and check_existing_outputs(existing_output_dir)
    )
    if skip_generation:
        if args.reuse_raw:
            print(f"\n[Info] Found existing raw outputs, skipping generation")
        else:
            print(f"\n[Info] Found existing raw outputs in {output_dir}, skipping generation")
        print(f"       Use --force_regenerate to regenerate")
    
    if skip_generation:
        raw_output_path = f"{output_dir}/raw_outputs.jsonl"
        sample_responses = load_raw_outputs(raw_output_path)

        prompts_path = f"{output_dir}/prompts.jsonl"
        if os.path.exists(prompts_path):
            print(f"  Loading prompts from: {prompts_path}")
            with open(prompts_path) as f:
                prompts_data = [json.loads(line) for line in f]
            if len(prompts_data) == len(datas):
                for i, p in enumerate(prompts_data):
                    if "messages" in p:
                        datas[i]["messages"] = p["messages"]
                    if "ground_truth" in p:
                        datas[i]["ground_truth"] = p["ground_truth"]
    else:
        print(f"\n[Step 3] Preparing prompts")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        is_instruct = any(x in args.model_name.lower() for x in ['instruct', 'chat', 'it'])
        
        def format_prompt(messages):
            if is_instruct and tokenizer.chat_template:
                return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            return "\n".join(m["content"] for m in messages) + "\n"
        
        prompts = [format_prompt(d["messages"]) for d in datas]
        print(f"  ✓ Formatted {len(prompts)} prompts")

        prompts_path = f"{output_dir}/prompts.jsonl"
        print(f"  Saving prompts to: {prompts_path}")
        with open(prompts_path, 'w') as f:
            for i, (prompt, data) in enumerate(zip(prompts, datas)):
                record = {
                    "idx": i,
                    "prompt": prompt,
                    "messages": data["messages"],
                    "ground_truth": data["ground_truth"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        sample_responses = generate_responses(
            args, datas, prompts, top_k_perturbs, base_model_path, output_dir, handler
        )

        meta = {
            "dataset": args.dataset,
            "data_path": data_path,
            "model": base_model_path,
            "seeds_file": args.seeds_file,
            "num_samples": len(datas),
            "num_models": len(top_k_perturbs),
            "generation_complete": True,
        }
        meta_path = f"{output_dir}/meta.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"  ✓ Metadata saved: {meta_path}")

    print(f"\n[Step 4] Filtering by correctness")
    print(f"  Selection mode: {'all correct traces' if args.all_traces else 'shortest correct trace'}")
    
    require_format = not args.no_format_check
    sft_data = []
    stats = {
        "total_samples": len(datas),
        "total_responses": sum(len(r) for r in sample_responses.values()),
        "samples_with_correct": 0,
        "samples_without_correct": 0,
        "total_correct_traces": 0,
        "selected_traces": 0,
    }
    
    for sample_idx in range(len(datas)):
        if sample_idx not in sample_responses:
            stats["samples_without_correct"] += 1
            continue
            
        responses = sample_responses[sample_idx]
        data = datas[sample_idx]
        ground_truth = data["ground_truth"]
        
        correct_traces = filter_correct_responses(
            responses, ground_truth, handler,
            require_format_marker=require_format,
            all_traces=args.all_traces
        )
        
        if correct_traces:
            stats["samples_with_correct"] += 1
            for resp_text, _ in responses:
                if handler.compute_reward(resp_text, ground_truth) > 0:
                    stats["total_correct_traces"] += 1
            stats["selected_traces"] += len(correct_traces)
            
            for trace in correct_traces:
                sft_data.append({
                    "sample_idx": sample_idx,
                    "messages": data["messages"],
                    "response": trace["response"],
                    "extracted_answer": trace["extracted_answer"],
                    "ground_truth": ground_truth,
                    "response_length": trace["response_length"],
                    "model_idx": trace.get("model_idx"),
                    "seed": trace.get("seed"),
                    "sigma": trace.get("sigma"),
                })
        else:
            stats["samples_without_correct"] += 1

    print(f"\n  Filtering breakdown:")
    print(f"    Total samples:              {stats['total_samples']}")
    print(f"    Total responses:            {stats['total_responses']}")
    print(f"    Samples with correct resp:  {stats['samples_with_correct']}")
    print(f"    Samples without correct:    {stats['samples_without_correct']}")
    print(f"    Total correct traces:       {stats['total_correct_traces']}")
    print(f"    Selected traces:            {stats['selected_traces']}")
    
    if not sft_data:
        print("\n" + "=" * 70)
        print("ERROR: No correct responses found!")
        print("=" * 70)
        return

    lengths = [d["response_length"] for d in sft_data]
    print(f"\n  Response length stats:")
    print(f"    Min: {min(lengths)}, Max: {max(lengths)}, Avg: {sum(lengths)/len(lengths):.0f}")

    print(f"\n[Step 5] Saving SFT dataset")
    sft_output_dir = f"{output_dir}/sft"
    save_sft_dataset(sft_data, sft_output_dir, args.dataset)

    meta_path = f"{output_dir}/meta.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {}
    
    meta.update({
        "all_traces": args.all_traces,
        "require_format_marker": require_format,
        "filter_stats": stats,
        "sft_samples": len(sft_data),
    })
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print("SUCCESS: Distillation Pipeline Complete")

if __name__ == "__main__":
    main()

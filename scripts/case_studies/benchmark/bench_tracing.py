"""Benchmark circuit tracing throughput across batch sizes.

Usage:
    python scripts/case_studies/benchmark/bench_tracing.py --model-id meta-llama/Llama-3.1-8B-Instruct
    python scripts/case_studies/benchmark/bench_tracing.py --model-id Qwen/Qwen3-8B --batch-sizes 1 2 4 8
"""

import argparse
import importlib.util
import json
import logging
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch
from circuits.tracing.clja import ADAGConfig
from circuits.tracing.trace import CircuitData, convert_inputs_to_circuits
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def make_config(device: str) -> ADAGConfig:
    return ADAGConfig(
        device=device,
        verbose=False,
        parent_threshold=None,
        edge_threshold=0.01,
        node_attribution_threshold=None,
        topk=None,
        batch_aggregation="any",
        topk_neurons=None,
        percentage_threshold=0.005,
        use_relp_grad=True,
        disable_half_rule=False,
        disable_stop_grad=False,
        ablation_mode="zero",
        use_stop_grad_on_mlps=True,
        return_nodes_only=False,
        focus_last_residual=False,
        skip_attr_contrib=False,
        center_logits=False,
        ig_steps=None,
        ig_mode="ig-inputs",
        return_only_important_neurons=False,
        apply_blacklist=True,
    )


def _worker_fn(
    worker_id: int,
    gpu_ids: list[int],
    model_id: str,
    prompts: list[str],
    seed_responses: list[str],
    labels: list[str],
    batch_size: int,
    k: int,
) -> CircuitData:
    """Run circuit tracing on a shard of data using the assigned GPU(s)."""
    logger.info("Worker %d starting on GPU(s) %s with %d prompts", worker_id, gpu_ids, len(prompts))

    if len(gpu_ids) == 1:
        device_map = {"": f"cuda:{gpu_ids[0]}"}
        device = f"cuda:{gpu_ids[0]}"
    else:
        max_memory = {gpu: "70GiB" for gpu in gpu_ids}
        device_map = "sequential"
        device = f"cuda:{gpu_ids[0]}"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        max_memory={gpu: "70GiB" for gpu in gpu_ids} if len(gpu_ids) > 1 else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = make_config(device)
    data = convert_inputs_to_circuits(
        model,
        tokenizer,
        prompts,
        config=config,
        seed_responses=seed_responses,
        labels=labels,
        num_datapoints=len(prompts),
        batch_size=batch_size,
        k=k,
        ignore_bos=False,
    )
    logger.info("Worker %d finished", worker_id)
    return data


def bench_single_gpu(
    model,
    tokenizer,
    batch_size: int,
    n_prompts: int,
    prompts: list[str],
    seed_responses: list[str],
    labels: list[str],
    config: ADAGConfig,
    k: int,
) -> dict:
    """Run tracing with a given batch size on a single GPU."""
    p, s, l = prompts[:n_prompts], seed_responses[:n_prompts], labels[:n_prompts]

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    data = convert_inputs_to_circuits(
        model,
        tokenizer,
        p,
        config=config,
        seed_responses=s,
        labels=l,
        num_datapoints=len(p),
        batch_size=batch_size,
        k=k,
        ignore_bos=False,
    )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    return {
        "batch_size": batch_size,
        "n_prompts": n_prompts,
        "n_workers": 1,
        "gpus_per_model": 1,
        "elapsed_s": round(elapsed, 2),
        "per_prompt_s": round(elapsed / n_prompts, 2),
        "peak_gpu_gb": round(peak_mem, 2),
        "n_nodes": len(data.df_node),
        "n_edges": len(data.df_edge),
    }


def bench_multi_gpu(
    model_id: str,
    batch_size: int,
    n_prompts: int,
    gpus_per_model: int,
    prompts: list[str],
    seed_responses: list[str],
    labels: list[str],
    k: int,
) -> dict:
    """Run tracing with multiple workers, each on its own GPU group."""
    num_gpus = torch.cuda.device_count()
    num_workers = max(1, num_gpus // gpus_per_model)
    p, s, l = prompts[:n_prompts], seed_responses[:n_prompts], labels[:n_prompts]

    shard_indices = np.array_split(range(len(p)), num_workers)
    worker_args = []
    for wid, indices in enumerate(shard_indices):
        if len(indices) == 0:
            continue
        gpu_start = wid * gpus_per_model
        gpu_ids = list(range(gpu_start, gpu_start + gpus_per_model))
        worker_args.append(
            (
                wid,
                gpu_ids,
                model_id,
                [p[i] for i in indices],
                [s[i] for i in indices],
                [l[i] for i in indices],
                batch_size,
                k,
            )
        )

    logger.info(
        "Spawning %d workers (%d GPUs/model, %s prompts/worker)",
        len(worker_args),
        gpus_per_model,
        [len(wa[3]) for wa in worker_args],
    )

    t0 = time.perf_counter()
    mp.set_start_method("spawn", force=True)
    with mp.Pool(len(worker_args)) as pool:
        shards = pool.starmap(_worker_fn, worker_args)
    elapsed = time.perf_counter() - t0

    data = CircuitData.merge(shards)

    return {
        "batch_size": batch_size,
        "n_prompts": n_prompts,
        "n_workers": len(worker_args),
        "gpus_per_model": gpus_per_model,
        "elapsed_s": round(elapsed, 2),
        "per_prompt_s": round(elapsed / n_prompts, 2),
        "peak_gpu_gb": -1,  # can't measure across processes easily
        "n_nodes": len(data.df_node),
        "n_edges": len(data.df_edge),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--n-prompts", type=int, default=50, help="Number of prompts to trace")
    parser.add_argument("--k", type=int, default=5, help="Top-k logits to trace")
    parser.add_argument(
        "--gpus-per-model",
        type=int,
        default=1,
        help="GPUs per model copy. >1 with multiple GPUs enables multi-worker parallelism.",
    )
    parser.add_argument("--warmup", action="store_true", help="Run 1 warmup prompt first")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    num_workers = max(1, num_gpus // args.gpus_per_model)
    use_multi = num_workers > 1

    logger.info(
        "GPUs: %d, gpus_per_model: %d, workers: %d", num_gpus, args.gpus_per_model, num_workers
    )

    # Load dataset
    data_dir = Path(__file__).resolve().parent.parent.parent / "circuit_prep" / "data"
    spec = importlib.util.spec_from_file_location("data.capitals", data_dir / "capitals.py")
    dataset_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset_mod)
    prompts, seed_responses, labels = (
        dataset_mod.prompts,
        dataset_mod.seed_responses,
        dataset_mod.labels,
    )

    # For single-GPU, load model once
    model, tokenizer, load_time = None, None, 0.0
    if not use_multi:
        device = "cuda:0"
        logger.info("Loading model %s...", args.model_id)
        t_load = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        load_time = time.perf_counter() - t_load
        logger.info("Model loaded in %.1fs", load_time)

        config = make_config(device)

        # Warmup
        if args.warmup:
            logger.info("Running warmup (1 prompt, batch_size=1)...")
            bench_single_gpu(
                model, tokenizer, 1, 1, prompts, seed_responses, labels, config, args.k
            )
            logger.info("Warmup done")

    results = {
        "model_id": args.model_id,
        "n_prompts": args.n_prompts,
        "k": args.k,
        "dataset": "capitals",
        "num_gpus": num_gpus,
        "gpus_per_model": args.gpus_per_model,
        "num_workers": num_workers,
        "load_time_s": round(load_time, 2),
        "gpu": torch.cuda.get_device_name(0) if num_gpus > 0 else "N/A",
        "runs": [],
    }

    for bs in args.batch_sizes:
        if bs > args.n_prompts:
            logger.info("Skipping batch_size=%d (> n_prompts=%d)", bs, args.n_prompts)
            continue

        logger.info("Benchmarking batch_size=%d...", bs)
        if use_multi:
            r = bench_multi_gpu(
                args.model_id,
                bs,
                args.n_prompts,
                args.gpus_per_model,
                prompts,
                seed_responses,
                labels,
                args.k,
            )
        else:
            r = bench_single_gpu(
                model,
                tokenizer,
                bs,
                args.n_prompts,
                prompts,
                seed_responses,
                labels,
                config,
                args.k,
            )

        logger.info(
            "  batch_size=%d: %.1fs total, %.2fs/prompt, %.1fGB peak, %d workers",
            bs,
            r["elapsed_s"],
            r["per_prompt_s"],
            r["peak_gpu_gb"],
            r["n_workers"],
        )
        results["runs"].append(r)

    # Print summary
    print("\n" + "=" * 80)
    print(f"Model: {args.model_id}")
    print(f"Dataset: capitals ({args.n_prompts} prompts, k={args.k})")
    print(f"GPU: {results['gpu']} x{num_gpus}, {num_workers} worker(s)")
    if load_time > 0:
        print(f"Model load: {load_time:.1f}s")
    print("-" * 80)
    print(
        f"{'Batch':>6} {'Workers':>8} {'Total':>8} {'Per-prompt':>10} {'Peak GPU':>10} {'Nodes':>8} {'Edges':>8}"
    )
    for r in results["runs"]:
        peak = f"{r['peak_gpu_gb']:.1f}GB" if r["peak_gpu_gb"] >= 0 else "N/A"
        print(
            f"{r['batch_size']:>6} {r['n_workers']:>8} {r['elapsed_s']:>7.1f}s "
            f"{r['per_prompt_s']:>9.2f}s {peak:>10} {r['n_nodes']:>8} {r['n_edges']:>8}"
        )
    print("=" * 80)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved results to %s", out_path)


if __name__ == "__main__":
    main()

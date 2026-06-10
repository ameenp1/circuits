"""
Prepare and save circuit pickles from a dataset.

Supports YAML config files for reproducible experiments. CLI args override config values.
Automatically parallelizes across available GPUs when multiple are present.

Usage:
    python scripts/circuit_prep/prep.py --config configs/capitals.yaml
    python scripts/circuit_prep/prep.py --dataset texas --output-path /tmp/texas.pkl
    python scripts/circuit_prep/prep.py --config configs/texas_qwen3.yaml --gpus-per-model 2
"""

import argparse
import importlib.util
import logging
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml
from circuits.tracing.clja import ADAGConfig
from circuits.tracing.trace import CircuitData, convert_inputs_to_circuits
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI args with YAML config support (YAML sets defaults, CLI overrides)."""
    parser = argparse.ArgumentParser(description="Prepare and save circuit pickles from a dataset")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--model-id",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model ID. Default: meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name — imports from scripts.circuit_prep.data.<dataset> (e.g. texas)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save the circuit pickle file",
    )
    parser.add_argument(
        "--percentage-threshold",
        type=float,
        default=0.005,
        help="Attribution percentage threshold. Default: 0.005",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for circuit tracing. Default: 1",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of top logits to trace. Default: 5",
    )
    parser.add_argument(
        "--apply-blacklist",
        action="store_true",
        help="Apply neuron blacklist during tracing.",
    )
    parser.add_argument(
        "--no-apply-blacklist",
        action="store_true",
        help="Disable neuron blacklist during tracing.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output during tracing.",
    )
    parser.add_argument(
        "--keep-bos",
        action="store_true",
        default=False,
        help="Include BOS token in circuit (default: excluded).",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt to prepend to each input (via chat template).",
    )
    parser.add_argument(
        "--gpus-per-model",
        type=int,
        default=1,
        help="Number of GPUs per model copy. Default: 1. Use 2 for large models (e.g. Qwen3-32B).",
    )

    # First parse to get config path
    temp_args, _ = parser.parse_known_args()

    # Load YAML config and set as defaults
    if temp_args.config:
        config_path = Path(temp_args.config)
        if not config_path.is_absolute():
            config_path = Path(__file__).parent / config_path
        with open(config_path) as f:
            config = yaml.safe_load(f)

        for key, value in config.items():
            dest = key.replace("-", "_")
            for action in parser._actions:
                if action.dest == dest:
                    if isinstance(value, bool) and action.const is not None:
                        action.default = value
                    else:
                        action.default = value
                    break

    args = parser.parse_args()
    return args


def _make_adag_config(args: argparse.Namespace, apply_blacklist: bool, device: str) -> ADAGConfig:
    """Create an ADAGConfig from parsed args."""
    return ADAGConfig(
        device=device,
        verbose=args.verbose,
        parent_threshold=None,
        edge_threshold=0.01,
        node_attribution_threshold=None,
        topk=None,
        batch_aggregation="any",
        topk_neurons=None,
        percentage_threshold=args.percentage_threshold,
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
        apply_blacklist=apply_blacklist,
    )


def _worker_fn(
    worker_id: int,
    gpu_ids: list[int],
    model_id: str,
    prompts: list[str],
    seed_responses: list[str],
    labels: list[str],
    args: argparse.Namespace,
    apply_blacklist: bool,
    ignore_bos: bool,
) -> CircuitData:
    """Run circuit tracing on a shard of data using the assigned GPU(s)."""
    logger.info("Worker %d starting on GPU(s) %s with %d prompts", worker_id, gpu_ids, len(prompts))

    if len(gpu_ids) == 1:
        device_map = {"": f"cuda:{gpu_ids[0]}"}
        max_memory = None
        device = f"cuda:{gpu_ids[0]}"
    else:
        max_memory = {gpu: "70GiB" for gpu in gpu_ids}
        device_map = "sequential"
        device = f"cuda:{gpu_ids[0]}"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        max_memory=max_memory,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = _make_adag_config(args, apply_blacklist, device)

    data = convert_inputs_to_circuits(
        model,
        tokenizer,
        prompts,
        config=config,
        seed_responses=seed_responses,
        labels=labels,
        num_datapoints=len(prompts),
        batch_size=args.batch_size,
        k=args.k,
        ignore_bos=ignore_bos,
        system_prompt=args.system_prompt,
    )

    logger.info("Worker %d finished", worker_id)
    return data


def main():
    args = parse_args()

    if args.dataset is None:
        logger.error("--dataset is required (via CLI or config)")
        return 1

    if args.output_path is None:
        logger.error("--output-path is required (via CLI or config)")
        return 1

    # Resolve apply_blacklist (default True unless --no-apply-blacklist)
    apply_blacklist = not args.no_apply_blacklist

    ignore_bos = not args.keep_bos

    # Validate dataset file exists before loading the model (which is slow)
    data_dir = Path(__file__).parent / "data"
    dataset_file = data_dir / f"{args.dataset}.py"
    if not dataset_file.exists():
        logger.error("Dataset file not found: %s", dataset_file)
        logger.error(
            "Available datasets: %s",
            [f.stem for f in data_dir.glob("*.py") if f.stem != "__init__"],
        )
        return 1

    # Determine parallelism
    num_gpus = torch.cuda.device_count()
    gpus_per_model = args.gpus_per_model
    num_workers = max(1, num_gpus // gpus_per_model) if num_gpus > 0 else 1
    logger.info(
        "GPUs available: %d, GPUs per model: %d, workers: %d", num_gpus, gpus_per_model, num_workers
    )

    # Load dataset — for static datasets we can do this before model loading.
    # For dynamic datasets (get_dataset), we need a model, so fall back to single-worker.
    logger.info("Loading dataset from %s", dataset_file)
    spec = importlib.util.spec_from_file_location(f"data.{args.dataset}", dataset_file)
    dataset_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset_mod)

    has_dynamic_dataset = hasattr(dataset_mod, "get_dataset")

    if num_workers <= 1 or has_dynamic_dataset:
        # Single-worker path: load model in main process (current behavior)
        if has_dynamic_dataset and num_workers > 1:
            logger.warning(
                "Dynamic datasets (get_dataset) are not supported with multi-GPU parallelism. "
                "Falling back to single worker."
            )

        logger.info("Loading model %s...", args.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, torch_dtype=torch.bfloat16, device_map="sequential"
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if has_dynamic_dataset:
            prompts, seed_responses, labels = dataset_mod.get_dataset(model, tokenizer)
        else:
            prompts = dataset_mod.prompts
            seed_responses = dataset_mod.seed_responses
            labels = dataset_mod.labels

        logger.info(
            "Building circuit (threshold=%.4f, batch_size=%d, k=%d, blacklist=%s)...",
            args.percentage_threshold,
            args.batch_size,
            args.k,
            apply_blacklist,
        )
        device = "cuda" if num_gpus > 0 else "cpu"
        if device == "cpu":
            logger.warning(
                "No CUDA GPU detected — tracing on CPU. This works but is slow; a single "
                "consumer GPU (e.g. RTX 3080) or a RunPod box is much faster for Step 1. "
                "Steps 2-5 (export/describe/group/view) are CPU-only and unaffected."
            )
        config = _make_adag_config(args, apply_blacklist, device)
        data = convert_inputs_to_circuits(
            model,
            tokenizer,
            prompts,
            config=config,
            seed_responses=seed_responses,
            labels=labels,
            num_datapoints=len(prompts),
            batch_size=args.batch_size,
            k=args.k,
            ignore_bos=ignore_bos,
            system_prompt=args.system_prompt,
        )
    else:
        # Multi-worker path: split prompts across GPU groups
        prompts = dataset_mod.prompts
        seed_responses = dataset_mod.seed_responses
        labels = dataset_mod.labels

        n = len(prompts)
        shard_indices = np.array_split(range(n), num_workers)

        # Build per-worker GPU assignments
        worker_args = []
        for worker_id, indices in enumerate(shard_indices):
            if len(indices) == 0:
                continue
            gpu_start = worker_id * gpus_per_model
            gpu_ids = list(range(gpu_start, gpu_start + gpus_per_model))
            shard_prompts = [prompts[i] for i in indices]
            shard_seed_responses = [seed_responses[i] for i in indices]
            shard_labels = [labels[i] for i in indices]
            worker_args.append(
                (
                    worker_id,
                    gpu_ids,
                    args.model_id,
                    shard_prompts,
                    shard_seed_responses,
                    shard_labels,
                    args,
                    apply_blacklist,
                    ignore_bos,
                )
            )

        logger.info(
            "Spawning %d workers for %d prompts (%s prompts/worker)",
            len(worker_args),
            n,
            [len(wa[3]) for wa in worker_args],
        )

        mp.set_start_method("spawn", force=True)
        with mp.Pool(len(worker_args)) as pool:
            shards = pool.starmap(_worker_fn, worker_args)

        data = CircuitData.merge(shards)
        logger.info("Merged %d shards into single CircuitData", len(shards))

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.save_to_pickle(str(output_path))
    logger.info("Saved CircuitData to %s", output_path)

    return 0


if __name__ == "__main__":
    exit(main())

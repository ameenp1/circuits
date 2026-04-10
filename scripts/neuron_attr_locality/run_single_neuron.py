#!/usr/bin/env python
"""Analyze a single specified neuron: find top-k activating exemplars, then compute
input attributions (what tokens caused it to fire) and output contributions (what future
tokens it contributes to) for each exemplar.

Two-pass approach:
1. Stream through samples with cheap forward passes, maintaining a top-k heap for the
   single (layer, neuron, polarity) combination.
2. For each heap winner, do a forward pass with hooks + backward passes to compute
   input attr_vec and output contrib_vec.
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Flush prints immediately for slurm logs
sys.stdout.reconfigure(line_buffering=True)


def log(msg: str):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {msg}")


from circuits.tracing.grad import (
    remove_forward_hooks,
    revert_stop_nonlinear_grad,
    stop_nonlinear_grad,
)
from circuits.tracing.trace import get_chat_template
from circuits.utils.constants import RESULTS_DIR


def prepare_sample(
    sample: dict,
    tokenizer,
    max_seq_len: int,
    dataset: str,
) -> tuple[list[int], list[int]] | None:
    """Returns (input_ids_list, allowed_positions) or None if sample should be skipped."""
    if dataset == "fineweb":
        input_ids_list = tokenizer.encode(sample["text"], add_special_tokens=True)[:max_seq_len]
        seq_len = len(input_ids_list)
        if seq_len < 3:
            return None
        return input_ids_list, list(range(seq_len))

    # wildchat
    conversation = sample["conversation"]
    if len(conversation) < 2:
        return None
    if conversation[0]["role"] != "user" or conversation[1]["role"] != "assistant":
        return None

    user_msg = {"role": "user", "content": conversation[0]["content"]}
    asst_msg = {"role": "assistant", "content": conversation[1]["content"]}

    chat_template = get_chat_template(tokenizer)
    full_ids = tokenizer.apply_chat_template(
        [user_msg, asst_msg], tokenize=True, chat_template=chat_template
    )[:max_seq_len]

    if len(full_ids) < 3:
        return None
    return list(full_ids), list(range(len(full_ids)))


def entropy(probs: torch.Tensor) -> float:
    """Compute entropy of a probability distribution (base e)."""
    probs = probs[probs > 0]
    return -(probs * probs.log()).sum().item()


LOCALITY_KS = [1, 2, 4, 8]


def compute_attr_metrics(attr_vec: torch.Tensor, token_pos: int) -> dict:
    """Compute locality metrics from an attr vector, excluding BOS at position 0."""
    attr_sum = attr_vec.sum().item()
    abs_attr = attr_vec.abs()
    abs_attr_total = abs_attr.sum().item()
    attr_bos = abs_attr[0].item()
    abs_attr_no_bos = abs_attr[1:]
    abs_attr_no_bos_total = abs_attr_no_bos.sum().item()
    attr_bos_frac = attr_bos / abs_attr_total if abs_attr_total > 0 else 0.0

    locality_k = {}
    if abs_attr_no_bos_total > 0 and token_pos > 0:
        for k in LOCALITY_KS:
            start = max(1, token_pos - k + 1)
            locality_k[f"attr_locality_k{k}"] = (
                abs_attr[start : token_pos + 1].sum().item() / abs_attr_no_bos_total
            )
        attr_probs_no_bos = abs_attr_no_bos / abs_attr_no_bos.sum()
        attr_ent = entropy(attr_probs_no_bos)
    else:
        for k in LOCALITY_KS:
            locality_k[f"attr_locality_k{k}"] = 0.0
        attr_ent = 0.0

    return {
        "attr_sum": attr_sum,
        **locality_k,
        "attr_bos_frac": attr_bos_frac,
        "attr_entropy": attr_ent,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a single neuron: top-k exemplars with input attr + output contrib."
    )
    parser.add_argument("--layer", type=int, required=True, help="Layer index")
    parser.add_argument("--neuron", type=int, required=True, help="Neuron index")
    parser.add_argument(
        "--polarity",
        choices=["pos", "neg"],
        default="pos",
        help="Activation polarity to select (default: pos)",
    )
    parser.add_argument(
        "--model-id", default="meta-llama/Llama-3.1-8B-Instruct", help="HuggingFace model ID"
    )
    parser.add_argument(
        "--dataset",
        choices=["fineweb", "wildchat"],
        default="fineweb",
        help="Dataset to use",
    )
    parser.add_argument(
        "--num-samples", type=int, default=1000, help="Number of samples to stream through"
    )
    parser.add_argument("--max-seq-len", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--top-k", type=int, default=20, help="Exemplars to keep")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: auto)",
    )
    parser.add_argument(
        "--detailed", action="store_true", help="Save attr_vec and tokens per result"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for Pass 1 forward passes"
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = str(RESULTS_DIR / "neuron_single" / args.dataset)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    log(f"Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="sequential"
    )
    model.eval()

    target_layer = args.layer
    target_neuron = args.neuron
    polarity = args.polarity
    log(f"Target: layer={target_layer}, neuron={target_neuron}, polarity={polarity}")

    # ==================== PASS 1: Forward-only, build top-k heap ====================
    log("=== Pass 1: Collecting top-k activations (forward passes only) ===")

    heap: list[tuple[float, int, dict]] = []  # min-heap: (abs_act, counter, info)
    heap_counter = 0

    if args.dataset == "fineweb":
        ds = load_dataset(
            "HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True
        )
    else:
        ds = load_dataset("allenai/WildChat", split="train").shuffle(seed=args.seed)

    # Pre-collect samples, then process in batches
    batch_buf: list[tuple[int, list[int], list[int]]] = []  # (sample_idx, ids, allowed_pos)
    samples_processed = 0

    def _flush_batch():
        nonlocal heap_counter
        if not batch_buf:
            return
        B = len(batch_buf)
        max_len = max(len(ids) for _, ids, _ in batch_buf)
        input_ids_batch = torch.zeros(B, max_len, dtype=torch.long, device=device)
        attn_masks = torch.zeros(B, max_len, dtype=torch.long, device=device)
        for b, (_, ids, _) in enumerate(batch_buf):
            input_ids_batch[b, : len(ids)] = torch.tensor(ids)
            attn_masks[b, : len(ids)] = 1

        activation_cache: dict[str, torch.Tensor] = {}

        def _hook(module, input, output):
            activation_cache["act"] = input[0].detach().cpu()

        h = model.model.layers[target_layer].mlp.down_proj.register_forward_hook(_hook)
        with torch.no_grad():
            model(input_ids_batch, attention_mask=attn_masks)
        h.remove()

        # activation_cache["act"]: (B, max_len, D)
        neuron_acts = activation_cache["act"][:, :, target_neuron]  # (B, max_len)

        for b, (sample_idx, input_ids_list, allowed_positions) in enumerate(batch_buf):
            for pos in allowed_positions:
                if pos == 0:  # skip BOS
                    continue
                act_val = neuron_acts[b, pos].item()
                signed_act = act_val if polarity == "pos" else -act_val
                if signed_act <= 0:
                    continue

                if len(heap) < args.top_k or signed_act > heap[0][0]:
                    info = {
                        "sample_idx": sample_idx,
                        "token_pos": pos,
                        "activation": act_val,
                        "input_ids_list": input_ids_list,
                    }
                    if len(heap) < args.top_k:
                        heapq.heappush(heap, (signed_act, heap_counter, info))
                    else:
                        heapq.heapreplace(heap, (signed_act, heap_counter, info))
                    heap_counter += 1

        del activation_cache
        torch.cuda.empty_cache()

    for sample_idx, sample in enumerate(ds):
        if samples_processed >= args.num_samples:
            break

        prepared = prepare_sample(sample, tokenizer, args.max_seq_len, args.dataset)
        if prepared is None:
            continue
        input_ids_list, allowed_positions = prepared
        samples_processed += 1
        batch_buf.append((sample_idx, input_ids_list, allowed_positions))

        if samples_processed % 100 == 0 or samples_processed == 1:
            log(f"  Sample {samples_processed}/{args.num_samples}")

        if len(batch_buf) >= args.batch_size:
            _flush_batch()
            batch_buf.clear()

    _flush_batch()
    batch_buf.clear()

    log(f"Pass 1 done. {len(heap)} exemplars in heap.")
    if len(heap) == 0:
        log("No exemplars found. Exiting.")
        return

    # Sort heap entries by activation magnitude descending
    heap_entries = sorted(heap, key=lambda x: -x[0])

    # ==================== PASS 2: Grad attribution + contribution ====================
    log("=== Pass 2: Computing input attributions + output contributions ===")

    stop_nonlinear_grad(model, use_relp_grad=True)

    # After stop_nonlinear_grad, mlp is wrapped: real down_proj is at mlp.mlp.down_proj
    all_results = []

    for rank, (abs_act, _counter, info) in enumerate(heap_entries):
        input_ids_list = info["input_ids_list"]
        token_pos = info["token_pos"]
        seq_len = len(input_ids_list)
        input_ids = torch.tensor([input_ids_list], device=device)
        tokens = [tokenizer.decode(t) for t in input_ids_list]

        log(
            f"  Exemplar {rank + 1}/{len(heap_entries)}: "
            f"sample={info['sample_idx']}, pos={token_pos}, act={info['activation']:.4f}"
        )

        # Disable param grads
        for param in model.parameters():
            param.requires_grad = False

        # Clear any leftover hooks on target layer
        remove_forward_hooks(model.model.layers[target_layer].mlp.mlp.down_proj)

        # Forward pass with hook to cache the target neuron's activation (in graph)
        neuron_act_cache: dict[str, torch.Tensor] = {}

        def _cache_hook(module, input, output):
            neuron_act_cache["act"] = input[0]  # (B, T, D) — keep in graph

        h = model.model.layers[target_layer].mlp.mlp.down_proj.register_forward_hook(_cache_hook)

        embeds = model.model.embed_tokens(input_ids).detach().requires_grad_()
        out = model(inputs_embeds=embeds)
        logits = out.logits  # (1, T, V) — in graph

        h.remove()

        neuron_activation = neuron_act_cache["act"][0, token_pos, target_neuron]  # scalar, in graph

        # --- Input attribution: backward from neuron activation to embeddings ---
        embed_grad = torch.autograd.grad(neuron_activation, embeds, retain_graph=True)[
            0
        ]  # (1, T, D)
        # attr_vec: grad * embed summed over d_model
        attr_vec = (embed_grad[0, :seq_len] * embeds[0, :seq_len].detach()).sum(dim=-1)  # (T,)
        attr_vec = attr_vec.detach().cpu()

        metrics = compute_attr_metrics(attr_vec, token_pos)
        credit_ratio = metrics["attr_sum"] / (info["activation"] + 1e-10)

        # --- Output contributions: backward from each future position's true next-token logit ---
        # For positions [token_pos, token_pos+1, ..., seq_len-2], backward from logit of
        # input_ids[p+1] at position p, and read off this neuron's grad * act contribution.
        future_positions = list(range(token_pos, seq_len - 1))
        contrib_vec = []

        mlp_act_full = neuron_act_cache["act"]  # (1, T, D)

        for fi, fut_pos in enumerate(future_positions):
            next_token_id = input_ids_list[fut_pos + 1]
            goal = logits[0, fut_pos, next_token_id]

            is_last = fi == len(future_positions) - 1
            grad_act = torch.autograd.grad(goal, mlp_act_full, retain_graph=not is_last)[
                0
            ]  # (1, T, D)

            # This neuron's contribution: grad at (token_pos, target_neuron) * activation
            neuron_grad = grad_act[0, token_pos, target_neuron].item()
            neuron_contrib = neuron_grad * info["activation"]
            contrib_vec.append(neuron_contrib)

        # Clean up
        remove_forward_hooks(model.model.layers[target_layer].mlp.mlp.down_proj)
        del out, logits, embeds, neuron_act_cache, mlp_act_full
        torch.cuda.empty_cache()

        result = {
            "rank": rank,
            "sample_idx": info["sample_idx"],
            "token_pos": token_pos,
            "activation": info["activation"],
            "seq_len": seq_len,
            "credit_ratio": credit_ratio,
            **metrics,
            "contrib_vec": contrib_vec,
            "contrib_positions": future_positions,
        }
        if args.detailed:
            result["tokens"] = tokens
            result["attr_vec"] = attr_vec.tolist()

        all_results.append(result)

    revert_stop_nonlinear_grad(model)

    # Print summary
    print(f"\n{'=' * 100}")
    print(f"Layer {target_layer}, Neuron {target_neuron}, Polarity {polarity}")
    print(f"Total: {len(all_results)} exemplars")
    print(f"{'=' * 100}")
    k_headers = " | ".join(f"{'k=' + str(k):>6}" for k in LOCALITY_KS)
    header = (
        f"{'rank':>4} | {'sample':>6} | {'pos':>4} | {'act':>8} | "
        f"{'credit':>6} | {k_headers} | {'bos':>6} | {'ent':>6} | {'contrib_len':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in all_results:
        k_vals = " | ".join(f"{r[f'attr_locality_k{k}']:>6.3f}" for k in LOCALITY_KS)
        print(
            f"{r['rank']:>4} | {r['sample_idx']:>6} | {r['token_pos']:>4} | "
            f"{r['activation']:>8.4f} | {r['credit_ratio']:>6.3f} | "
            f"{k_vals} | {r['attr_bos_frac']:>6.3f} | {r['attr_entropy']:>6.2f} | "
            f"{len(r['contrib_vec']):>10}"
        )

    # Save results
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    filename = f"L{target_layer}_N{target_neuron}_{polarity}.json"
    json_path = out_path / filename

    output = {
        "config": {
            "layer": target_layer,
            "neuron": target_neuron,
            "polarity": polarity,
            "model_id": args.model_id,
            "dataset": args.dataset,
            "num_samples": args.num_samples,
            "max_seq_len": args.max_seq_len,
            "top_k": args.top_k,
            "seed": args.seed,
            "detailed": args.detailed,
        },
        "results": all_results,
    }
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
    import os

    os._exit(0)  # skip Python finalization to avoid PyGILState crash with CUDA threads

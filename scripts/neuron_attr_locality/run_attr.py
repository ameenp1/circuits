#!/usr/bin/env python
"""Measure how localized neuron gradient attributions (attr_map) are across token positions.

Two-pass approach:
1. Stream through all samples with cheap forward passes, maintaining top-k heaps by activation
   magnitude per (neuron, polarity). Stores the input_ids and token position for each heap entry.
2. For only the final heap winners, reload the relevant samples and compute expensive gradient
   attributions.
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Flush prints immediately for slurm logs
sys.stdout.reconfigure(line_buffering=True)


def log(msg: str):
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {msg}")


from circuits.tracing.attribution import _get_neuron_attr_and_contrib
from circuits.tracing.grad import revert_stop_nonlinear_grad, stop_nonlinear_grad
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
    ids_no_gen = tokenizer.apply_chat_template(
        [user_msg], add_generation_prompt=False, chat_template=chat_template
    )
    ids_with_gen = tokenizer.apply_chat_template(
        [user_msg], add_generation_prompt=True, chat_template=chat_template
    )
    full_ids = tokenizer.apply_chat_template(
        [user_msg, asst_msg], tokenize=True, chat_template=chat_template
    )[:max_seq_len]

    # Only the <|start_header_id|> token (one past the trailing eot_id)
    start_header_pos = len(ids_no_gen)
    allowed_positions = [start_header_pos] if start_header_pos < len(full_ids) else []
    if not allowed_positions:
        return None

    if len(full_ids) < 3:
        return None
    return list(full_ids), allowed_positions


def entropy(probs: torch.Tensor) -> float:
    """Compute entropy of a probability distribution (base e)."""
    probs = probs[probs > 0]
    return -(probs * probs.log()).sum().item()


LOCALITY_KS = [1, 2, 4, 8]


def compute_attr_metrics(attr_vec: torch.Tensor, token_pos: int) -> dict:
    """Compute locality metrics from an attr vector, excluding BOS at position 0.

    Computes:
    - attr_locality_k{K}: fraction of |attr| (excl BOS) in the last K tokens up to and including
      the firing position. E.g. k=1 is just the firing token, k=4 is positions
      [token_pos-3, ..., token_pos].
    - attr_bos_frac: fraction of total |attr| at BOS.
    - attr_entropy: entropy of |attr| distribution (excl BOS).
    """
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
            start = max(1, token_pos - k + 1)  # clamp to 1 to exclude BOS
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


def compute_neuron_attrs_batched(
    model,
    input_ids: torch.Tensor,
    seq_len: int,
    candidates: list[tuple[int, int, int]],
) -> list[torch.Tensor]:
    """Run batched grad attribution for multiple (layer, nid, pos) tuples.

    Returns list of attr_vec tensors (src,), one per candidate, in the same order.
    """
    src_tokens = list(range(seq_len))
    dummy_tgt = [0]
    dummy_focus_logits = [input_ids[0, 1].item()]

    # Build neuron_cfg grouped by layer, preserving order
    neuron_cfg: dict[int, list[list[int]]] = defaultdict(list)
    for layer_idx, nid, pos in candidates:
        neuron_cfg[layer_idx].append([pos, nid])
    # Sort by layer for deterministic ordering (matches function's iteration order)
    neuron_cfg = dict(sorted(neuron_cfg.items()))

    attr, contrib, embed_grad_contrib, neuron_tags = _get_neuron_attr_and_contrib(
        model,
        neuron_cfg,
        input_ids,
        src_tokens,
        dummy_tgt,
        focus_positions=dummy_tgt,
        focus_logits=[dummy_focus_logits],
        attention_masks=None,
        use_relp_grad=True,
        neuron_chunk_size=200,
    )
    # attr shape: (n_neurons, 1, src)
    # neuron_tags gives the order: we need to map back to candidates
    tag_to_idx = {}
    for i, tag in enumerate(neuron_tags):
        tag_to_idx[(tag.layer, tag.neuron, tag.token)] = i

    result_vecs = []
    for layer_idx, nid, pos in candidates:
        idx = tag_to_idx[(layer_idx, nid, pos)]
        result_vecs.append(attr[idx, 0, :].detach().cpu())

    del attr, contrib, embed_grad_contrib
    torch.cuda.empty_cache()
    return result_vecs


def main():
    parser = argparse.ArgumentParser(
        description="Measure locality of neuron gradient attributions (attr_map)."
    )
    parser.add_argument(
        "--model-id", default="meta-llama/Llama-3.1-8B-Instruct", help="HuggingFace model ID"
    )
    parser.add_argument(
        "--neurons-per-layer", type=int, default=1, help="Neurons to select per layer"
    )
    parser.add_argument(
        "--num-samples", type=int, default=1, help="Number of samples to stream through"
    )
    parser.add_argument(
        "--pass0-samples",
        type=int,
        default=10,
        help="(wildchat) Number of samples for Pass 0 neuron screening",
    )
    parser.add_argument("--max-seq-len", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--top-k", type=int, default=20, help="Keep top-k instances per neuron")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--detailed", action="store_true", help="Save attr_vec and tokens per result"
    )
    parser.add_argument(
        "--dataset",
        choices=["fineweb", "wildchat"],
        default="fineweb",
        help="Dataset to use: fineweb (all positions) or wildchat (separator tokens only)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: auto based on dataset)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for Pass 1 forward passes"
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = str(RESULTS_DIR / "neuron_attr_locality" / args.dataset)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    log(f"Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="sequential"
    )
    model.eval()

    num_layers = len(model.model.layers)
    intermediate_dim = model.config.intermediate_size

    if args.dataset == "fineweb":
        # Random neuron sampling for fineweb
        torch.manual_seed(args.seed)
        sampled_neurons: list[tuple[int, int]] = []
        for layer_idx in range(num_layers):
            neuron_indices = torch.randint(0, intermediate_dim, (args.neurons_per_layer,))
            for nid in neuron_indices:
                sampled_neurons.append((layer_idx, nid.item()))
        log(f"Randomly sampled {len(sampled_neurons)} neurons across {num_layers} layers")
    else:
        # ==================== PASS 0: Screen neurons for separator preference ====================
        log(
            f"=== Pass 0: Screening neurons for separator preference ({args.pass0_samples} samples) ==="
        )

        # Count how often each neuron's argmax lands on a separator position
        sep_argmax_count = torch.zeros(num_layers, intermediate_dim, dtype=torch.long)
        ds0 = load_dataset("allenai/WildChat", split="train").shuffle(seed=args.seed)
        p0_processed = 0
        for sample in ds0:
            if p0_processed >= args.pass0_samples:
                break
            prepared = prepare_sample(sample, tokenizer, args.max_seq_len, args.dataset)
            if prepared is None:
                continue
            input_ids_list, allowed_positions = prepared
            p0_processed += 1
            seq_len = len(input_ids_list)
            input_ids = torch.tensor([input_ids_list], device=device)

            sep_mask = torch.zeros(seq_len, dtype=torch.bool)
            for p in allowed_positions:
                sep_mask[p] = True

            # Forward pass, collect activations at all layers
            all_acts: dict[int, torch.Tensor] = {}
            hooks = []
            for lid in range(num_layers):

                def _make_hook(lid: int):
                    def hook_fn(module, input, output):
                        all_acts[lid] = input[0][0].detach().cpu()  # (seq_len, intermediate)

                    return hook_fn

                h = model.model.layers[lid].mlp.down_proj.register_forward_hook(_make_hook(lid))
                hooks.append(h)

            with torch.no_grad():
                model(input_ids)
            for h in hooks:
                h.remove()

            for lid in range(num_layers):
                acts = all_acts[lid]  # (seq_len, intermediate)
                argmax_pos = acts.abs().argmax(dim=0)  # (intermediate,)
                is_sep = sep_mask[argmax_pos]  # (intermediate,) bool
                sep_argmax_count[lid] += is_sep.long()

            del all_acts
            torch.cuda.empty_cache()

            if p0_processed % 5 == 0 or p0_processed == 1:
                log(f"  Pass 0: {p0_processed}/{args.pass0_samples} samples")

        # Select top neurons_per_layer per layer by separator preference
        sampled_neurons = []
        for lid in range(num_layers):
            counts = sep_argmax_count[lid]
            topk_vals, topk_ids = counts.topk(args.neurons_per_layer)
            for nid, cnt in zip(topk_ids.tolist(), topk_vals.tolist()):
                sampled_neurons.append((lid, nid))
            if lid % 8 == 0:
                log(
                    f"  Layer {lid}: top neuron has sep-argmax "
                    f"{topk_vals[0].item()}/{p0_processed} samples"
                )

        log(f"Pass 0 done. Selected {len(sampled_neurons)} separator-preferring neurons.")

    # ==================== PASS 1: Forward-only, build top-k heaps ====================
    log("=== Pass 1: Collecting top-k activations (forward passes only) ===")

    # Min-heaps for top-k by activation magnitude, keyed by (layer, neuron, polarity).
    # Each heap entry: (abs_activation, counter, info_dict)
    # info_dict stores sample_idx, token_pos, activation, input_ids_list for later grad pass.
    heaps: dict[tuple[int, int, str], list] = {}
    for layer_idx, nid in sampled_neurons:
        for pol in ["pos", "neg"]:
            heaps[(layer_idx, nid, pol)] = []

    heap_counter = 0

    if args.dataset == "fineweb":
        ds = load_dataset(
            "HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True
        )
    else:
        ds = load_dataset("allenai/WildChat", split="train").shuffle(seed=args.seed)

    layers_needed = sorted(set(l for l, _ in sampled_neurons))

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

        activations: dict[int, torch.Tensor] = {}
        hooks = []
        for layer_idx in layers_needed:

            def _make_hook(lid: int):
                def hook_fn(module, input, output):
                    activations[lid] = input[0].detach().cpu()

                return hook_fn

            h = model.model.layers[layer_idx].mlp.down_proj.register_forward_hook(
                _make_hook(layer_idx)
            )
            hooks.append(h)

        with torch.no_grad():
            model(input_ids_batch, attention_mask=attn_masks)

        for h in hooks:
            h.remove()

        for b, (sample_idx, input_ids_list, allowed_positions) in enumerate(batch_buf):
            for layer_idx, nid in sampled_neurons:
                neuron_acts = activations[layer_idx][b, :, nid]

                for pos in allowed_positions:
                    act_val = neuron_acts[pos].item()

                    for pol, signed_act in [("pos", act_val), ("neg", -act_val)]:
                        if signed_act <= 0:
                            continue

                        heap = heaps[(layer_idx, nid, pol)]
                        if len(heap) < args.top_k or signed_act > heap[0][0]:
                            info = {
                                "sample_idx": sample_idx,
                                "token_pos": pos,
                                "activation": act_val,
                                "input_ids_list": input_ids_list,
                                "act_vec": neuron_acts.tolist(),
                            }
                            if len(heap) < args.top_k:
                                heapq.heappush(heap, (signed_act, heap_counter, info))
                            else:
                                heapq.heapreplace(heap, (signed_act, heap_counter, info))
                            heap_counter += 1

        del activations
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

    total_in_heaps = sum(len(h) for h in heaps.values())
    log(f"Pass 1 done. {total_in_heaps} instances in heaps.")

    # ==================== PASS 2: Grad attribution for heap winners ====================
    log("=== Pass 2: Computing grad attributions for heap winners ===")

    # Group heap entries by their input_ids_list (same sample → same forward pass)
    # Key: tuple(input_ids_list) for hashing
    sample_groups: dict[tuple[int, ...], list[dict]] = defaultdict(list)
    for (layer_idx, nid, pol), heap in heaps.items():
        for abs_act, _counter, info in heap:
            key = tuple(info["input_ids_list"])
            sample_groups[key].append(
                {
                    "layer": layer_idx,
                    "neuron": nid,
                    "polarity": pol,
                    "token_pos": info["token_pos"],
                    "activation": info["activation"],
                    "sample_idx": info["sample_idx"],
                    "abs_act": abs_act,
                    "act_vec": info["act_vec"],
                }
            )

    log(f"  {len(sample_groups)} unique samples to process, {total_in_heaps} total neurons")

    stop_nonlinear_grad(model, use_relp_grad=True)

    all_results = []
    processed = 0
    for ids_tuple, entries in sample_groups.items():
        input_ids_list = list(ids_tuple)
        seq_len = len(input_ids_list)
        input_ids = torch.tensor([input_ids_list], device=device)
        tokens = [tokenizer.decode(t) for t in input_ids_list]

        candidate_tuples = [(e["layer"], e["neuron"], e["token_pos"]) for e in entries]
        attr_vecs = compute_neuron_attrs_batched(model, input_ids, seq_len, candidate_tuples)

        for entry, attr_vec in zip(entries, attr_vecs):
            metrics = compute_attr_metrics(attr_vec, entry["token_pos"])
            result = {
                "layer": entry["layer"],
                "neuron": entry["neuron"],
                "polarity": entry["polarity"],
                "token_pos": entry["token_pos"],
                "activation": entry["activation"],
                "sample_idx": entry["sample_idx"],
                "credit_ratio": metrics["attr_sum"] / (entry["activation"] + 1e-10),
                **metrics,
            }
            if args.detailed:
                result["tokens"] = tokens
                result["attr_vec"] = attr_vec.tolist()
                result["act_vec"] = entry["act_vec"]
            all_results.append(result)

        processed += len(entries)
        if processed % 500 == 0 or processed == total_in_heaps:
            log(f"  {processed}/{total_in_heaps} neurons done")

    revert_stop_nonlinear_grad(model)

    # Sort by layer, neuron, polarity, then by activation magnitude descending
    all_results.sort(key=lambda r: (r["layer"], r["neuron"], r["polarity"], -abs(r["activation"])))

    # Print summary table
    print(f"\n{'=' * 110}")
    print(f"Total: {len(all_results)} results")
    print(f"{'=' * 110}")
    k_headers = " | ".join(f"{'k=' + str(k):>6}" for k in LOCALITY_KS)
    header = (
        f"{'layer':>5} | {'neuron':>6} | {'pol':>3} | {'sample':>6} | {'pos':>4} | "
        f"{'act':>8} | {'credit':>6} | {k_headers} | {'bos':>6} | {'ent':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in all_results:
        k_vals = " | ".join(f"{r[f'attr_locality_k{k}']:>6.3f}" for k in LOCALITY_KS)
        print(
            f"{r['layer']:>5} | {r['neuron']:>6} | {r['polarity']:>3} | "
            f"{r['sample_idx']:>6} | {r['token_pos']:>4} | "
            f"{r['activation']:>8.4f} | {r['credit_ratio']:>6.3f} | "
            f"{k_vals} | {r['attr_bos_frac']:>6.3f} | {r['attr_entropy']:>6.2f}"
        )

    # Save results
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()

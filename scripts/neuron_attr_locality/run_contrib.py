#!/usr/bin/env python
"""Measure where the top-contributing neurons are located relative to each target token.

For each target position, do one backward pass from that logit to get grad×act for ALL neurons
across ALL layers. Per layer, find the neuron with the highest |grad×act| (ignoring BOS),
and record the distance from that neuron's token position to the target.

This is cheap: one backward pass per target token.
"""

from __future__ import annotations

import argparse
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
    """Returns (input_ids_list, target_positions) or None if sample should be skipped.

    target_positions: positions whose next-token logit we trace back from.
    For fineweb: all positions except the last (every token predicts the next).
    For wildchat: positions within the separator region (so we trace the logit that predicts
    the first assistant token, etc.).
    """
    if dataset == "fineweb":
        input_ids_list = tokenizer.encode(sample["text"], add_special_tokens=True)[:max_seq_len]
        seq_len = len(input_ids_list)
        if seq_len < 3:
            return None
        # Every position except the last can predict a next token
        return input_ids_list, list(range(seq_len - 1))

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

    # Target: separator token positions that have a next token to predict
    sep_start = len(ids_no_gen) - 1
    sep_end = len(ids_with_gen)
    target_positions = [p for p in range(sep_start, sep_end) if p < len(full_ids) - 1]
    if not target_positions:
        return None

    if len(full_ids) < 3:
        return None
    return list(full_ids), target_positions


def main():
    parser = argparse.ArgumentParser(
        description="Measure distance of top-contributing neurons to each target token."
    )
    parser.add_argument(
        "--model-id", default="meta-llama/Llama-3.1-8B-Instruct", help="HuggingFace model ID"
    )
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to process")
    parser.add_argument("--max-seq-len", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dataset",
        choices=["fineweb", "wildchat"],
        default="fineweb",
        help="Dataset to use",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Number of samples to process in parallel"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: auto based on dataset)",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = str(RESULTS_DIR / "neuron_contrib_locality" / args.dataset)

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

    stop_nonlinear_grad(model, use_relp_grad=True)

    # Load dataset
    if args.dataset == "fineweb":
        ds = load_dataset(
            "HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True
        )
    else:
        ds = load_dataset("allenai/WildChat", split="train").shuffle(seed=args.seed)

    # Pre-collect all samples, then process in batches
    all_prepared = []  # list of (sample_idx, input_ids_list, target_positions, tokens)
    samples_processed = 0
    for sample_idx, sample in enumerate(ds):
        if samples_processed >= args.num_samples:
            break
        prepared = prepare_sample(sample, tokenizer, args.max_seq_len, args.dataset)
        if prepared is None:
            continue
        input_ids_list, target_positions = prepared
        tokens = [tokenizer.decode(t) for t in input_ids_list]
        all_prepared.append((sample_idx, input_ids_list, target_positions, tokens))
        samples_processed += 1

    log(f"Collected {samples_processed} samples, processing in batches of {args.batch_size}")

    all_results = []
    for batch_start in range(0, len(all_prepared), args.batch_size):
        batch = all_prepared[batch_start : batch_start + args.batch_size]
        B = len(batch)
        max_len = max(len(ids) for _, ids, _, _ in batch)

        # Pad input_ids and build attention masks
        input_ids_batch = torch.zeros(B, max_len, dtype=torch.long, device=device)
        attn_masks = torch.zeros(B, max_len, dtype=torch.long, device=device)
        for b, (_, ids, _, _) in enumerate(batch):
            input_ids_batch[b, : len(ids)] = torch.tensor(ids)
            attn_masks[b, : len(ids)] = 1

        # Union of all target positions in this batch
        all_targets: set[int] = set()
        for _, _, tgts, _ in batch:
            all_targets.update(tgts)

        log(
            f"  Batch {batch_start // args.batch_size + 1} "
            f"({B} samples, max_len={max_len}, {len(all_targets)} unique targets)"
        )

        # --- Single forward pass for this batch ---
        for param in model.parameters():
            param.requires_grad = False

        # Clear and register hooks to capture MLP activations
        for i in range(num_layers):
            remove_forward_hooks(model.model.layers[i].mlp.mlp.down_proj)

        embeds = model.model.embed_tokens(input_ids_batch).detach().requires_grad_()
        cache: dict[int, torch.Tensor] = {}

        for li in range(num_layers):

            def _hook(layer_index):
                def fn(_, input, output):
                    cache[layer_index] = input[0]

                return fn

            model.model.layers[li].mlp.mlp.down_proj.register_forward_hook(_hook(li))

        out = model(inputs_embeds=embeds, attention_mask=attn_masks)
        logits_all = out.logits  # (B, T, V) — stays in graph

        grad_targets = [cache[i] for i in range(num_layers)]

        # --- Loop over targets: only backward, reusing the forward graph ---
        sorted_targets = sorted(all_targets)
        for ti, tgt_pos in enumerate(sorted_targets):
            if (ti + 1) % 25 == 0:
                log(f"    target {ti + 1}/{len(all_targets)}")

            # Build per-batch focus logits
            focus_logit_ids = []
            for b, (_, ids, _, _) in enumerate(batch):
                if tgt_pos + 1 < len(ids):
                    focus_logit_ids.append(ids[tgt_pos + 1])
                else:
                    focus_logit_ids.append(0)

            # Select logits at tgt_pos for each batch item
            foc_log = torch.tensor(focus_logit_ids, device=device)  # (B,)
            selected = logits_all[:, tgt_pos, :]  # (B, V)
            goal = selected[torch.arange(B, device=device), foc_log].sum()

            is_last = ti == len(sorted_targets) - 1
            all_grads = torch.autograd.grad(goal, grad_targets, retain_graph=not is_last)

            # Extract results for each batch item that has this target
            for b, (sidx, ids, tgts, toks) in enumerate(batch):
                if tgt_pos not in tgts:
                    continue
                seq_len = len(ids)

                tgt_result = {
                    "sample_idx": sidx,
                    "target_pos": tgt_pos,
                    "target_token": toks[tgt_pos],
                    "predicted_token": toks[tgt_pos + 1],
                    "seq_len": seq_len,
                    "per_layer": [],
                }

                for lid in range(num_layers):
                    # grad * activation (zero ablation)
                    attr = (all_grads[lid][b, :seq_len] * cache[lid][b, :seq_len]).detach()
                    attr[0, :] = 0.0  # zero out BOS

                    abs_attr = attr.abs()
                    flat_idx = abs_attr.argmax().item()
                    best_pos = flat_idx // attr.shape[1]
                    best_nid = flat_idx % attr.shape[1]
                    best_val = attr[best_pos, best_nid].item()
                    distance = tgt_pos - best_pos

                    tgt_result["per_layer"].append(
                        {
                            "layer": lid,
                            "neuron": best_nid,
                            "token_pos": best_pos,
                            "distance": distance,
                            "attr_value": best_val,
                            "token_at_pos": toks[best_pos],
                        }
                    )

                all_results.append(tgt_result)

        # Cleanup hooks and free graph
        for i in range(num_layers):
            remove_forward_hooks(model.model.layers[i].mlp.mlp.down_proj)
        del out, logits_all, embeds, cache, grad_targets
        torch.cuda.empty_cache()

    revert_stop_nonlinear_grad(model)

    # Print summary: mean distance per layer
    print(f"\n{'=' * 60}")
    print(f"Total: {len(all_results)} target positions across {samples_processed} samples")
    print(f"{'=' * 60}")
    print(f"{'layer':>5} | {'mean_dist':>9} | {'med_dist':>8} | {'std_dist':>8} | {'frac_at_0':>9}")
    print("-" * 50)
    for lid in range(num_layers):
        distances = [r["per_layer"][lid]["distance"] for r in all_results]
        dt = torch.tensor(distances, dtype=torch.float)
        frac_at_0 = (dt == 0).float().mean().item()
        print(
            f"{lid:>5} | {dt.mean().item():>9.2f} | {dt.median().item():>8.1f} | "
            f"{dt.std().item():>8.2f} | {frac_at_0:>9.3f}"
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
    import os

    os._exit(0)  # skip Python finalization to avoid PyGILState crash with CUDA threads

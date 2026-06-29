"""
add_logit_weights.py — Fix #2: global promote/demote via logit-weights for traced Gemma MLP neurons.

NOT causal. This is the standard "logit-weights" (direct-path) heuristic, computed to MATCH the
transcoder side's hosted top_logits/bottom_logits lens. Weights-only: no corpus, no forward pass.
For each neuron it projects the down_proj output direction through Gemma-2's
post_feedforward_layernorm + final RMSNorm gains into the (tied) unembedding, and reads the top /
bottom vocab tokens. Adds `top_logits` / `bottom_logits` into each card of mlp_exemplars.json,
in place — so serve_with_cards.py and generate_description.py read them straight off the card.

Gemma-2 correctness (vs a Llama-shaped `W_U @ norm(down_proj_col)`):
  - routes through `post_feedforward_layernorm` (the extra sandwich norm) AND the final
    `model.norm`, using `_effective_norm_weight` (the (1+weight) gain) — not hardcoded.
  - tied embeddings: unembed = the input embedding (`get_output_embeddings`).
  - final logit softcapping is monotonic -> does not change top-k ranking -> skipped.
  - the RMS normalization (1/||x||) is context-dependent -> omitted (this is logit-weights, not
    a causal contribution).
  - intermediate layers L+1..end are ignored (direct path) -> a shallow, late-layer-biased
    heuristic. Label it as such; do not call it causal.

Polarity: '+' uses the down_proj column, '-' uses its negation (the neuron writes with negative
activation), so promote/demote is correct per polarity.

Run on a GPU box. Needs HF_TOKEN + `.env`. Seconds (no forward passes).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description="Add Gemma-2-correct logit-weights to the exemplar store.")
    ap.add_argument("--exemplars", type=Path, default=Path("custom_automation/np_data/mlp_exemplars.json"))
    ap.add_argument("--model-id", default="google/gemma-2-2b")
    ap.add_argument("--top-k", type=int, default=5)   # match the SLT side (Neuronpedia stores 5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from circuits.tracing.grad import _effective_norm_weight  # (1 + weight) gain, Gemma-2-aware

    store = json.loads(args.exemplars.read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(args.model_id)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype).to(args.device).eval()

    dev = args.device
    W_U = model.get_output_embeddings().weight.detach().float().to(dev)          # [vocab, d_model] (tied)
    g_final = _effective_norm_weight(model.model.norm).detach().float().to(dev)  # [d_model]
    layers = model.model.layers
    g_pff = {
        li: _effective_norm_weight(layers[li].post_feedforward_layernorm).detach().float().to(dev)
        for li in range(len(layers))
    }

    updated = 0
    with torch.no_grad():
        for key, card in store.items():
            try:  # key = "L{layer}_N{neuron}_{pol}"
                lpart, npart, pol = key.split("_")
                layer, neuron = int(lpart[1:]), int(npart[1:])
            except ValueError:
                continue
            d = layers[layer].mlp.down_proj.weight[:, neuron].detach().float().to(dev)  # [d_model]
            if pol == "-":
                d = -d
            d = d * g_pff[layer] * g_final                 # post-FF norm gain + final norm gain
            logits = W_U @ d                                # [vocab]; tied unembed, softcap skipped (monotonic)
            top = torch.topk(logits, args.top_k).indices.tolist()
            bot = torch.topk(-logits, args.top_k).indices.tolist()
            card["top_logits"] = [tok.decode([i]) for i in top]
            card["bottom_logits"] = [tok.decode([i]) for i in bot]
            updated += 1

    args.exemplars.write_text(json.dumps(store), encoding="utf-8")
    print(f"Added logit-weights (top/bottom-{args.top_k}) to {updated} cards -> {args.exemplars}")

    # spot-check a few so a Gemma-2 norm error is visible
    for key in list(store)[:4]:
        c = store[key]
        print(f"  {key}  promote={c.get('top_logits', [])[:5]}  demote={c.get('bottom_logits', [])[:5]}")


if __name__ == "__main__":
    main()

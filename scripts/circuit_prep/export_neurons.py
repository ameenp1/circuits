"""
CLI: export ADAG's top-N MLP neurons *with their associated text* for a single prompt
(or a single traced circuit), for use in your own description / feature-grouping pipeline.

The reusable logic lives in `circuits.analysis.neuron_export`. For many prompts at once,
use `scripts/circuit_prep/batch_export_neurons.py` instead.

Two ways to run:

  # A) trace a fresh prompt and export
  uv run python scripts/circuit_prep/export_neurons.py \
      --model-id google/gemma-2-2b \
      --prompt "What is the capital of the state containing Dallas? Answer:" \
      --target " Austin" --top-n 30 --out neurons.json

  # B) export from a circuit already traced by prep.py (configs/capitals_gemma.yaml)
  uv run python scripts/circuit_prep/export_neurons.py \
      --model-id google/gemma-2-2b \
      --circuit results/case_studies/capitals_gemma_circuit.pkl \
      --top-n 50 --out neurons.json
"""

import argparse
import json


def _main() -> None:
    import torch
    from circuits.analysis.circuit_ops import Circuit
    from circuits.analysis.neuron_export import (
        export_top_neurons,
        export_top_neurons_from_circuit,
    )
    from circuits.tracing.trace import CircuitData
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(
        description="Export top-N MLP neurons + text for Gemma 2 2B as JSON."
    )
    parser.add_argument("--model-id", default="google/gemma-2-2b", help="HuggingFace model ID.")
    parser.add_argument("--circuit", default=None, help="Path to an existing circuit pickle.")
    parser.add_argument("--prompt", default=None, help="Prompt to trace (if no --circuit).")
    parser.add_argument("--target", default=None, help="Target token to trace to (e.g. ' Austin').")
    parser.add_argument("--seed-response", default="Answer:", help="Seed/assistant prefix.")
    parser.add_argument("--top-n", type=int, default=30, help="Number of neurons to export.")
    parser.add_argument("--top-tokens", type=int, default=8, help="Top tokens per example.")
    parser.add_argument(
        "--percentage-threshold", type=float, default=0.01, help="Attribution pruning cutoff."
    )
    parser.add_argument("--k", type=int, default=5, help="Top-k logits to trace.")
    parser.add_argument("--out", default="neurons.json", help="Output JSON path.")
    parser.add_argument("--device", default=None, help="Device (default: auto).")
    args = parser.parse_args()

    if (args.circuit is None) == (args.prompt is None):
        parser.error("Provide exactly one of --circuit or --prompt.")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model_id} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    if args.circuit is not None:
        print(f"Loading circuit from {args.circuit}...")
        data = CircuitData.load_from_pickle(args.circuit)
        circuit = Circuit(data, tokenizer=tokenizer, num_layers=model.config.num_hidden_layers)
        results = export_top_neurons_from_circuit(
            circuit, tokenizer, top_n=args.top_n, top_tokens=args.top_tokens
        )
    else:
        label = args.target.strip() if args.target else "prediction"
        results = export_top_neurons(
            model,
            tokenizer,
            prompts=[args.prompt],
            labels=[label],
            seed_responses=[args.seed_response],
            targets=[[args.target]] if args.target else None,
            top_n=args.top_n,
            top_tokens=args.top_tokens,
            percentage_threshold=args.percentage_threshold,
            k=args.k,
            device=device,
        )

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(results)} neurons to {args.out}")
    for r in results[:10]:
        ex = r["examples"][0] if r.get("examples") else {}
        top_in = ", ".join(f"{t!r}" for t, _ in ex.get("top_input_tokens", [])[:4])
        print(
            f"  #{r['rank']:>2} L{r['layer']:>2} N{r['neuron']:<6} "
            f"({r['polarity']}, attr={r['attribution']:.3f})  drivers: {top_in}"
        )


if __name__ == "__main__":
    _main()

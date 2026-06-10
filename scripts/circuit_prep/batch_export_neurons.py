"""
Batch export of top-N neurons + text for MANY attribution graphs at once (Gemma 2 2B).

Recommended two-step workflow:

  1. Trace many prompts in one batched (optionally multi-GPU) run with the existing prep.py,
     which writes a single CircuitData pickle covering the whole dataset:

       uv run python scripts/circuit_prep/prep.py --config configs/capitals_gemma.yaml
       # or multi-GPU:
       sbatch --gres=gpu:4 --export=CONFIG=configs/capitals_gemma.yaml scripts/circuit_prep/prep.sbatch

  2. Export per-prompt (per-graph) top-N neurons + text from that pickle. This step only needs
     the tokenizer (no model weights / GPU):

       uv run python scripts/circuit_prep/batch_export_neurons.py \
           --circuit results/case_studies/capitals_gemma_circuit.pkl \
           --model-id google/gemma-2-2b \
           --dataset capitals \
           --top-n 30 --mode per-prompt --out capitals_neurons.json

Modes:
  --mode per-prompt   one top-N list per prompt (separate attribution graph). [default]
  --mode aggregate    one top-N list across the whole dataset.

Output is a JSON list (or JSONL with --jsonl). `--dataset <name>` (a module under
scripts/circuit_prep/data/) is optional and only used to attach the prompt/target text to
each entry; ci_idx aligns with the order prompts were traced in.
"""

import argparse
import importlib.util
import json
from pathlib import Path

from circuits.analysis.circuit_ops import Circuit
from circuits.analysis.neuron_export import (
    export_per_prompt_from_circuit,
    export_top_neurons_from_circuit,
)
from circuits.tracing.trace import CircuitData
from circuits.utils.constants import N_LAYERS_MAPPING


def _load_dataset_text(dataset: str) -> tuple[list[str] | None, list[str] | None]:
    """Load (prompts, labels) from a static data module under scripts/circuit_prep/data/."""
    data_file = Path(__file__).parent / "data" / f"{dataset}.py"
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset module not found: {data_file}")
    spec = importlib.util.spec_from_file_location(f"data.{dataset}", data_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    prompts = getattr(mod, "prompts", None)
    labels = getattr(mod, "labels", None)
    return prompts, labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch export top-N neurons + text from a traced circuit."
    )
    parser.add_argument("--circuit", required=True, help="CircuitData pickle from prep.py.")
    parser.add_argument(
        "--model-id", default="google/gemma-2-2b", help="Model ID (for the tokenizer)."
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional data module name to attach prompt/target text to each entry.",
    )
    parser.add_argument("--top-n", type=int, default=30, help="Neurons per graph.")
    parser.add_argument("--top-tokens", type=int, default=8, help="Top tokens per example.")
    parser.add_argument(
        "--mode", choices=["per-prompt", "aggregate"], default="per-prompt", help="Export mode."
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Override num layers (default: from model-id, else inferred from circuit).",
    )
    parser.add_argument("--out", default="neurons.json", help="Output path.")
    parser.add_argument("--jsonl", action="store_true", help="Write JSONL (one line per graph).")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    print(f"Loading circuit from {args.circuit}...")
    data = CircuitData.load_from_pickle(args.circuit)
    num_layers = args.num_layers or N_LAYERS_MAPPING.get(data.model_id or args.model_id)
    circuit = Circuit(data, tokenizer=tokenizer, num_layers=num_layers)

    prompts = labels = None
    if args.dataset is not None:
        prompts, labels = _load_dataset_text(args.dataset)

    if args.mode == "per-prompt":
        results = export_per_prompt_from_circuit(
            circuit,
            tokenizer,
            top_n=args.top_n,
            top_tokens=args.top_tokens,
            num_layers=num_layers,
            prompts=prompts,
            targets=labels,
        )
        n_graphs = len(results)
    else:
        results = export_top_neurons_from_circuit(
            circuit, tokenizer, top_n=args.top_n, top_tokens=args.top_tokens, num_layers=num_layers
        )
        n_graphs = 1

    out_path = Path(args.out)
    if args.jsonl:
        with open(out_path, "w") as f:
            rows = results if args.mode == "per-prompt" else [results]
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    if args.mode == "per-prompt":
        total_neurons = sum(len(r["neurons"]) for r in results)
        print(f"Wrote {n_graphs} graphs ({total_neurons} neuron entries) to {out_path}")
        for r in results[:5]:
            top = r["neurons"][0] if r["neurons"] else {}
            drivers = ", ".join(repr(t) for t, _ in top.get("top_input_tokens", [])[:3])
            print(
                f"  ci={r['ci_idx']:<3} {r.get('prompt') or r['label']!r:.60}"
                f"  → L{top.get('layer')} N{top.get('neuron')} drivers: {drivers}"
            )
    else:
        print(f"Wrote {len(results)} aggregate neurons to {out_path}")


if __name__ == "__main__":
    main()

"""Generate v2 descriptions for neurons from run_attr.py results.

Generates two sets of descriptions per neuron:
  1. attr-based: highlights tokens by gradient attribution (what caused the neuron to fire)
  2. act-based: highlights tokens by activation magnitude (what co-activates with the neuron)

Usage:
    python scripts/neuron_attr_locality/describe_neurons.py \
        --results /path/to/results.json \
        --model-id meta-llama/Llama-3.1-8B-Instruct \
        --output /path/to/descriptions.json
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from circuits.analysis.cluster import NeuronId
from circuits.descriptions.label import label_clusters
from circuits.utils.constants import RESULTS_DIR
from transformers import AutoTokenizer

sys.stdout.reconfigure(line_buffering=True)


def log(msg: str) -> None:
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {msg}")


def build_df_and_cis(
    neuron_groups: dict[tuple[int, int, str], list[dict]],
    tokenizer: AutoTokenizer,
    view: str,
) -> tuple[pd.DataFrame, list[list[int]]]:
    """Build df_node and cis for label_clusters from grouped results.

    Args:
        view: "attr" to use attr_vec, "act" to use act_vec.
    """
    token_seq_to_ci_idx: dict[tuple[str, ...], int] = {}
    cis: list[list[int]] = []
    df_rows = []

    for (layer, neuron, polarity), entries in neuron_groups.items():
        pol_char = "+" if polarity == "pos" else "-"
        nid = NeuronId(layer=layer, token=-1, neuron=neuron, polarity=pol_char)

        for entry in entries:
            tokens = entry["tokens"]

            if view == "attr":
                vec = list(entry["attr_vec"])
            else:
                vec = list(entry["act_vec"])

            # Mask BOS token (set to 0)
            if len(vec) > 0:
                vec[0] = 0.0

            # Apply polarity sign: for neg, negate so pipeline sees positive values
            if polarity == "neg":
                vec = [-v for v in vec]

            # Register this token sequence as a ci
            tokens_key = tuple(tokens)
            if tokens_key not in token_seq_to_ci_idx:
                ci_idx = len(cis)
                token_seq_to_ci_idx[tokens_key] = ci_idx
                input_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in tokens]
                cis.append(input_ids)
            else:
                ci_idx = token_seq_to_ci_idx[tokens_key]

            df_rows.append(
                {
                    "input_variable": nid,
                    "attr_map": vec,
                    "contrib_map": None,
                    "label": f"ci_idx___{ci_idx}",
                }
            )

    return pd.DataFrame(df_rows), cis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate attr and activation descriptions for neurons."
    )
    parser.add_argument(
        "--results",
        type=str,
        default=str(RESULTS_DIR / "neuron_attr_locality/fineweb/results.json"),
        help="Path to results.json from run_attr.py (must have --detailed)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model ID (for tokenizer)",
    )
    parser.add_argument(
        "--num-expl-samples",
        type=int,
        default=5,
        help="Number of explanations per neuron per sign",
    )
    parser.add_argument("--min-highlights", type=int, default=4, help="Minimum highlighted tokens")
    parser.add_argument(
        "--threshold-mode",
        type=str,
        default="quantile",
        choices=["quantile", "topk"],
        help="Threshold mode for highlighting",
    )
    parser.add_argument("--output", type=str, default="", help="Output path for descriptions JSON")
    parser.add_argument("--gpu-idx", type=int, default=0, help="GPU index for VLLM explainer")
    parser.add_argument(
        "--max-neurons", type=int, default=None, help="Max neurons to describe (for testing)"
    )
    parser.add_argument(
        "--views",
        type=str,
        default="attr,act",
        help="Comma-separated views to generate: attr, act (default: both)",
    )
    args = parser.parse_args()

    views = [v.strip() for v in args.views.split(",")]

    # Load results
    log(f"Loading results from {args.results}")
    with open(args.results) as f:
        results = json.load(f)

    # Check that results have detailed fields
    if not results:
        print("ERROR: results.json is empty")
        return
    if "attr_vec" not in results[0] or "tokens" not in results[0]:
        print("ERROR: results.json missing attr_vec/tokens. Re-run run_attr.py with --detailed")
        return
    if "act" in views and "act_vec" not in results[0]:
        print(
            "ERROR: results.json missing act_vec. Re-run run_attr.py with --detailed (new version)"
        )
        return

    log(f"Loaded {len(results)} result entries")

    # Load tokenizer
    log(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Group results by (layer, neuron, polarity)
    neuron_groups: dict[tuple[int, int, str], list[dict]] = defaultdict(list)
    for r in results:
        key = (r["layer"], r["neuron"], r["polarity"])
        neuron_groups[key].append(r)

    log(f"Found {len(neuron_groups)} unique (layer, neuron, polarity) groups")

    # Generate descriptions for each view
    all_view_results: dict[str, tuple] = {}
    for view in views:
        log(f"\n{'=' * 80}")
        log(f"Generating {view} descriptions")
        log(f"{'=' * 80}")

        df_node, cis = build_df_and_cis(neuron_groups, tokenizer, view)
        log(f"Built df_node with {len(df_node)} rows, {len(cis)} unique input sequences")

        log(f"Running label_clusters ({view}, no scoring)")
        view_results, _, view_exemplars, _ = label_clusters(
            df_node=df_node,
            cis=cis,
            tokenizer=tokenizer,
            target_logits=None,
            num_expl_samples=args.num_expl_samples,
            min_highlights=args.min_highlights,
            threshold_mode=args.threshold_mode,
            score_explanations=False,
            skip_attr=False,
            skip_contrib=True,
            gpu_idx=args.gpu_idx,
            max_neurons=args.max_neurons,
            verbose=True,
        )
        all_view_results[view] = (view_results, view_exemplars)

    # Serialize results
    output_data = {}
    # Collect all neuron ids across views
    all_nids: set[NeuronId] = set()
    for view_results, _ in all_view_results.values():
        all_nids.update(view_results.keys())

    for nid in sorted(all_nids, key=lambda n: (n.layer, n.neuron, n.polarity)):
        key = f"L{nid.layer}:N{nid.neuron}({nid.polarity})"
        entry: dict = {
            "neuron_id": {
                "layer": nid.layer,
                "token": nid.token,
                "neuron": nid.neuron,
                "polarity": nid.polarity,
            },
        }
        for view in views:
            view_results, view_exemplars = all_view_results[view]
            entry[f"{view}_explanations"] = view_results.get(nid, {"pos": [], "neg": []})
            entry[f"{view}_exemplars"] = view_exemplars.get(nid, {"pos": [], "neg": []})
        output_data[key] = entry

    # Print summary
    log(f"\n{'=' * 80}")
    log(f"Generated descriptions for {len(output_data)} neurons")
    log(f"{'=' * 80}")
    for key, data in sorted(output_data.items()):
        for view in views:
            for sign in ("pos", "neg"):
                expls = data[f"{view}_explanations"].get(sign, [])
                if expls:
                    log(f"  {key} [{view}/{sign}]: {len(expls)} explanations")
                    for i, e in enumerate(expls[:3]):
                        log(f"    {i}: {e[:120]}")

    # Save
    if not args.output:
        results_dir = Path(args.results).parent
        args.output = str(results_dir / "descriptions.json")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    log(f"Saved descriptions to {output_path}")


if __name__ == "__main__":
    main()

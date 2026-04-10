"""Steer circuit clusters and print top output probs.

Usage:
    python scripts/case_studies/capitals/steer_circuit.py circuit.pkl --label 10 --multiplier 0
"""

import argparse
import logging

import torch
from circuits.analysis.circuit_ops import Circuit
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("circuit", type=str, help="Path to circuit pickle")
    parser.add_argument(
        "--label", type=str, default="", help="Filter to this label (index or name)"
    )
    parser.add_argument("--multiplier", type=float, default=0.0)
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--cluster-state", type=str, default="", help="Cluster state JSON")
    args = parser.parse_args()

    logger.info("Loading circuit from %s", args.circuit)
    c = Circuit.load_from_pickle(args.circuit)
    t = AutoTokenizer.from_pretrained(args.model_id)
    c.set_tokenizer(t, num_layers=32)

    if args.cluster_state:
        c.load_cluster_state(args.cluster_state)

    import pandas as pd

    # Filter to target label if specified
    lbl = None
    if args.label:
        try:
            ci_idx = int(args.label)
            lbl = c.labels[ci_idx]
        except ValueError:
            lbl = args.label
            ci_idx = next(i for i, l in enumerate(c.labels) if lbl.strip() in str(l))

        old_label = f"{lbl}___{ci_idx}"
        new_label = f"{lbl}___0"

        n_before = len(c.df_node)
        c.df_node = c.df_node[c.df_node["label"] == old_label].copy()
        c.df_node["label"] = new_label
        c.labels = [lbl]
        c.cis = [c.cis[ci_idx]]
        c.attention_masks = [c.attention_masks[ci_idx]] if c.attention_masks else []
        c.target_logits = [c.target_logits[ci_idx]] if c.target_logits else []
        logger.info("Filtered to %s: %d -> %d nodes", old_label, n_before, len(c.df_node))

    # Drop edges (not needed for steering, saves memory + time)
    c.df_edge = pd.DataFrame(columns=c.df_edge.columns)

    logger.info("Loading model %s...", args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Ensure df_node has input_variable column (needed by steer)
    from circuits.analysis.cluster import prepare_circuit_data

    if "input_variable" not in c.df_node.columns:
        logger.info("Preparing circuit data (adding input_variable)...")
        c.df_node, c.df_edge = prepare_circuit_data(c.df_node, c.df_edge)

    logger.info("Steering at multiplier=%.1f...", args.multiplier)
    c.steer(model, multiplier=args.multiplier, verbose=True, store_results=True)

    cto = c.cluster_to_output
    cluster_id_to_name = getattr(c, "_cluster_id_to_name", {})

    # Filter output to target label
    if lbl:
        cto = cto[cto["label"].str.contains(lbl.strip(), regex=False)]

    print(f"\n{'=' * 80}")
    print(f"STEERING AT x{args.multiplier} — top {args.top_k} output probs per cluster")
    print(f"{'=' * 80}")

    for _, row in cto.iterrows():
        cl = row["cluster"]
        cl_name = cluster_id_to_name.get(cl, f"C{cl}")
        label = row["label"]
        top_toks = row["top_tokens"][: args.top_k]
        top_probs = row["top_tokens_probs"][: args.top_k]
        print(f"\n{cl_name} (C{cl}) — {label}:")
        for tok, prob in zip(top_toks, top_probs):
            print(f"  {tok:>15}  {prob:.2%}")


if __name__ == "__main__":
    main()

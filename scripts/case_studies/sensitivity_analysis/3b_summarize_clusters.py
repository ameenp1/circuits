"""Generate summary labels for clusters using rich or batch mode.

Standalone script that loads a circuit + cluster state, generates summary labels,
and saves the updated cluster state.

Usage:
    python scripts/case_studies/sensitivity_analysis/3b_summarize_clusters.py
    python scripts/case_studies/sensitivity_analysis/3b_summarize_clusters.py --mode batch
"""

import argparse
import logging

from circuits.analysis.circuit_ops import Circuit
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CIRCUIT_PICKLE = "results/case_studies/sensitivity_analysis_circuit.pkl"
CLUSTER_STATE = "results/case_studies/sensitivity_analysis/cluster_state_20260323_131824_mv_k20_harmonic.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuit", default=CIRCUIT_PICKLE)
    parser.add_argument("--cluster-state", default=CLUSTER_STATE)
    parser.add_argument(
        "--mode",
        default="rich",
        choices=["rich", "batch"],
        help="'rich' (per-cluster with neuron descs) or 'batch' (all clusters in one prompt)",
    )
    parser.add_argument(
        "--summary-model",
        default="claude-opus-4-6",
        help="Anthropic model for summarization",
    )
    parser.add_argument("--max-neurons", type=int, default=20)
    parser.add_argument(
        "--neurons-only", action="store_true", help="Only use neuron descs, skip attr/contrib"
    )
    parser.add_argument(
        "--attr-only", action="store_true", help="Only use attr/contrib descs, skip neuron descs"
    )
    parser.add_argument(
        "--top-k-examples", type=int, default=0, help="Include top K dataset examples per cluster"
    )
    args = parser.parse_args()

    logger.info("Loading circuit from %s", args.circuit)
    c = Circuit.load_from_pickle(args.circuit)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    c.set_tokenizer(tokenizer, num_layers=32)
    c.load_cluster_state(args.cluster_state)

    if args.mode == "rich":
        logger.info("Fetching per-neuron descriptions...")
        c.fetch_descriptions()

    logger.info("Generating summary labels (mode=%s, model=%s)...", args.mode, args.summary_model)
    labels = c.summarize_clusters(
        mode=args.mode,
        summary_model=args.summary_model,
        max_neurons_per_cluster=args.max_neurons,
        neurons_only=args.neurons_only,
        attr_only=args.attr_only,
        top_k_examples=args.top_k_examples,
    )

    for name, lbl in sorted(labels.items(), key=lambda x: int(x[0].replace("C", ""))):
        logger.info("  %s -> %s", name, lbl)

    c.save_cluster_state(args.cluster_state)
    logger.info("Saved updated cluster state to %s", args.cluster_state)


if __name__ == "__main__":
    main()

"""Cluster, generate descriptions, and batch-summarize the user modelling circuit.

Usage:
    python scripts/case_studies/user_modelling/describe_and_summarize.py
    python scripts/case_studies/user_modelling/describe_and_summarize.py --n-clusters 32
    python scripts/case_studies/user_modelling/describe_and_summarize.py --skip-descriptions
"""

import argparse
import json
import logging
from datetime import datetime

from circuits.analysis.circuit_ops import Circuit
from circuits.utils.constants import RESULTS_DIR
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CIRCUIT_PICKLE = RESULTS_DIR / "case_studies/user_modelling/wikipedia_country_circuit.pkl"
OUTPUT_DIR = RESULTS_DIR / "case_studies/user_modelling"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuit", type=str, default=str(CIRCUIT_PICKLE))
    parser.add_argument("--n-clusters", type=int, default=32)
    parser.add_argument("--combine", type=str, default="harmonic")
    parser.add_argument("--num-expl-samples", type=int, default=1)
    parser.add_argument("--no-score", action="store_true")
    parser.add_argument(
        "--skip-descriptions",
        action="store_true",
        help="Skip description generation, only rerun summary labels",
    )
    parser.add_argument("--top-k-examples", type=int, default=10)
    parser.add_argument("--summary-model", type=str, default="claude-opus-4-6")
    args = parser.parse_args()

    logger.info("Loading circuit from %s", args.circuit)
    circuit = Circuit.load_from_pickle(args.circuit)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    circuit.set_tokenizer(tokenizer, num_layers=32)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    canonical_state = OUTPUT_DIR / f"cluster_state_k{args.n_clusters}_{args.combine}.json"

    if not args.skip_descriptions:
        # Step 1: Cluster
        logger.info("Clustering (k=%d, combine=%s)...", args.n_clusters, args.combine)
        circuit.cluster_multiview(
            n_clusters=args.n_clusters,
            combine=args.combine,
            get_desc=False,
            verbose=True,
        )

        # Step 2: Generate descriptions
        logger.info(
            "Generating descriptions (samples=%d, score=%s)...",
            args.num_expl_samples,
            not args.no_score,
        )
        result = circuit.label_clusters_simulator_v2(
            score_explanations=not args.no_score,
            num_expl_samples=args.num_expl_samples,
            min_highlights=1,
            threshold_mode="quantile",
            contrib_model_name="claude-haiku-4-5-20251001",
            verbose=True,
        )
        attr_results, contrib_results, _, _, _ = result

        # Save descriptions JSON
        cluster_id_to_name: dict[int, str] = getattr(circuit, "_cluster_id_to_name", {})

        def neuron_key_to_name(neuron_key: object) -> str:
            cid = neuron_key.layer if hasattr(neuron_key, "layer") else neuron_key
            return cluster_id_to_name.get(cid, f"C{cid}")

        output_data: dict = {
            "metadata": {
                "circuit_pickle": str(args.circuit),
                "timestamp": timestamp,
                "scored": not args.no_score,
                "n_clusters": args.n_clusters,
                "combine": args.combine,
                "num_expl_samples": args.num_expl_samples,
            },
            "cluster_id_to_name": {str(k): v for k, v in cluster_id_to_name.items()},
            "attr": {},
            "contrib": {},
        }

        for results, key in [(attr_results, "attr"), (contrib_results, "contrib")]:
            for neuron_key, sign_data in results.items():
                name = neuron_key_to_name(neuron_key)
                output_data[key][name] = {}
                for sign in ["pos", "neg", "combined"]:
                    output_data[key][name][sign] = [
                        (
                            {"explanation": expl.explanation, "score": expl.score}
                            if hasattr(expl, "explanation")
                            else {"explanation": str(expl), "score": None}
                        )
                        for expl in sign_data.get(sign, [])
                    ]

        desc_out = OUTPUT_DIR / f"descriptions_{timestamp}_k{args.n_clusters}.json"
        with open(desc_out, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        logger.info("Saved descriptions to %s", desc_out)

        # Save intermediate cluster state (before summary)
        circuit.save_cluster_state(canonical_state)
    else:
        logger.info("Loading cluster state from %s", canonical_state)
        circuit.load_cluster_state(str(canonical_state))

    # Step 3: Batch summary with exemplars
    logger.info(
        "Generating batch summary labels (model=%s, top_k_examples=%d)...",
        args.summary_model,
        args.top_k_examples,
    )
    labels = circuit.summarize_clusters(
        mode="batch",
        summary_model=args.summary_model,
        top_k_examples=args.top_k_examples,
    )
    for name, lbl in sorted(labels.items()):
        logger.info("  %s -> %s", name, lbl)

    # Save final cluster state
    circuit.save_cluster_state(canonical_state)
    logger.info("Cluster state saved to %s", canonical_state)


if __name__ == "__main__":
    main()

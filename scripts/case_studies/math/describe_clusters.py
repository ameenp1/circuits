"""Generate descriptions and batch summary labels for math circuit clusters.

Uses 1 sample, no scoring for descriptions, then batch summary with exemplars (opus).

Usage:
    python scripts/case_studies/math/describe_clusters.py \
        --cluster-state /path/to/cluster_state.json
    python scripts/case_studies/math/describe_clusters.py \
        --cluster-state /path/to/cluster_state.json --skip-descriptions
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from circuits.analysis.circuit_ops import Circuit
from circuits.utils.constants import RESULTS_DIR
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CIRCUIT_PICKLE = RESULTS_DIR / "case_studies/math_circuit.pkl"
OUTPUT_DIR = RESULTS_DIR / "case_studies/math"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuit", type=str, default=str(CIRCUIT_PICKLE))
    parser.add_argument("--cluster-state", type=str, required=True)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument(
        "--skip-descriptions",
        action="store_true",
        help="Skip description generation, only rerun summary labels",
    )
    parser.add_argument("--top-k-examples", type=int, default=10)
    parser.add_argument(
        "--summary-model",
        type=str,
        default="claude-opus-4-6",
    )
    args = parser.parse_args()

    logger.info("Loading circuit from %s", args.circuit)
    circuit = Circuit.load_from_pickle(args.circuit)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    circuit.set_tokenizer(tokenizer, num_layers=32)

    logger.info("Loading cluster state from %s", args.cluster_state)
    circuit.load_cluster_state(args.cluster_state)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cluster_id_to_name: dict[int, str] = getattr(circuit, "_cluster_id_to_name", {})

    if not args.skip_descriptions:
        logger.info("Generating descriptions (1 sample, no scoring)...")
        result = circuit.label_clusters_simulator_v2(
            score_explanations=False,
            num_expl_samples=1,
            min_highlights=1,
            threshold_mode="quantile",
            contrib_model_name="claude-haiku-4-5-20251001",
            verbose=True,
        )
        attr_results, contrib_results, _, _, _ = result

        # Save descriptions JSON
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        desc_out = (
            Path(args.output) if args.output else OUTPUT_DIR / f"descriptions_{timestamp}.json"
        )

        def neuron_key_to_name(neuron_key: object) -> str:
            cid = neuron_key.layer if hasattr(neuron_key, "layer") else neuron_key
            return cluster_id_to_name.get(cid, f"C{cid}")

        output_data: dict = {
            "metadata": {
                "circuit_pickle": str(args.circuit),
                "cluster_state": args.cluster_state,
                "timestamp": timestamp,
                "scored": False,
                "num_expl_samples": 1,
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

        with open(desc_out, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        logger.info("Saved descriptions to %s", desc_out)

    # Generate batch summary labels with exemplars
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

    # Save updated cluster state
    circuit.save_cluster_state(args.cluster_state)
    logger.info("Updated cluster state at %s", args.cluster_state)


if __name__ == "__main__":
    main()

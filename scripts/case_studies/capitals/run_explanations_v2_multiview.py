"""
Cluster capitals circuit with multi-view spectral clustering, generate v2 descriptions,
and visualize.

Usage:
    python scripts/case_studies/capitals/run_explanations_v2_multiview.py
    python scripts/case_studies/capitals/run_explanations_v2_multiview.py --skip-attr
    python scripts/case_studies/capitals/run_explanations_v2_multiview.py --n-clusters 16
"""

import argparse
import json
import logging
from datetime import datetime

from circuits.analysis.circuit_ops import Circuit
from circuits.utils.constants import RESULTS_DIR
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

CIRCUIT_PICKLE = RESULTS_DIR / "case_studies/capitals_circuit.pkl"
OUTPUT_DIR = RESULTS_DIR / "case_studies/capitals/explanations_v2_multiview"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-clusters", type=int, default=64)
    parser.add_argument("--no-score", action="store_true")
    parser.add_argument("--skip-attr", action="store_true")
    parser.add_argument("--num-expl-samples", type=int, default=5)
    parser.add_argument("--min-highlights", type=int, default=1)
    parser.add_argument(
        "--threshold-mode", type=str, default="quantile", choices=["topk", "quantile"]
    )
    parser.add_argument("--combine", type=str, default="harmonic")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Step 1: Load and cluster
    logger.info("Loading circuit from %s", CIRCUIT_PICKLE)
    circuit = Circuit.load_from_pickle(str(CIRCUIT_PICKLE))

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    circuit.set_tokenizer(tokenizer, num_layers=32)

    logger.info("Clustering with multi-view (k=%d, combine=%s)...", args.n_clusters, args.combine)
    circuit.cluster_multiview(
        n_clusters=args.n_clusters,
        combine=args.combine,
        get_desc=False,
        verbose=True,
    )

    # Step 2: Generate v2 descriptions
    logger.info("Generating v2 descriptions (skip_attr=%s)...", args.skip_attr)
    result = circuit.label_clusters_simulator_v2(
        score_explanations=not args.no_score,
        num_expl_samples=args.num_expl_samples,
        min_highlights=args.min_highlights,
        threshold_mode=args.threshold_mode,
        skip_attr=args.skip_attr,
        contrib_model_name="claude-haiku-4-5-20251001",
        verbose=True,
    )
    attr_results, contrib_results, attr_exemplars, contrib_exemplars, cluster_to_neurons = result

    cluster_id_to_name: dict[int, str] = getattr(circuit, "_cluster_id_to_name", {})

    # Print summary
    print("\n" + "=" * 80)
    print("DESCRIPTIONS_V2 MULTIVIEW RESULTS")
    print("=" * 80)

    for results_dict, label in [(attr_results, "attr"), (contrib_results, "contrib")]:
        for neuron_key, sign_data in sorted(results_dict.items(), key=lambda x: str(x[0])):
            cid = neuron_key.layer if hasattr(neuron_key, "layer") else neuron_key
            name = cluster_id_to_name.get(cid, f"C{cid}")
            print(f"\n--- {name} (C{cid}) ---")
            for sign in ["pos", "neg", "combined"]:
                for expl in sign_data.get(sign, []):
                    if hasattr(expl, "explanation"):
                        score = f"{expl.score:.3f}" if expl.score is not None else "N/A"
                        print(f"  {label}/{sign}: [{score}] {expl.explanation}")

    # Step 3: Save JSON for visualize_explanations
    suffix = f"mv_k{args.n_clusters}_{args.combine}"
    if args.skip_attr:
        suffix += "_noattr"
    output_file = OUTPUT_DIR / f"explanations_v2_{timestamp}_{suffix}.json"

    def neuron_key_to_name(neuron_key: object) -> str:
        cid = neuron_key.layer if hasattr(neuron_key, "layer") else neuron_key
        return cluster_id_to_name.get(cid, f"C{cid}")

    output_data: dict = {
        "metadata": {
            "circuit_pickle": str(CIRCUIT_PICKLE),
            "timestamp": timestamp,
            "scored": not args.no_score,
            "skip_attr": args.skip_attr,
            "n_clusters": args.n_clusters,
            "combine": args.combine,
            "num_expl_samples": args.num_expl_samples,
            "min_highlights": args.min_highlights,
            "backend": "descriptions",
        },
        "cluster_id_to_name": {str(k): v for k, v in cluster_id_to_name.items()},
        "attr": {},
        "contrib": {},
        "attr_exemplars": {},
        "contrib_exemplars": {},
    }

    for results_dict, key in [(attr_results, "attr"), (contrib_results, "contrib")]:
        for neuron_key, sign_data in results_dict.items():
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

    for exemplars, key in [
        (attr_exemplars, "attr_exemplars"),
        (contrib_exemplars, "contrib_exemplars"),
    ]:
        for neuron_key, sign_exemplars in exemplars.items():
            name = neuron_key_to_name(neuron_key)
            output_data[key][name] = dict(sign_exemplars)

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    logger.info("Results saved to %s", output_file)

    # Step 4: Generate summary labels (rich mode, attr-only, opus)
    logger.info("Generating rich summary labels (attr_only=True)...")
    summary_labels = circuit.summarize_clusters(mode="rich", attr_only=True)
    for name, lbl in sorted(summary_labels.items()):
        logger.info("  %s -> %s", name, lbl)

    # Step 5: Save cluster state (lightweight JSON with assignments + descriptions + labels)
    cluster_state_out = OUTPUT_DIR / f"cluster_state_{timestamp}_{suffix}.json"
    circuit.save_cluster_state(cluster_state_out)

    # Also save to the canonical location
    canonical_state = (
        RESULTS_DIR / f"case_studies/capitals/cluster_state_k{args.n_clusters}_{args.combine}.json"
    )
    circuit.save_cluster_state(canonical_state)
    logger.info("Cluster state saved to %s and %s", cluster_state_out, canonical_state)

    print(f"\nVisualize with:\n  python -m circuits.analysis.visualize_explanations {output_file}")
    print(
        f"Plot circuit graph:\n  python scripts/case_studies/capitals/plot_circuit_graph.py"
        f" {CIRCUIT_PICKLE} --cluster-state {canonical_state} --label 0"
    )


if __name__ == "__main__":
    main()

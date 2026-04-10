"""Analyse multi-view spectral clusters with neuron descriptions.

Runs multi-view spectral clustering at a chosen k, fetches neuron descriptions,
and prints per-cluster summaries.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

from circuits.analysis.circuit_ops import Circuit
from circuits.analysis.cluster import UNCLUSTERED_CLUSTER_ID
from circuits.utils.constants import RESULTS_DIR
from transformers import AutoTokenizer


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--circuit-pickle",
        type=str,
        default=str(RESULTS_DIR / "case_studies/capitals_circuit.pkl"),
    )
    parser.add_argument("--n-clusters", type=int, default=16)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-14B")
    parser.add_argument(
        "--output",
        type=str,
        default=str(RESULTS_DIR / "case_studies/multiview_cluster_descriptions.json"),
    )
    parser.add_argument(
        "--weight-by-attribution",
        action="store_true",
        help="Weight CI contributions by geometric mean of neuron attributions",
    )
    parser.add_argument(
        "--use-attributions-as-view",
        action="store_true",
        help="Add attribution importance profile as an additional similarity view",
    )
    args = parser.parse_args()

    print(f"Loading circuit from {args.circuit_pickle}")
    circuit = Circuit.load_from_pickle(args.circuit_pickle)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    circuit.set_tokenizer(tokenizer)

    # Cluster using circuit API
    print(f"Running multi-view spectral clustering with k={args.n_clusters}...")
    circuit.cluster_multiview(
        n_clusters=args.n_clusters,
        get_desc=True,
        weight_by_attribution=args.weight_by_attribution,
        use_attributions_as_view=args.use_attributions_as_view,
        verbose=True,
    )

    # Analyse clusters from _cluster_map
    # Group NeuronIds by cluster
    cluster_to_nids: dict[str, list] = defaultdict(list)
    for nid, cl in circuit._cluster_map.items():
        cluster_to_nids[cl].append(nid)

    print(f"\n{'=' * 80}")
    print(f"CLUSTER ANALYSIS (k={args.n_clusters})")
    print(f"{'=' * 80}")

    cluster_data = []
    for cluster_id in sorted(cluster_to_nids.keys()):
        if cluster_id == UNCLUSTERED_CLUSTER_ID:
            continue
        if not cluster_id.startswith("C"):
            continue

        nids = cluster_to_nids[cluster_id]
        # Deduplicate by (layer, neuron, polarity)
        unique_keys = set()
        unique_nids = []
        for nid in nids:
            key = (nid.layer, nid.neuron, nid.polarity)
            if key not in unique_keys:
                unique_keys.add(key)
                unique_nids.append(nid)

        # Get descriptions
        descs = []
        for nid in unique_nids:
            desc = circuit.neuron_label_cache.get((nid.layer, nid.neuron), "")
            if desc and desc != "N.A." and desc != "?":
                descs.append(f"  L{nid.layer}:N{nid.neuron}({nid.polarity}): {desc}")

        descs = sorted(set(descs))

        # Layer distribution
        layer_counts = Counter(nid.layer for nid in unique_nids)

        print(f"\n--- {cluster_id} ({len(unique_nids)} neurons) ---")
        print(f"  Layers: {dict(sorted(layer_counts.items()))}")
        for d in descs[:20]:
            print(d)
        if len(descs) > 20:
            print(f"  ... and {len(descs) - 20} more")

        cluster_data.append(
            {
                "cluster": cluster_id,
                "n_neurons": len(unique_nids),
                "layer_dist": dict(sorted(layer_counts.items())),
                "descriptions": descs,
            }
        )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(cluster_data, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()

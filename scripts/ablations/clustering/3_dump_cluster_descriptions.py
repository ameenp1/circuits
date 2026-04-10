"""Dump cluster -> neuron description mappings for multi-view spectral clustering."""

import argparse
from collections import defaultdict

from circuits.analysis.circuit_ops import Circuit
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--circuit-pickle",
        type=str,
        default="results/case_studies/capitals_circuit.pkl",
    )
    parser.add_argument("--n-clusters", type=int, default=64)
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument(
        "--combine",
        type=str,
        default="mean",
        choices=["mean", "harmonic"],
        help="How to combine attr/contrib similarities",
    )
    args = parser.parse_args()

    circuit = Circuit.load_from_pickle(args.circuit_pickle)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    circuit.set_tokenizer(tokenizer)

    circuit.cluster_multiview(
        n_clusters=args.n_clusters,
        get_desc=True,
        combine=args.combine,
        verbose=True,
    )

    # Build cluster -> (neuron_key, description) from _cluster_map + neuron_label_cache
    cluster_descs: dict[str, dict[str, str]] = defaultdict(dict)
    for nid, cl in circuit._cluster_map.items():
        if not str(cl).startswith("C"):
            continue
        desc = circuit.neuron_label_cache.get((nid.layer, nid.neuron), "")
        key = f"L{nid.layer}:N{nid.neuron}({nid.polarity})"
        if key not in cluster_descs[cl]:
            cluster_descs[cl][key] = desc if desc else "(none)"

    lines = []
    for cl in sorted(cluster_descs.keys(), key=lambda x: int(x[1:])):
        neurons = cluster_descs[cl]
        lines.append(f"\n=== {cl} ({len(neurons)} neurons) ===")
        for key, desc in neurons.items():
            d = desc[:120] if desc else "(none)"
            lines.append(f"  {key}: {d}")

    output = "\n".join(lines)
    print(output)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

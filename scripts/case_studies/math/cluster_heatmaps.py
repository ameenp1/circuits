"""Cluster the math circuit and plot summed attribution heatmaps per cluster.

For each cluster, sums the scalar attribution across all neurons in the cluster
for each example (x + y pair), then reshapes into a 100x100 heatmap.

Usage:
    python scripts/case_studies/math/cluster_heatmaps.py
    python scripts/case_studies/math/cluster_heatmaps.py --n-clusters 16
    python scripts/case_studies/math/cluster_heatmaps.py --circuit /path/to/circuit.pkl
"""

import argparse
import logging
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from circuits.analysis.circuit_ops import Circuit
from circuits.utils.constants import RESULTS_DIR
from transformers import AutoTokenizer

matplotlib.use("Agg")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CIRCUIT_PICKLE = RESULTS_DIR / "case_studies/math_circuit.pkl"
OUTPUT_DIR = RESULTS_DIR / "case_studies/math"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuit", type=str, default=str(CIRCUIT_PICKLE))
    parser.add_argument("--cluster-state", type=str, default="", help="Cluster state JSON")
    parser.add_argument("--n-clusters", type=int, default=256)
    parser.add_argument("--combine", type=str, default="harmonic")
    parser.add_argument("--clusters", type=str, nargs="*", default=None, help="Only these clusters")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    logger.info("Loading circuit from %s", args.circuit)
    circuit = Circuit.load_from_pickle(args.circuit)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    circuit.set_tokenizer(tokenizer, num_layers=32)

    if args.cluster_state:
        circuit.load_cluster_state(args.cluster_state)
    elif circuit._cluster_map:
        logger.info("Using existing cluster assignments from pickle")
    else:
        logger.info(
            "Clustering with multiview (k=%d, combine=%s)...", args.n_clusters, args.combine
        )
        circuit.cluster_multiview(
            n_clusters=args.n_clusters,
            combine=args.combine,
            get_desc=False,
            verbose=True,
        )

    cluster_map = circuit._cluster_map
    cluster_id_to_name = getattr(circuit, "_cluster_id_to_name", {})
    num_layers = circuit.num_layers or 32

    # Map (layer, neuron) -> cluster
    neuron_to_cluster: dict[tuple[int, int], str] = {}
    for nid, cl in cluster_map.items():
        neuron_to_cluster[(int(nid.layer), int(nid.neuron))] = cl

    # Only include clusters from cluster_id_to_name (the real spectral clusters)
    valid_cluster_names = set(cluster_id_to_name.values())
    active_clusters = sorted(
        valid_cluster_names & set(neuron_to_cluster.values()),
        key=lambda c: int(c[1:]) if c[1:].isdigit() else c,
    )
    if args.clusters:
        active_clusters = [c for c in active_clusters if c in args.clusters]
    logger.info("Found %d active clusters", len(active_clusters))

    # Build label -> example index mapping
    # Labels are "x + y = z" format, examples are x*100+y order
    label_to_idx: dict[str, int] = {}
    for i, lbl in enumerate(circuit.labels):
        label_to_idx[lbl] = i

    n_examples = len(circuit.labels)
    logger.info("Circuit has %d examples", n_examples)

    # For each cluster, accumulate attribution per example
    cluster_attr: dict[str, np.ndarray] = {cl: np.zeros(n_examples) for cl in active_clusters}

    logger.info("Aggregating attributions per cluster...")
    for _, row in circuit.df_node.iterrows():
        layer = int(row["layer"])
        neuron = int(row["neuron"])
        if not (0 <= layer < num_layers):
            continue
        cl = neuron_to_cluster.get((layer, neuron), "-1")
        if cl not in cluster_attr:
            continue

        label = row["label"]
        # Labels have "___N" suffix encoding the example index
        if "___" in str(label):
            idx = int(str(label).rsplit("___", 1)[1])
        elif label in label_to_idx:
            idx = label_to_idx[label]
        else:
            continue

        if idx < n_examples:
            cluster_attr[cl][idx] += float(row["attribution"])

    # Determine grid size: sqrt(n_examples) if it's a perfect square
    grid_size = int(math.isqrt(n_examples))
    if grid_size * grid_size != n_examples:
        logger.warning(
            "n_examples=%d is not a perfect square, using %dx%d grid (truncating)",
            n_examples,
            grid_size,
            grid_size,
        )

    # Plot heatmaps
    n_clusters = len(active_clusters)
    ncols = min(7, n_clusters)
    nrows = math.ceil(n_clusters / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 1.6, nrows * 1.6),
        squeeze=False,
    )

    for i, cl_name in enumerate(active_clusters):
        ax = axes[i // ncols][i % ncols]
        heatmap = cluster_attr[cl_name][: grid_size * grid_size].reshape(grid_size, grid_size)

        # Diverging colormap centered at 0
        vmax = max(abs(heatmap.max()), abs(heatmap.min()), 1e-8)
        im = ax.imshow(
            heatmap,
            cmap="RdBu",
            vmin=-vmax,
            vmax=vmax,
            origin="lower",
            aspect="equal",
        )

        summary_labels = getattr(circuit, "_cluster_summary_labels", {})
        desc = summary_labels.get(cl_name, "")
        display_name = f"{cl_name}: {desc}" if desc else cl_name
        ax.set_title(display_name, fontsize=7, pad=2)
        ax.set_xlabel("y", fontsize=6)
        ax.set_ylabel("x", fontsize=6)
        ax.tick_params(labelsize=4)

    # Hide unused axes
    for i in range(n_clusters, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    fig.suptitle(
        f"Summed attribution per cluster (k={len(active_clusters)})",
        fontsize=10,
        y=1.01,
    )
    plt.tight_layout()

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = OUTPUT_DIR / f"cluster_heatmaps_k{len(active_clusters)}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    logger.info("Saved to %s", out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()

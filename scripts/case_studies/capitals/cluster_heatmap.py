"""Plot cluster × CI attribution heatmap for the capitals circuit.

For each cluster, sums the scalar attribution across all neurons in that cluster
for each CI (example), producing a (n_clusters × n_examples) heatmap.

Usage:
    python scripts/case_studies/capitals/cluster_heatmap.py
    python scripts/case_studies/capitals/cluster_heatmap.py --n-clusters 64
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as p9
from circuits.analysis.circuit_ops import Circuit
from circuits.utils.constants import RESULTS_DIR
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CIRCUIT_PICKLE = RESULTS_DIR / "case_studies/capitals_circuit.pkl"
OUTPUT_DIR = RESULTS_DIR / "case_studies/capitals"

p9.theme_set(
    p9.theme_bw(base_size=10, base_family="P052")
    + p9.theme(
        text=p9.element_text(color="#000"),
        axis_title=p9.element_text(size=8),
        axis_text=p9.element_text(size=5),
        axis_text_x=p9.element_text(rotation=90, ha="center", size=4),
        axis_text_y=p9.element_text(size=4),
        panel_grid_major=p9.element_blank(),
        panel_grid_minor=p9.element_blank(),
        legend_position="bottom",
        legend_title=p9.element_text(size=7),
        legend_text=p9.element_text(size=6),
        legend_key_height=6,
        legend_key_width=40,
    )
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuit", type=str, default=str(CIRCUIT_PICKLE))
    parser.add_argument("--n-clusters", type=int, default=64)
    parser.add_argument("--combine", type=str, default="mean")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-14B")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    logger.info("Loading circuit from %s", args.circuit)
    circuit = Circuit.load_from_pickle(args.circuit)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    circuit.set_tokenizer(tokenizer)

    logger.info("Clustering with multiview (k=%d, combine=%s)...", args.n_clusters, args.combine)
    circuit.cluster_multiview(
        n_clusters=args.n_clusters,
        combine=args.combine,
        get_desc=False,
        verbose=True,
    )

    cluster_map = circuit._cluster_map
    cluster_id_to_name = getattr(circuit, "_cluster_id_to_name", {})

    # Map (layer, neuron) -> cluster name
    neuron_to_cluster: dict[tuple[int, int], str] = {}
    for nid, cl in cluster_map.items():
        neuron_to_cluster[(int(nid.layer), int(nid.neuron))] = cl

    valid_cluster_names = set(cluster_id_to_name.values())
    active_clusters = sorted(
        valid_cluster_names & set(neuron_to_cluster.values()),
        key=lambda c: int(c[1:]) if c.startswith("C") else c,
    )
    logger.info("Found %d active clusters", len(active_clusters))

    # Build label -> index mapping
    label_to_idx: dict[str, int] = {}
    for i, lbl in enumerate(circuit.labels):
        label_to_idx[lbl] = i

    n_examples = len(circuit.labels)
    n_clusters = len(active_clusters)
    cluster_to_row = {cl: i for i, cl in enumerate(active_clusters)}

    # Accumulate attribution: (n_clusters, n_examples)
    attr_matrix = np.zeros((n_clusters, n_examples))

    logger.info("Aggregating attributions per cluster...")
    for _, row in circuit.df_node.iterrows():
        layer = int(row["layer"])
        neuron = int(row["neuron"])
        cl = neuron_to_cluster.get((layer, neuron))
        if cl is None or cl not in cluster_to_row:
            continue

        label = row["label"]
        if "___" in str(label):
            idx = int(str(label).rsplit("___", 1)[1])
        elif label in label_to_idx:
            idx = label_to_idx[label]
        else:
            continue

        if idx < n_examples:
            attr_matrix[cluster_to_row[cl], idx] += float(row["attribution"])

    # Build long-form DataFrame for plotnine
    rows = []
    ci_labels = [lbl.strip() for lbl in circuit.labels]
    for i, cl in enumerate(active_clusters):
        for j in range(n_examples):
            rows.append({"cluster": cl, "ci": ci_labels[j], "attr": attr_matrix[i, j]})
    df = pd.DataFrame(rows)

    # Preserve ordering via categorical
    df["cluster"] = pd.Categorical(df["cluster"], categories=active_clusters, ordered=True)
    df["ci"] = pd.Categorical(df["ci"], categories=ci_labels, ordered=True)

    vmax = max(abs(attr_matrix.max()), abs(attr_matrix.min()), 1e-8)

    plot = (
        p9.ggplot(df, p9.aes(x="ci", y="cluster", fill="attr"))
        + p9.geom_tile()
        + p9.scale_fill_gradient2(
            low="#2166ac",
            mid="white",
            high="#b2182b",
            midpoint=0,
            limits=(-vmax, vmax),
            name="Summed attr.",
        )
        + p9.scale_y_discrete(breaks=active_clusters[::4])
        + p9.labs(x="CI (example)", y="Cluster")
        + p9.theme(figure_size=(3.3, 4.0))
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = OUTPUT_DIR / f"cluster_ci_heatmap_k{n_clusters}.pdf"
    plot.save(out_path, dpi=300)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()

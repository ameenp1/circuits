"""Generate interactive Plotly heatmap of multi-view similarity matrix.

Neurons are ordered by cluster assignment. Hover shows neuron ID + description.
Outputs a self-contained HTML file.
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go  # type: ignore
from circuits.analysis.circuit_ops import Circuit


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--circuit-pickle",
        type=str,
        default="results/case_studies/capitals_circuit.pkl",
    )
    parser.add_argument("--n-clusters", type=int, default=16)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-14B")
    parser.add_argument(
        "--output",
        type=str,
        default="results/case_studies/multiview_sim_heatmap.html",
    )
    parser.add_argument(
        "--max-neurons",
        type=int,
        default=0,
        help="Subsample to this many neurons (0 = all). Useful if browser is slow.",
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

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    circuit.set_tokenizer(tokenizer)

    # Cluster and store sim matrix on circuit object
    print(f"Running multi-view spectral clustering with k={args.n_clusters}...")
    circuit.cluster_multiview(
        n_clusters=args.n_clusters,
        get_desc=True,
        weight_by_attribution=args.weight_by_attribution,
        use_attributions_as_view=args.use_attributions_as_view,
        verbose=True,
    )

    # Access stored multiview results
    sim_filtered = circuit._mv_sim_matrix
    neuron_ids_filtered = circuit._mv_neuron_ids
    cluster_labels = circuit._mv_cluster_labels
    overlap_filtered = circuit._mv_overlap_counts
    n = len(neuron_ids_filtered)
    print(f"Similarity matrix: {sim_filtered.shape}")

    # Build NeuronId -> description mapping from neuron_label_cache
    nid_to_desc: dict[str, str] = {}
    for nid in neuron_ids_filtered:
        desc = circuit.neuron_label_cache.get((nid.layer, nid.neuron), "")
        if not desc or desc in ("N.A.", "?"):
            desc = "(no description)"
        key = f"L{nid.layer}:N{nid.neuron}({nid.polarity})"
        if key not in nid_to_desc:
            nid_to_desc[key] = desc

    # Sort neurons by cluster, then by layer
    order = sorted(range(n), key=lambda i: (cluster_labels[i], neuron_ids_filtered[i].layer))

    if args.max_neurons > 0 and args.max_neurons < n:
        # Subsample: take proportional samples from each cluster
        from collections import Counter

        counts = Counter(cluster_labels)
        sampled_indices: list[int] = []
        for cl in sorted(counts.keys()):
            cl_indices = [i for i in order if cluster_labels[i] == cl]
            n_sample = max(1, int(len(cl_indices) * args.max_neurons / n))
            step = max(1, len(cl_indices) // n_sample)
            sampled_indices.extend(cl_indices[::step][:n_sample])
        order = sampled_indices
        print(f"Subsampled to {len(order)} neurons")

    sim_ordered = sim_filtered[np.ix_(order, order)]
    overlap_ordered = overlap_filtered[np.ix_(order, order)]
    nids_ordered = [neuron_ids_filtered[i] for i in order]
    clusters_ordered = [int(cluster_labels[i]) for i in order]

    # Build hover text and tick labels
    tick_labels = []
    hover_text = []
    for i, nid in enumerate(nids_ordered):
        key = f"L{nid.layer}:N{nid.neuron}({nid.polarity})"
        desc = nid_to_desc.get(key, "(no description)")
        tick_labels.append(f"C{clusters_ordered[i]} | {key}")
        row_hover = []
        for j, nid2 in enumerate(nids_ordered):
            key2 = f"L{nid2.layer}:N{nid2.neuron}({nid2.polarity})"
            desc2 = nid_to_desc.get(key2, "(no description)")
            # Truncate descriptions for hover
            d1 = desc[:80] + "..." if len(desc) > 80 else desc
            d2 = desc2[:80] + "..." if len(desc2) > 80 else desc2
            row_hover.append(
                f"sim={sim_ordered[i, j]:.3f} | overlap={overlap_ordered[i, j]}<br>"
                f"<b>Row:</b> C{clusters_ordered[i]} {key}<br>{d1}<br>"
                f"<b>Col:</b> C{clusters_ordered[j]} {key2}<br>{d2}"
            )
        hover_text.append(row_hover)

    # Cluster boundaries for annotation
    boundaries = []
    prev_cl = clusters_ordered[0]
    for i, cl in enumerate(clusters_ordered):
        if cl != prev_cl:
            boundaries.append(i)
            prev_cl = cl

    print(f"Building heatmap with {len(order)} neurons...")

    fig = go.Figure(
        data=go.Heatmap(
            z=sim_ordered,
            x=tick_labels,
            y=tick_labels,
            text=hover_text,
            hoverinfo="text",
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Cosine Similarity"),
        )
    )

    # Add cluster boundary lines
    shapes = []
    for b in boundaries:
        shapes.append(
            dict(
                type="line",
                x0=b - 0.5,
                x1=b - 0.5,
                y0=-0.5,
                y1=len(order) - 0.5,
                line=dict(color="black", width=1),
            )
        )
        shapes.append(
            dict(
                type="line",
                x0=-0.5,
                x1=len(order) - 0.5,
                y0=b - 0.5,
                y1=b - 0.5,
                line=dict(color="black", width=1),
            )
        )

    fig.update_layout(
        title=f"Multi-View Similarity Matrix (k={args.n_clusters}, {len(order)} neurons)",
        width=max(1200, len(order) * 3),
        height=max(1200, len(order) * 3),
        shapes=shapes,
        xaxis=dict(showticklabels=False, title="Neurons (ordered by cluster)"),
        yaxis=dict(
            showticklabels=False, title="Neurons (ordered by cluster)", autorange="reversed"
        ),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"Saved interactive heatmap to {output_path}")
    file_size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"File size: {file_size_mb:.1f} MB")


if __name__ == "__main__":
    main()

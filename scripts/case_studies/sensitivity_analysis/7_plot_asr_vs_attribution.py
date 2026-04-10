"""Plot ASR vs per-cluster attribution as a facetted scatter plot."""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as p9
from circuits.analysis.circuit_ops import Circuit
from scipy import stats

p9.theme_set(
    p9.theme_bw(base_size=10, base_family="Palatino")
    + p9.theme(
        text=p9.element_text(color="#000"),
        axis_title=p9.element_text(size=10),
        axis_text=p9.element_text(size=8),
        legend_text=p9.element_text(size=8),
        legend_title=p9.element_text(size=9),
        panel_grid_major=p9.element_line(size=1, color="#dddddd"),
        panel_grid_minor=p9.element_blank(),
        strip_background=p9.element_blank(),
        legend_margin=0,
    )
)


def parse_asr_from_label(label: str) -> float:
    match = re.search(r"asr([\d.]+)", label)
    if match:
        return float(match.group(1))
    raise ValueError(f"Could not parse ASR from label: {label}")


def main():
    parser = argparse.ArgumentParser(description="Plot ASR vs cluster attribution.")
    parser.add_argument(
        "--circuit",
        default="results/case_studies/sensitivity_analysis_circuit.pkl",
    )
    parser.add_argument(
        "--cluster-state",
        default="results/case_studies/sensitivity_analysis/cluster_state_20260323_131824_mv_k20_harmonic.json",
    )
    parser.add_argument("--clusters", default="C3,C9,C16")
    parser.add_argument("--output", default="outputs/asr_vs_attribution.pdf")
    args = parser.parse_args()

    target_clusters = [s.strip() for s in args.clusters.split(",")]

    # Load circuit and cluster state
    c = Circuit.load_from_pickle(args.circuit)
    c.load_cluster_state(args.cluster_state)

    df = c.df_node.copy()

    # Parse ASR from labels
    label_to_asr = {}
    for label in df["label"].unique():
        label_to_asr[label] = parse_asr_from_label(label)
    df["asr"] = df["label"].map(label_to_asr)

    # Filter to hidden layers only
    num_layers = int(df["layer"].max())
    df = df[(df["layer"] >= 0) & (df["layer"] < num_layers)]

    # Build cluster map: (layer, neuron) -> cluster name
    cluster_map: dict[tuple[int, int], str] = {}
    for nid, cl in c._cluster_map.items():
        cluster_map[(int(nid.layer), int(nid.neuron))] = cl

    # Group by (layer, neuron, label): sum attribution across token positions
    grouped = (
        df.groupby(["layer", "neuron", "label"])
        .agg(total_attr=("attribution", "sum"), asr=("asr", "first"))
        .reset_index()
    )

    # Sum attribution per cluster per CI
    all_labels = sorted(df["label"].unique())
    asr_vec = np.clip(np.array([label_to_asr[l] for l in all_labels]), 0.0, 1.0)

    cluster_attr: dict[str, np.ndarray] = {}
    for (layer, neuron), sub in grouped.groupby(["layer", "neuron"]):
        cl = cluster_map.get((int(layer), int(neuron)), "?")
        attr_by_label = dict(zip(sub["label"], sub["total_attr"]))
        attr_vec = np.array([attr_by_label.get(l, 0.0) for l in all_labels])
        if cl not in cluster_attr:
            cluster_attr[cl] = np.zeros(len(all_labels))
        cluster_attr[cl] += attr_vec

    # Get summary labels for facet titles
    summary_labels = getattr(c, "_cluster_summary_labels", {})

    # Build long-form DataFrame
    rows = []
    for cl in target_clusters:
        if cl not in cluster_attr:
            print(f"Warning: cluster {cl} not found in circuit, skipping.")
            continue
        attr_vec = cluster_attr[cl]
        sl = summary_labels.get(cl, "")
        facet_label = f"{cl}: {sl}" if sl else cl
        for i in range(len(all_labels)):
            rows.append(
                {
                    "cluster": facet_label,
                    "attribution": float(attr_vec[i]),
                    "asr": float(np.clip(asr_vec[i], 0.0, 1.0)),
                }
            )

    plot_df = pd.DataFrame(rows)
    if len(plot_df) == 0:
        print("No data to plot.")
        return

    # Compute Pearson r per cluster for annotation
    r_labels = []
    for cl in plot_df["cluster"].unique():
        sub = plot_df[plot_df["cluster"] == cl]
        r, _ = stats.pearsonr(sub["attribution"], sub["asr"])
        r_labels.append(
            {
                "cluster": cl,
                "label_text": f"r = {r:.3f}",
                "attribution": sub["attribution"].min()
                + 0.05 * (sub["attribution"].max() - sub["attribution"].min()),
                "asr": 0.95,
            }
        )
    r_df = pd.DataFrame(r_labels)

    p = (
        p9.ggplot(plot_df, p9.aes(x="attribution", y="asr", color="cluster"))
        + p9.geom_point(alpha=0.5, size=1.5)
        + p9.geom_smooth(method="lm", se=True, size=0.8)
        + p9.geom_text(
            data=r_df,
            mapping=p9.aes(x="attribution", y="asr", label="label_text"),
            ha="left",
            va="top",
            size=8,
            color="black",
        )
        + p9.facet_wrap("~cluster", scales="free_x")
        + p9.scale_color_brewer(type="qual", palette="Set1")
        + p9.scale_y_continuous(limits=(0, 1))
        + p9.labs(x="Cluster Attribution", y="ASR")
        + p9.guides(color=False)
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    p9.ggsave(p, filename=str(out_path), width=6, height=2.5, dpi=300)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()

"""Correlate neuron attribution with ASR across all CIs."""

import argparse
import re

import numpy as np
import pandas as pd
from circuits.analysis.circuit_ops import Circuit
from scipy import stats


def parse_asr_from_label(label: str) -> float:
    """Extract ASR value from label like 'base_asr0.055' or 'part0_alt0_asr0.060___0'."""
    match = re.search(r"asr([\d.]+)", label)
    if match:
        return float(match.group(1))
    raise ValueError(f"Could not parse ASR from label: {label}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--circuit",
        default="results/case_studies/sensitivity_analysis_circuit.pkl",
    )
    parser.add_argument("--top-k", type=int, default=30, help="Number of top neurons to print")
    parser.add_argument("--min-cis", type=int, default=10, help="Min CIs a neuron must appear in")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    c = Circuit.load_from_pickle(args.circuit)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    c.tokenizer = tokenizer
    c.cluster_multiview(n_clusters=20, get_desc=True)
    desc_cache = c.neuron_label_cache
    df = c.df_node.copy()

    # Parse ASR from labels
    label_to_asr = {}
    for label in df["label"].unique():
        label_to_asr[label] = parse_asr_from_label(label)
    df["asr"] = df["label"].map(label_to_asr)

    # Filter to hidden neurons only (exclude embedding layer=-1 and logit layer=max)
    num_layers = int(df["layer"].max())
    df = df[(df["layer"] >= 0) & (df["layer"] < num_layers)]

    # Group by (layer, neuron): for each CI, sum attribution across token positions
    grouped = (
        df.groupby(["layer", "neuron", "label"])
        .agg(
            total_attr=("attribution", "sum"),
            asr=("asr", "first"),
        )
        .reset_index()
    )

    # Pivot: one row per (layer, neuron), columns = labels, values = total_attr
    # Fill missing with 0 (neuron not in that CI's circuit)
    all_labels = sorted(df["label"].unique())
    asr_vec = np.array([label_to_asr[l] for l in all_labels])

    # Build cluster map: (layer, neuron) -> cluster name
    cluster_map = {}
    for nid, cl in c._cluster_map.items():
        cluster_map[(int(nid.layer), int(nid.neuron))] = cl

    results = []
    for (layer, neuron), sub in grouped.groupby(["layer", "neuron"]):
        # Build attribution vector aligned with all_labels
        attr_by_label = dict(zip(sub["label"], sub["total_attr"]))
        attr_vec = np.array([attr_by_label.get(l, 0.0) for l in all_labels])

        n_present = (attr_vec != 0).sum()
        if n_present < args.min_cis:
            continue

        r, p = stats.pearsonr(attr_vec, asr_vec)
        results.append(
            {
                "layer": int(layer),
                "neuron": int(neuron),
                "cluster": cluster_map.get((int(layer), int(neuron)), "?"),
                "correlation": r,
                "p_value": p,
                "abs_corr": abs(r),
                "n_cis": int(n_present),
                "mean_attr": float(attr_vec.mean()),
            }
        )

    results_df = pd.DataFrame(results).sort_values("abs_corr", ascending=False)

    # Look up descriptions from cache
    def get_desc(layer: int, neuron: int) -> str:
        from neurondb.filters import NeuronPolarity

        for polarity in [NeuronPolarity.POS, NeuronPolarity.NEG]:
            key = (layer, neuron, polarity)
            if key in desc_cache:
                desc = desc_cache[key]
                if desc and desc != "N.A.":
                    return desc
        # Try 2-tuple key
        key2 = (layer, neuron)
        if key2 in desc_cache:
            return desc_cache[key2]
        return ""

    # --- Cluster-level correlations ---
    # Sum attribution per cluster per CI, then correlate with ASR
    results_df["cluster"] = results_df["cluster"].fillna("?")
    # Re-use per-neuron attr vecs: group neurons by cluster, sum their attr per CI
    cluster_attr: dict[str, np.ndarray] = {}
    for (layer, neuron), sub in grouped.groupby(["layer", "neuron"]):
        cl = cluster_map.get((int(layer), int(neuron)), "?")
        attr_by_label = dict(zip(sub["label"], sub["total_attr"]))
        attr_vec = np.array([attr_by_label.get(l, 0.0) for l in all_labels])
        if cl not in cluster_attr:
            cluster_attr[cl] = np.zeros(len(all_labels))
        cluster_attr[cl] += attr_vec

    cluster_results = []
    for cl, attr_vec in cluster_attr.items():
        r, p = stats.pearsonr(attr_vec, asr_vec)
        n_neurons = sum(1 for v in cluster_map.values() if v == cl)
        cluster_results.append(
            {
                "cluster": cl,
                "correlation": r,
                "p_value": p,
                "abs_corr": abs(r),
                "n_neurons": n_neurons,
                "mean_attr": float(attr_vec.mean()),
            }
        )
    cluster_results_df = pd.DataFrame(cluster_results).sort_values("abs_corr", ascending=False)

    # Get cluster descriptions
    cluster_descs = getattr(c, "_cluster_descriptions", {})

    print(f"Total neurons analyzed: {len(results_df)}")
    print(f"Total CIs: {len(all_labels)}")
    print(f"ASR range: [{asr_vec.min():.3f}, {asr_vec.max():.3f}]")

    # Print cluster table
    print()
    print("CLUSTER correlations with ASR:")
    print("=" * 120)
    print(
        f"{'Cluster':>8} {'Corr':>8} {'|Corr|':>8} {'p-value':>10} "
        f"{'#Neurons':>8} {'Mean Attr':>10}  Description"
    )
    print("-" * 120)
    for _, row in cluster_results_df.iterrows():
        cl_desc = cluster_descs.get(row["cluster"], "")
        cl_desc_trunc = cl_desc[:70] + "..." if len(cl_desc) > 70 else cl_desc
        print(
            f"{row['cluster']:>8} {row['correlation']:8.3f} {row['abs_corr']:8.3f} "
            f"{row['p_value']:10.2e} {row['n_neurons']:8.0f} "
            f"{row['mean_attr']:10.4f}  {cl_desc_trunc}"
        )

    # --- Edge-level correlations ---
    df_edge = c.df_edge.copy()
    if len(df_edge) > 0:
        df_edge["asr"] = df_edge["label"].map(label_to_asr)

        # Parse src->tgt into a single edge key (layer, neuron strings already encode both)
        # Group by (layer, token, neuron) = (src_layer->tgt_layer, src_tok->tgt_tok, src_n->tgt_n)
        # Sum attribution across CIs for each unique edge identity (ignoring token positions)
        df_edge["edge_id"] = df_edge["layer"] + "|" + df_edge["neuron"]
        edge_grouped = (
            df_edge.groupby(["edge_id", "label"])
            .agg(
                total_attr=("attribution", "sum"),
                asr=("asr", "first"),
            )
            .reset_index()
        )

        edge_results = []
        for edge_id, sub in edge_grouped.groupby("edge_id"):
            attr_by_label = dict(zip(sub["label"], sub["total_attr"]))
            attr_vec = np.array([attr_by_label.get(l, 0.0) for l in all_labels])

            n_present = (attr_vec != 0).sum()
            if n_present < args.min_cis:
                continue

            r, p = stats.pearsonr(attr_vec, asr_vec)
            # Parse edge_id back into components
            layer_part, neuron_part = edge_id.split("|")
            src_layer, tgt_layer = layer_part.split("->")
            src_neuron, tgt_neuron = neuron_part.split("->")
            # Look up cluster for src and tgt
            src_cl = cluster_map.get((int(src_layer), int(src_neuron)), "?")
            tgt_cl = cluster_map.get((int(tgt_layer), int(tgt_neuron)), "?")

            edge_results.append(
                {
                    "src_layer": int(src_layer),
                    "src_neuron": int(src_neuron),
                    "tgt_layer": int(tgt_layer),
                    "tgt_neuron": int(tgt_neuron),
                    "src_cluster": src_cl,
                    "tgt_cluster": tgt_cl,
                    "correlation": r,
                    "p_value": p,
                    "abs_corr": abs(r),
                    "n_cis": int(n_present),
                    "mean_attr": float(attr_vec.mean()),
                }
            )

        edge_results_df = pd.DataFrame(edge_results).sort_values("abs_corr", ascending=False)

        print()
        print(f"Top {args.top_k} EDGES by |correlation| with ASR:")
        print("=" * 130)
        print(
            f"{'SrcL':>5} {'SrcN':>7} {'SrcCl':>6} -> {'TgtL':>5} {'TgtN':>7} {'TgtCl':>6}"
            f"  {'Corr':>8} {'|Corr|':>8} {'p-value':>10} {'#CIs':>5} {'Mean Attr':>10}"
            f"  Src Desc"
        )
        print("-" * 130)
        for _, row in edge_results_df.head(args.top_k).iterrows():
            src_desc = get_desc(row["src_layer"], row["src_neuron"])
            src_desc_trunc = src_desc[:45] + "..." if len(src_desc) > 45 else src_desc
            print(
                f"{row['src_layer']:5} {row['src_neuron']:7} {row['src_cluster']:>6} -> "
                f"{row['tgt_layer']:5} {row['tgt_neuron']:7} {row['tgt_cluster']:>6}"
                f"  {row['correlation']:8.3f} {row['abs_corr']:8.3f} {row['p_value']:10.2e}"
                f" {row['n_cis']:5} {row['mean_attr']:10.4f}"
                f"  {src_desc_trunc}"
            )
    else:
        print("\nNo edges in circuit data.")

    # Print neuron table
    print()
    print(f"Top {args.top_k} NEURONS by |correlation| with ASR:")
    print("=" * 120)
    print(
        f"{'Layer':>5} {'Neuron':>7} {'Cluster':>8} {'Corr':>8} {'|Corr|':>8} "
        f"{'p-value':>10} {'#CIs':>5} {'Mean Attr':>10}  Description"
    )
    print("-" * 120)
    for _, row in results_df.head(args.top_k).iterrows():
        desc = get_desc(int(row["layer"]), int(row["neuron"]))
        desc_trunc = desc[:55] + "..." if len(desc) > 55 else desc
        print(
            f"{row['layer']:5.0f} {row['neuron']:7.0f} {row['cluster']:>8} {row['correlation']:8.3f} "
            f"{row['abs_corr']:8.3f} {row['p_value']:10.2e} {row['n_cis']:5.0f} "
            f"{row['mean_attr']:10.4f}  {desc_trunc}"
        )


if __name__ == "__main__":
    main()

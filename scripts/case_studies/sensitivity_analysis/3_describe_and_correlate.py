"""Cluster, describe, and correlate neuron/edge/cluster attribution with ASR."""

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from circuits.analysis.circuit_ops import Circuit
from scipy import stats
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

CIRCUIT_PICKLE = "results/case_studies/sensitivity_analysis_circuit.pkl"
OUTPUT_DIR = Path("results/case_studies/sensitivity_analysis")


def parse_asr_from_label(label: str) -> float:
    match = re.search(r"asr([\d.]+)", label)
    if match:
        return float(match.group(1))
    raise ValueError(f"Could not parse ASR from label: {label}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuit", default=CIRCUIT_PICKLE)
    parser.add_argument("--n-clusters", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--min-cis", type=int, default=10)
    parser.add_argument("--skip-describe", action="store_true", help="Skip description generation")
    parser.add_argument("--skip-attr", action="store_true", help="Skip attr descriptions")
    parser.add_argument("--num-expl-samples", type=int, default=5)
    parser.add_argument("--combine", type=str, default="harmonic")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load circuit + tokenizer
    logger.info("Loading circuit from %s", args.circuit)
    c = Circuit.load_from_pickle(args.circuit)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    c.set_tokenizer(tokenizer, num_layers=32)

    # Cluster with multiview
    logger.info("Clustering with multi-view (k=%d, combine=%s)...", args.n_clusters, args.combine)
    c.cluster_multiview(
        n_clusters=args.n_clusters,
        combine=args.combine,
        get_desc=True,
        verbose=True,
    )
    desc_cache = c.neuron_label_cache

    # Generate v2 descriptions (vllm for attr, API for contrib)
    if not args.skip_describe:
        logger.info("Generating v2 descriptions...")
        result = c.label_clusters_simulator_v2(
            score_explanations=True,
            num_expl_samples=args.num_expl_samples,
            skip_attr=args.skip_attr,
            attr_backend="vllm",
            contrib_model_name="claude-haiku-4-5-20251001",
            verbose=True,
        )
        attr_results, contrib_results, attr_exemplars, contrib_exemplars, cluster_to_neurons = (
            result
        )

        # Save descriptions JSON
        cluster_id_to_name = getattr(c, "_cluster_id_to_name", {})
        suffix = f"mv_k{args.n_clusters}_{args.combine}"
        output_file = OUTPUT_DIR / f"explanations_v2_{timestamp}_{suffix}.json"

        def neuron_key_to_name(neuron_key):
            cid = neuron_key.layer if hasattr(neuron_key, "layer") else neuron_key
            return cluster_id_to_name.get(cid, f"C{cid}")

        output_data = {
            "metadata": {
                "circuit_pickle": str(args.circuit),
                "timestamp": timestamp,
                "n_clusters": args.n_clusters,
                "combine": args.combine,
            },
            "cluster_id_to_name": {str(k): v for k, v in cluster_id_to_name.items()},
            "attr": {},
            "contrib": {},
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

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        logger.info("Descriptions saved to %s", output_file)

        # Save cluster state
        cluster_state_out = OUTPUT_DIR / f"cluster_state_{timestamp}_{suffix}.json"
        c.save_cluster_state(cluster_state_out)

    # --- Correlation analysis ---
    cluster_descs = getattr(c, "_cluster_descriptions", {})
    cluster_map = {}
    for nid, cl in c._cluster_map.items():
        cluster_map[(int(nid.layer), int(nid.neuron))] = cl

    df = c.df_node.copy()
    label_to_asr = {}
    for label in df["label"].unique():
        label_to_asr[label] = parse_asr_from_label(label)
    df["asr"] = df["label"].map(label_to_asr)

    num_layers = int(df["layer"].max())
    df = df[(df["layer"] >= 0) & (df["layer"] < num_layers)]

    grouped = (
        df.groupby(["layer", "neuron", "label"])
        .agg(total_attr=("attribution", "sum"), asr=("asr", "first"))
        .reset_index()
    )
    all_labels = sorted(df["label"].unique())
    asr_vec = np.array([label_to_asr[l] for l in all_labels])

    # Neuron correlations
    results = []
    for (layer, neuron), sub in grouped.groupby(["layer", "neuron"]):
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

    # Neuron description lookup
    def get_desc(layer: int, neuron: int) -> str:
        from neurondb.filters import NeuronPolarity

        for polarity in [NeuronPolarity.POS, NeuronPolarity.NEG]:
            key = (layer, neuron, polarity)
            if key in desc_cache:
                desc = desc_cache[key]
                if desc and desc != "N.A.":
                    return desc
        key2 = (layer, neuron)
        if key2 in desc_cache:
            return desc_cache[key2]
        return ""

    # Cluster correlations
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

    # Edge correlations
    df_edge = c.df_edge.copy()
    edge_results_df = pd.DataFrame()
    if len(df_edge) > 0:
        df_edge["asr"] = df_edge["label"].map(label_to_asr)
        df_edge["edge_id"] = df_edge["layer"] + "|" + df_edge["neuron"]
        edge_grouped = (
            df_edge.groupby(["edge_id", "label"])
            .agg(total_attr=("attribution", "sum"), asr=("asr", "first"))
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
            layer_part, neuron_part = edge_id.split("|")
            src_layer, tgt_layer = layer_part.split("->")
            src_neuron, tgt_neuron = neuron_part.split("->")
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

    # --- Print tables ---
    print(f"\nTotal neurons analyzed: {len(results_df)}")
    print(f"Total CIs: {len(all_labels)}")
    print(f"ASR range: [{asr_vec.min():.3f}, {asr_vec.max():.3f}]")

    print("\nCLUSTER correlations with ASR:")
    print("=" * 130)
    print(
        f"{'Cluster':>8} {'Corr':>8} {'|Corr|':>8} {'p-value':>10} "
        f"{'#Neurons':>8} {'Mean Attr':>10}  Description"
    )
    print("-" * 130)
    for _, row in cluster_results_df.iterrows():
        cl_desc = cluster_descs.get(row["cluster"], "")
        cl_desc_trunc = cl_desc[:80] + "..." if len(cl_desc) > 80 else cl_desc
        print(
            f"{row['cluster']:>8} {row['correlation']:8.3f} {row['abs_corr']:8.3f} "
            f"{row['p_value']:10.2e} {row['n_neurons']:8.0f} "
            f"{row['mean_attr']:10.4f}  {cl_desc_trunc}"
        )

    if len(edge_results_df) > 0:
        print(f"\nTop {args.top_k} EDGES by |correlation| with ASR:")
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

    print(f"\nTop {args.top_k} NEURONS by |correlation| with ASR:")
    print("=" * 130)
    print(
        f"{'Layer':>5} {'Neuron':>7} {'Cluster':>8} {'Corr':>8} {'|Corr|':>8} "
        f"{'p-value':>10} {'#CIs':>5} {'Mean Attr':>10}  Description"
    )
    print("-" * 130)
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

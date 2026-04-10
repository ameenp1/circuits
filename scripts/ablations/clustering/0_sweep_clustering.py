"""Sweep clustering settings on capitals circuit and report balance metrics.

No descriptions or explanations — just clustering and measuring balance.
"""

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from circuits.analysis.circuit_ops import Circuit
from circuits.utils.constants import RESULTS_DIR
from numpy.typing import NDArray
from sklearn.cluster import AgglomerativeClustering, BisectingKMeans, KMeans, SpectralClustering


def cluster_balance_metrics(cluster_col: list[str], n_clusters: int) -> dict:
    """Compute balance metrics for a clustering assignment."""
    counts = Counter(cluster_col)
    counts.pop("-1", None)
    if not counts:
        return {"error": "no clusters"}

    sizes = sorted(counts.values(), reverse=True)
    total = sum(sizes)

    return {
        "n_neurons": total,
        "n_clusters_used": len(sizes),
        "largest": sizes[0],
        "smallest": sizes[-1],
        "ratio_largest_smallest": sizes[0] / max(sizes[-1], 1),
        "std": float(np.std(sizes)),
        "cv": float(np.std(sizes) / np.mean(sizes)),
        "mean": float(np.mean(sizes)),
        "median": float(np.median(sizes)),
        "pct_in_largest": sizes[0] / total * 100,
        "top5": sizes[:5],
        "bottom5": sizes[-5:],
    }


def run_clustering(X: NDArray, n_clusters: int, method: str, linkage: str | None = None) -> NDArray:
    """Run clustering with given method and return labels."""
    np.random.seed(42)
    random.seed(42)

    if method == "agglomerative":
        metric = "euclidean" if linkage == "ward" else "cosine"
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage or "ward",
            metric=metric,
        )
    elif method == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif method == "bisecting_kmeans":
        clusterer = BisectingKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif method == "spectral":
        clusterer = SpectralClustering(n_clusters=n_clusters, random_state=42, affinity="rbf")
    else:
        raise ValueError(f"Unknown method: {method}")

    return clusterer.fit_predict(X)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--circuit-pickle",
        type=str,
        default=str(RESULTS_DIR / "case_studies/capitals_circuit.pkl"),
    )
    parser.add_argument("--n-clusters", type=int, default=64)
    parser.add_argument(
        "--output",
        type=str,
        default=str(RESULTS_DIR / "case_studies/clustering_sweep.json"),
    )
    args = parser.parse_args()

    print(f"Loading circuit from {args.circuit_pickle}")
    circuit = Circuit.load_from_pickle(args.circuit_pickle)

    # Embedding configs: unit_norm on/off
    embed_configs = [
        {"do_layernorm": False, "label": "no_unitnorm"},
        {"do_layernorm": True, "label": "unitnorm"},
    ]

    # Clustering configs
    cluster_configs = [
        {"method": "agglomerative", "linkage": "ward", "label": "agg_ward"},
        {"method": "agglomerative", "linkage": "average", "label": "agg_avg_cosine"},
        {"method": "agglomerative", "linkage": "complete", "label": "agg_complete_cosine"},
        {"method": "kmeans", "linkage": None, "label": "kmeans"},
        {"method": "bisecting_kmeans", "linkage": None, "label": "bisecting_kmeans"},
        {"method": "spectral", "linkage": None, "label": "spectral"},
    ]

    results = []

    for ec in embed_configs:
        print(f"\n{'=' * 60}")
        print(f"Embedding: {ec['label']}")
        print(f"{'=' * 60}")

        # Use circuit.cluster() to get embeddings via the standard pipeline
        circuit.cluster(
            n_clusters=args.n_clusters,
            do_layernorm=ec["do_layernorm"],
            get_desc=False,
            verbose=True,
        )

        # Read stored embedding matrix from cluster()
        X = circuit._cluster_embedding_matrix
        print(f"  Embedding matrix: {X.shape}")

        for cc in cluster_configs:
            combo_label = f"{ec['label']}__{cc['label']}"
            print(f"  {cc['label']}...", end=" ", flush=True)

            try:
                raw_labels = run_clustering(X, args.n_clusters, cc["method"], cc["linkage"])
                str_labels = [str(x) for x in raw_labels]
                metrics = cluster_balance_metrics(str_labels, args.n_clusters)
                metrics["embed"] = ec["label"]
                metrics["cluster"] = cc["label"]
                metrics["combo"] = combo_label
                results.append(metrics)
                print(
                    f"largest={metrics['largest']}, smallest={metrics['smallest']}, "
                    f"cv={metrics['cv']:.3f}, pct_largest={metrics['pct_in_largest']:.1f}%"
                )
            except Exception as e:
                print(f"FAILED: {e}")
                results.append(
                    {
                        "embed": ec["label"],
                        "cluster": cc["label"],
                        "combo": combo_label,
                        "error": str(e),
                    }
                )

    # Sort by CV (lower = more balanced)
    valid = [r for r in results if "error" not in r]
    valid.sort(key=lambda r: r["cv"])

    print(f"\n{'=' * 80}")
    print("RESULTS RANKED BY BALANCE (coefficient of variation, lower = more balanced)")
    print(f"{'=' * 80}")
    print(f"{'Combo':<40} {'CV':>6} {'Largest':>8} {'Smallest':>8} " f"{'%Largest':>9} {'Std':>8}")
    print("-" * 81)
    for r in valid:
        print(
            f"{r['combo']:<40} {r['cv']:>6.3f} {r['largest']:>8} {r['smallest']:>8} "
            f"{r['pct_in_largest']:>8.1f}% {r['std']:>8.1f}"
        )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"n_clusters": args.n_clusters, "results": results}, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()

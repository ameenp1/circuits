"""Sweep multi-view similarity clustering on capitals circuit.

Compares baseline (concatenated embedding) clustering with multi-view
approaches (spectral on per-CI similarity, Leiden community detection).
"""

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from circuits.analysis.circuit_ops import Circuit
from circuits.analysis.multiview_cluster import (
    compute_contrib_sign_conflicts,
    compute_diagnostics,
    compute_intra_inter_sim_ratio,
    leiden_cluster,
    spectral_cluster,
)
from circuits.utils.constants import RESULTS_DIR
from numpy.typing import NDArray
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity as cos_sim_fn


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


def run_baseline_clustering(X: NDArray, n_clusters: int, method: str) -> NDArray:
    """Run baseline clustering on concatenated embedding matrix."""
    np.random.seed(42)
    random.seed(42)

    if method == "agg_ward":
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    elif method == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif method == "spectral_rbf":
        clusterer = SpectralClustering(n_clusters=n_clusters, random_state=42, affinity="rbf")
    else:
        raise ValueError(f"Unknown method: {method}")

    return clusterer.fit_predict(X)


def _parse_method_key(method: str) -> tuple[str, int]:
    """Parse method string into (row_name, k). E.g. 'mv_spectral_k16' -> ('mv_spectral', 16)."""
    # Find the _k{N} part
    parts = method.split("_k")
    k_str = parts[-1]
    # k_str might have suffixes like '_harmonic_linshift', but k is always the first int
    k = int(k_str.split("_")[0]) if "_" in k_str else int(k_str)
    # row name is everything except the _k{N} part
    row_name = method.replace(f"_k{k}", "")
    return row_name, k


# Display name mapping for LaTeX table rows
_ROW_DISPLAY_NAMES: dict[str, tuple[str, str | None]] = {
    # (display_name, section_header)
    "mv_spectral": ("Mean", "Multi-view spectral, normalised Laplacian, $\\max(0, S)$"),
    "mv_spectral_harmonic": (
        "Harmonic",
        None,
    ),
    "mv_spectral_linshift": (
        "Mean",
        "Multi-view spectral, normalised Laplacian, $(S{+}1)/2$",
    ),
    "mv_spectral_harmonic_linshift": ("Harmonic", None),
    "mv_spectral_unnorm": (
        "Mean",
        "Multi-view spectral, unnormalised Laplacian, $\\max(0, S)$",
    ),
    "mv_spectral_harmonic_unnorm": ("Harmonic", None),
    "mv_spectral_linshift_unnorm": (
        "Mean",
        "Multi-view spectral, unnormalised Laplacian, $(S{+}1)/2$",
    ),
    "mv_spectral_harmonic_linshift_unnorm": ("Harmonic", None),
    "baseline_kmeans": ("$K$-Means", "Concatenated embedding baselines"),
    "baseline_agg_ward": ("Ward", None),
    "baseline_spectral_rbf": ("Spectral (RBF)", None),
}

# Order of rows in the table
_ROW_ORDER = [
    "mv_spectral",
    "mv_spectral_harmonic",
    "mv_spectral_linshift",
    "mv_spectral_harmonic_linshift",
    "mv_spectral_unnorm",
    "mv_spectral_harmonic_unnorm",
    "mv_spectral_linshift_unnorm",
    "mv_spectral_harmonic_linshift_unnorm",
    "baseline_kmeans",
    "baseline_agg_ward",
    "baseline_spectral_rbf",
]


def print_latex_table(valid_results: list[dict]) -> None:
    """Print a LaTeX table with metrics as big groups and k as inner columns."""
    ks = [4, 8, 16, 32, 64]
    metrics = [
        ("CV", "cv", False),
        ("Silh", "silhouette", True),
        ("All\\%", "pct_all_sign_conflict", False),
    ]

    # Build lookup: (row_name, k) -> result
    lookup: dict[tuple[str, int], dict] = {}
    for r in valid_results:
        row_name, k = _parse_method_key(r["method"])
        lookup[(row_name, k)] = r

    # Determine which rows are present
    present_rows = [r for r in _ROW_ORDER if any((r, k) in lookup for k in ks)]

    n_metrics = len(metrics)
    n_ks = len(ks)
    total_cols = n_metrics * n_ks

    print(f"\n{'=' * 80}")
    print("LATEX TABLE")
    print(f"{'=' * 80}")

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Clustering quality across methods and number of clusters $k$.")
    lines.append("\\textbf{CV}: coefficient of variation of cluster sizes ($\\downarrow$);")
    lines.append("\\textbf{Silh}: silhouette score ($\\uparrow$);")
    lines.append(
        "\\textbf{All}: \\% of intra-cluster pairs with all opposing contribution signs ($\\downarrow$).}"
    )
    lines.append("\\label{tab:clustering-sweep}")
    lines.append("\\small")
    lines.append("\\adjustbox{max width=\\textwidth}{")

    # Column spec: l + (n_ks r's per metric)
    col_spec = "l " + " ".join(["*{%d}{r}" % n_ks] * n_metrics)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header row 1: metric groups
    header1_parts = []
    for i, (metric_name, _, _) in enumerate(metrics):
        start = 2 + i * n_ks
        end = start + n_ks - 1
        header1_parts.append(f"\\multicolumn{{{n_ks}}}{{c}}{{{metric_name}}}")
    lines.append("& " + " & ".join(header1_parts) + " \\\\")

    # cmidrule for each metric group
    cmidrules = []
    for i in range(n_metrics):
        start = 2 + i * n_ks
        end = start + n_ks - 1
        cmidrules.append(f"\\cmidrule(lr){{{start}-{end}}}")
    lines.append(" ".join(cmidrules))

    # Header row 2: k values repeated per metric
    k_headers = []
    for _ in metrics:
        for k in ks:
            k_headers.append(f"${k}$")
    lines.append("Method & " + " & ".join(k_headers) + " \\\\")
    lines.append("\\midrule")

    # Find best value per column (metric, k) for bolding
    best_per_col: dict[tuple[str, int], float] = {}
    for metric_name, metric_key, higher_better in metrics:
        for k in ks:
            vals = []
            for row_name in present_rows:
                r = lookup.get((row_name, k))
                if r is not None:
                    v = r.get(metric_key)
                    if v is not None:
                        vals.append(v)
            if vals:
                best_per_col[(metric_key, k)] = max(vals) if higher_better else min(vals)

    # Data rows
    last_section = None
    for row_name in present_rows:
        display_name, section = _ROW_DISPLAY_NAMES.get(row_name, (row_name, None))

        if section is not None and section != last_section:
            if last_section is not None:
                lines.append("\\addlinespace")
            lines.append(f"\\multicolumn{{{total_cols + 1}}}{{l}}{{\\textit{{{section}}}}} \\\\")
            last_section = section

        cells = []
        for metric_name, metric_key, higher_better in metrics:
            for k in ks:
                r = lookup.get((row_name, k))
                if r is None:
                    cells.append("--")
                    continue
                val = r.get(metric_key)
                if val is None:
                    cells.append("--")
                    continue
                is_best = val == best_per_col.get((metric_key, k))
                if metric_key == "pct_all_sign_conflict":
                    cell = f"{val:.1f}\\%"
                elif metric_key == "silhouette":
                    cell = f"${val:.2f}$" if val < 0 else f"{val:.2f}"
                else:
                    cell = f"{val:.2f}"
                if is_best:
                    cell = f"\\textbf{{{cell}}}"
                cells.append(cell)

        lines.append(f"\\quad {display_name} & " + " & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table*}")

    latex = "\n".join(lines)
    print(latex)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Sweep multi-view clustering on capitals circuit")
    parser.add_argument(
        "--circuit-pickle",
        type=str,
        default=str(RESULTS_DIR / "case_studies/capitals_circuit.pkl"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_DIR / "case_studies/multiview_clustering"),
    )
    parser.add_argument("--n-clusters", type=int, default=64, help="For spectral/baseline methods")
    parser.add_argument("--skip-leiden", action="store_true", help="Skip Leiden (needs igraph)")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip baseline methods")
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading circuit from {args.circuit_pickle}")
    circuit = Circuit.load_from_pickle(args.circuit_pickle)

    results = []
    all_labels: dict[str, list[str]] = {}  # method -> labels for ARI comparison

    # ── Baseline methods ──
    # Use circuit.cluster() to get embeddings, then re-cluster with different methods/k
    if not args.skip_baselines:
        print("\n" + "=" * 60)
        print("Baseline methods (concatenated embedding)")
        print("=" * 60)

        # Run cluster() once to get the embedding matrix and prepared data
        circuit.cluster(n_clusters=args.n_clusters, get_desc=False, verbose=True)
        X = circuit._cluster_embedding_matrix
        baseline_nids = circuit._cluster_neuron_ids
        df_prepared = circuit._cluster_df_prepared
        print(f"  Embedding matrix: {X.shape}")

        # Precompute cosine similarity and distance for metrics
        baseline_sim = cos_sim_fn(X)
        baseline_dist = cosine_distances(X)

        for method in ["agg_ward", "kmeans", "spectral_rbf"]:
            for k in [4, 8, 16, 32, 64]:
                label = f"baseline_{method}_k{k}"
                print(f"  {method} k={k}...", end=" ", flush=True)
                try:
                    raw_labels = run_baseline_clustering(X, k, method)
                    str_labels = [str(x) for x in raw_labels]
                    metrics = cluster_balance_metrics(str_labels, k)
                    metrics["method"] = label
                    metrics["type"] = "baseline"
                    n_unique = len(set(raw_labels.tolist()))
                    if n_unique > 1 and n_unique < len(raw_labels):
                        metrics["silhouette"] = float(
                            silhouette_score(baseline_dist, raw_labels, metric="precomputed")
                        )
                    else:
                        metrics["silhouette"] = None
                    ii = compute_intra_inter_sim_ratio(baseline_sim, raw_labels)
                    metrics.update(ii)
                    sc = compute_contrib_sign_conflicts(df_prepared, raw_labels, baseline_nids)
                    metrics.update(sc)
                    results.append(metrics)
                    if k == args.n_clusters:
                        all_labels[label] = str_labels
                    silh = metrics["silhouette"]
                    silh_str = f"{silh:.3f}" if silh is not None else "N/A"
                    print(
                        f"cv={metrics['cv']:.3f}, silh={silh_str}, "
                        f"ii={ii['intra_inter_ratio']:.2f}, "
                        f"any_conflict={sc['pct_any_sign_conflict']:.1f}%"
                    )
                except Exception as e:
                    print(f"FAILED: {e}")
                    results.append({"method": label, "type": "baseline", "error": str(e)})

    # ── Multi-view similarity ──
    # Use circuit.cluster_multiview() for the default config, then also sweep variants
    combine_modes = ["mean", "harmonic"]
    sim_data: dict[str, tuple] = {}  # combine_mode -> (sim_filtered, overlap_filtered)
    neuron_ids_filtered: list | None = None
    df_prepared_mv = None

    for combine_mode in combine_modes:
        print(f"\n{'=' * 60}")
        print(f"Computing multi-view per-CI similarities (combine={combine_mode})")
        print("=" * 60)

        # Use circuit.cluster_multiview() to compute similarities + cluster
        circuit.cluster_multiview(
            n_clusters=args.n_clusters,
            get_desc=False,
            weight_by_attribution=args.weight_by_attribution,
            use_attributions_as_view=args.use_attributions_as_view,
            combine=combine_mode,
            verbose=True,
        )

        # Read stored intermediate state
        sim_filtered = circuit._mv_sim_matrix
        overlap_filtered = circuit._mv_overlap_counts
        nids_f = circuit._mv_neuron_ids

        print(f"  Similarity matrix: {sim_filtered.shape}")
        print(
            f"  Overlap: mean={overlap_filtered.mean():.1f}, "
            f"pct_zero={(~circuit._mv_had_overlap).mean() * 100:.1f}%"
        )
        print(
            f"  Sim stats: mean={sim_filtered.mean():.3f}, "
            f"std={sim_filtered.std():.3f}, "
            f"min={sim_filtered.min():.3f}, max={sim_filtered.max():.3f}"
        )

        sim_data[combine_mode] = (sim_filtered, overlap_filtered)
        if neuron_ids_filtered is None:
            neuron_ids_filtered = nids_f
            df_prepared_mv = circuit._mv_df_prepared

    assert neuron_ids_filtered is not None, "No multi-view similarities computed"
    assert df_prepared_mv is not None

    # ── Multi-view spectral ──
    print("\n" + "=" * 60)
    print("Multi-view spectral clustering")
    print("=" * 60)

    # Ablation grid: (use_linear_shift, unnormalized, suffix)
    spectral_variants = [
        (False, False, ""),  # default: max(0,S) + normalized Laplacian
        (True, False, "_linshift"),  # linear shift (S+1)/2 + normalized
        (False, True, "_unnorm"),  # max(0,S) + unnormalized Laplacian
        (True, True, "_linshift_unnorm"),  # linear shift + unnormalized (old behavior)
    ]

    for combine_mode in combine_modes:
        sim_filtered, overlap_filtered = sim_data[combine_mode]
        combine_suffix = f"_{combine_mode}" if combine_mode != "mean" else ""

        for use_linear_shift, unnormalized, suffix in spectral_variants:
            variant_desc = [combine_mode]
            if use_linear_shift:
                variant_desc.append("linear_shift")
            else:
                variant_desc.append("max(0,S)")
            if unnormalized:
                variant_desc.append("unnorm_laplacian")
            else:
                variant_desc.append("norm_laplacian")
            print(f"\n  Variant: {', '.join(variant_desc)}")

            for k in [4, 8, 16, 32, 64]:
                label = f"mv_spectral_k{k}{combine_suffix}{suffix}"
                print(f"    k={k}...", end=" ", flush=True)
                try:
                    raw_labels = spectral_cluster(
                        sim_filtered,
                        n_clusters=k,
                        use_linear_shift=use_linear_shift,
                        unnormalized=unnormalized,
                    )
                    str_labels = [str(x) for x in raw_labels]
                    metrics = cluster_balance_metrics(str_labels, k)
                    diag = compute_diagnostics(
                        sim_filtered,
                        raw_labels,
                        overlap_filtered,
                        use_linear_shift=use_linear_shift,
                        unnormalized=unnormalized,
                    )
                    metrics["method"] = label
                    metrics["type"] = "mv_spectral"
                    metrics["combine"] = combine_mode
                    metrics["use_linear_shift"] = use_linear_shift
                    metrics["unnormalized"] = unnormalized
                    metrics["silhouette"] = diag["silhouette"]
                    metrics["eigenvalues"] = diag["eigenvalues"][:10]
                    ii = compute_intra_inter_sim_ratio(sim_filtered, raw_labels)
                    metrics.update(ii)
                    sc = compute_contrib_sign_conflicts(
                        df_prepared_mv, raw_labels, neuron_ids_filtered
                    )
                    metrics.update(sc)
                    results.append(metrics)
                    all_labels[label] = str_labels
                    print(
                        f"cv={metrics['cv']:.3f}, "
                        f"largest={metrics['largest']}, smallest={metrics['smallest']}, "
                        f"silhouette={diag['silhouette']:.3f}, "
                        f"ii={ii['intra_inter_ratio']:.2f}, "
                        f"any_conflict={sc['pct_any_sign_conflict']:.1f}%"
                        if diag["silhouette"] is not None
                        else ""
                    )
                except Exception as e:
                    print(f"FAILED: {e}")
                    results.append({"method": label, "type": "mv_spectral", "error": str(e)})

    # ── Multi-view Leiden ──
    if not args.skip_leiden:
        print("\n" + "=" * 60)
        print("Multi-view Leiden clustering")
        print("=" * 60)

        sim_filtered_mean, overlap_filtered_mean = sim_data["mean"]
        for knn in [5, 10, 15]:
            for resolution in [0.5, 1.0, 2.0]:
                label = f"mv_leiden_knn{knn}_res{resolution}"
                print(f"  knn={knn}, resolution={resolution}...", end=" ", flush=True)
                try:
                    raw_labels = leiden_cluster(sim_filtered_mean, knn=knn, resolution=resolution)
                    str_labels = [str(x) for x in raw_labels]
                    n_found = len(set(raw_labels))
                    metrics = cluster_balance_metrics(str_labels, n_found)
                    diag = compute_diagnostics(sim_filtered_mean, raw_labels, overlap_filtered_mean)
                    metrics["method"] = label
                    metrics["type"] = "mv_leiden"
                    metrics["silhouette"] = diag["silhouette"]
                    ii = compute_intra_inter_sim_ratio(sim_filtered_mean, raw_labels)
                    metrics.update(ii)
                    sc = compute_contrib_sign_conflicts(
                        df_prepared_mv, raw_labels, neuron_ids_filtered
                    )
                    metrics.update(sc)
                    results.append(metrics)
                    all_labels[label] = str_labels
                    print(
                        f"n_clusters={n_found}, cv={metrics['cv']:.3f}, "
                        f"largest={metrics['largest']}, smallest={metrics['smallest']}, "
                        f"ii={ii['intra_inter_ratio']:.2f}, "
                        f"any_conflict={sc['pct_any_sign_conflict']:.1f}%"
                    )
                except Exception as e:
                    print(f"FAILED: {e}")
                    results.append({"method": label, "type": "mv_leiden", "error": str(e)})

    # ── ARI comparison ──
    print("\n" + "=" * 60)
    print("Adjusted Rand Index between methods")
    print("=" * 60)

    ari_results = {}
    method_names = list(all_labels.keys())
    for i, m1 in enumerate(method_names):
        for m2 in method_names[i + 1 :]:
            l1, l2 = all_labels[m1], all_labels[m2]
            if len(l1) == len(l2):
                ari = adjusted_rand_score(l1, l2)
                key = f"{m1} vs {m2}"
                ari_results[key] = round(ari, 4)
                print(f"  {key}: {ari:.4f}")

    # ── Summary ──
    valid = [r for r in results if "error" not in r]
    valid.sort(key=lambda r: r.get("cv", float("inf")))

    print(f"\n{'=' * 80}")
    print("RESULTS RANKED BY BALANCE (CV, lower = more balanced)")
    print(f"{'=' * 80}")
    print(
        f"{'Method':<40} {'CV':>6} {'#Clust':>7} {'Largest':>8} {'Smallest':>8} "
        f"{'Silh':>8} {'I/I':>6} {'Any%':>6} {'All%':>6}"
    )
    print("-" * 100)
    for r in valid:
        silh = r.get("silhouette", None)
        silh_str = f"{silh:>8.3f}" if silh is not None else "     N/A"
        ii_ratio = r.get("intra_inter_ratio", None)
        ii_str = f"{ii_ratio:>6.2f}" if ii_ratio is not None else "   N/A"
        any_c = r.get("pct_any_sign_conflict", None)
        any_str = f"{any_c:>6.1f}" if any_c is not None else "   N/A"
        all_c = r.get("pct_all_sign_conflict", None)
        all_str = f"{all_c:>6.1f}" if all_c is not None else "   N/A"
        print(
            f"{r['method']:<40} {r['cv']:>6.3f} {r['n_clusters_used']:>7} "
            f"{r['largest']:>8} {r['smallest']:>8} {silh_str} {ii_str} {any_str} {all_str}"
        )

    # ── LaTeX table ──
    print_latex_table(valid)

    # ── Save ──
    output_path = output_dir / "multiview_sweep.json"
    with open(output_path, "w") as f:
        json.dump(
            {"n_clusters": args.n_clusters, "results": results, "ari": ari_results},
            f,
            indent=2,
            default=str,
        )
    print(f"\nSaved results to {output_path}")

    # Save diagnostics for the sim matrix (using default mean combine)
    sim_filtered_default, overlap_filtered_default = sim_data["mean"]
    diag_full = compute_diagnostics(
        sim_filtered_default,
        spectral_cluster(sim_filtered_default, n_clusters=args.n_clusters),
        overlap_filtered_default,
    )
    diag_path = output_dir / "diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diag_full, f, indent=2, default=str)
    print(f"Saved diagnostics to {diag_path}")


if __name__ == "__main__":
    main()

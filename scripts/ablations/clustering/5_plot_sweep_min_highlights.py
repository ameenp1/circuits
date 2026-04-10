"""Plot min_highlights × threshold_mode sweep results from existing explanation JSONs.

Reads scored explanation JSONs from the sweep directory and plots mean-of-max
attr score per cluster vs min_highlights, with separate lines for each
(threshold_mode, num_samples) combination.

Usage:
    python scripts/ablations/clustering/5_plot_sweep_min_highlights.py
    python scripts/ablations/clustering/5_plot_sweep_min_highlights.py --input-dir /path/to/jsons
    python scripts/ablations/clustering/5_plot_sweep_min_highlights.py --facet
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import plotnine as p9
from circuits.utils.constants import RESULTS_DIR

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

DEFAULT_INPUT_DIR = RESULTS_DIR / "case_studies/capitals/explanations_v2"
OUTPUT_DIR = RESULTS_DIR / "case_studies/capitals"


def extract_max_score_per_cluster(attr: dict) -> dict[str, float]:
    """For each cluster, take the max score across all signs and explanations."""
    cluster_scores: dict[str, float] = {}
    for cluster_name, sign_data in attr.items():
        best = None
        for sign in ("pos", "neg", "combined"):
            for expl in sign_data.get(sign, []):
                score = expl.get("score")
                if score is not None and (best is None or score > best):
                    best = score
        if best is not None:
            cluster_scores[cluster_name] = best
    return cluster_scores


def load_sweep_data(input_dir: Path) -> pd.DataFrame:
    """Load all scored explanation JSONs and extract per-cluster max scores."""
    rows: list[dict] = []

    for f in sorted(input_dir.glob("*.json")):
        with open(f) as fh:
            d = json.load(fh)

        m = d.get("metadata", {})
        if not m.get("scored", False):
            continue

        n = m.get("num_expl_samples", 5)
        mode = m.get("threshold_mode")
        mh = m.get("min_highlights")

        # Parse from filename for older files
        if mode is None:
            fm = re.search(r"(topk|quantile)", f.name)
            mode = fm.group(1) if fm else None
        if mh is None:
            fm = re.search(r"mh(\d+)", f.name)
            mh = int(fm.group(1)) if fm else None

        # Known files without metadata
        if mode is None or mh is None:
            if "20260305_232222" in f.name:
                mode, mh = "quantile", 4
            elif "20260305_235733" in f.name:
                mode, mh = "quantile", 4
            else:
                continue

        attr = d.get("attr", {})
        cluster_scores = extract_max_score_per_cluster(attr)

        for cluster_name, score in cluster_scores.items():
            rows.append(
                {
                    "mode": mode,
                    "k": mh,
                    "n": n,
                    "cluster": cluster_name,
                    "score": score,
                    "file": f.name,
                }
            )

    df = pd.DataFrame(rows)

    # Deduplicate: keep best mean score per (mode, k, n) group
    if not df.empty:
        group_means = df.groupby(["mode", "k", "n", "file"])["score"].mean().reset_index()
        best_files = group_means.loc[group_means.groupby(["mode", "k", "n"])["score"].idxmax()][
            "file"
        ]
        df = df[df["file"].isin(best_files)]

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--facet", action="store_true", help="Facet by cluster")
    args = parser.parse_args()

    df = load_sweep_data(args.input_dir)
    if df.empty:
        print("No data found.")
        return

    # Create series label
    df["series"] = df["mode"] + " ($n$=" + df["n"].astype(str) + ")"

    # Aggregate: mean ± std of max score across clusters
    agg = (
        df.groupby(["series", "k"])
        .agg(
            mean_score=("score", "mean"),
            std_score=("score", "std"),
        )
        .reset_index()
    )

    if args.facet:
        plot = (
            p9.ggplot(df, p9.aes(x="factor(k)", y="score", color="series", group="series"))
            + p9.geom_line()
            + p9.geom_point(size=2)
            + p9.facet_wrap("~cluster", ncol=3)
            + p9.labs(x="$k$", y="Max score (Pearson $r$)", color="")
            + p9.theme(figure_size=(10, 3 * ((df["cluster"].nunique() + 2) // 3)))
            + p9.ylim(-0.1, 1.05)
        )
        out = args.output_dir / "sweep_min_highlights_facet.pdf"
    else:
        plot = (
            p9.ggplot(agg, p9.aes(x="factor(k)", y="mean_score", color="series", group="series"))
            + p9.geom_line()
            + p9.geom_point(size=2)
            + p9.labs(x="$k$", y="Mean max score (Pearson $r$)", color="")
            + p9.theme(figure_size=(3, 2.2))
        )
        out = args.output_dir / "sweep_min_highlights.pdf"

    out.parent.mkdir(parents=True, exist_ok=True)
    plot.save(out, dpi=300)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Summary plots for contrib locality: distance of top neuron to target, by layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd
import plotnine as p9

# Use Palatino locally, P052 on cluster
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Palatino", "P052", "serif"]

p9.theme_set(
    p9.theme_bw(base_size=10)
    + p9.theme(
        text=p9.element_text(color="#000"),
        axis_title=p9.element_text(size=10),
        axis_text=p9.element_text(size=8),
        axis_text_x=p9.element_text(angle=45, hjust=0.5),
        legend_text=p9.element_text(size=8),
        legend_title=p9.element_text(size=9),
        panel_grid_major=p9.element_line(size=1, color="#dddddd"),
        panel_grid_minor=p9.element_blank(),
        strip_background=p9.element_blank(),
        legend_margin=0,
    )
)


def main():
    parser = argparse.ArgumentParser(description="Plot contrib locality by layer.")
    parser.add_argument(
        "--results", default="/tmp/contrib_bs4_results.json", help="Results JSON path"
    )
    parser.add_argument("--output", default=None, help="Output PDF path")
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    # Flatten: one row per (target, layer)
    rows = []
    for entry in data:
        for pl in entry["per_layer"]:
            rows.append(
                {
                    "sample_idx": entry["sample_idx"],
                    "target_pos": entry["target_pos"],
                    "seq_len": entry["seq_len"],
                    "layer": pl["layer"],
                    "distance": pl["distance"],
                    "abs_distance": abs(pl["distance"]),
                    "attr_value": pl["attr_value"],
                }
            )
    df = pd.DataFrame(rows)

    output_path = Path(args.output or str(Path(args.results).parent / "contrib_summary.pdf"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: Fraction at distance 0 by layer ---
    df_frac = df.groupby("layer")["distance"].apply(lambda x: (x == 0).mean()).reset_index()
    df_frac.columns = ["layer", "frac_at_0"]

    p_frac = (
        p9.ggplot(df_frac, p9.aes(x="factor(layer)", y="frac_at_0"))
        + p9.geom_col(fill="#4daf4a", width=0.8)
        + p9.scale_y_continuous(limits=(0, 1))
        + p9.labs(x="Layer", y="Frac. with top contrib. @ target")
        + p9.theme(figure_size=(4, 2.5))
    )
    p_frac.save(output_path, dpi=150)
    print(f"Saved frac_at_0 to {output_path}")

    # --- Plot 2: Mean absolute distance by layer ---
    df_dist = df.groupby("layer")["abs_distance"].agg(["mean", "median"]).reset_index()

    dist_path = output_path.with_name(output_path.stem + "_dist.pdf")
    p_dist = (
        p9.ggplot(df_dist, p9.aes(x="factor(layer)", y="mean"))
        + p9.geom_col(fill="#377eb8", width=0.8)
        + p9.geom_point(p9.aes(y="median"), color="#e41a1c", size=1.5)
        + p9.labs(x="Layer", y="|Distance| to target (tokens)")
        + p9.theme(figure_size=(4, 2.5))
    )
    p_dist.save(dist_path, dpi=150)
    print(f"Saved distance to {dist_path}")

    # --- Plot 3: Distance boxplot by layer ---
    box_path = output_path.with_name(output_path.stem + "_boxplot.pdf")
    p_box = (
        p9.ggplot(df, p9.aes(x="factor(layer)", y="abs_distance"))
        + p9.geom_boxplot(outlier_shape="", alpha=0.6, fill="#ff7f00")
        + p9.coord_cartesian(ylim=(0, 30))
        + p9.labs(x="Layer", y="|Distance| to target (tokens)")
        + p9.theme(figure_size=(4, 2.5))
    )
    p_box.save(box_path, dpi=150)
    print(f"Saved boxplot to {box_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Summary plot: locality_k boxplots and BOS fraction boxplots by layer, side by side."""

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
    parser = argparse.ArgumentParser(description="Plot summary of attr locality by layer.")
    parser.add_argument(
        "--results", default="outputs/neuron_attr_locality/results.json", help="Results JSON path"
    )
    parser.add_argument("--output", default=None, help="Output PDF path")
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    # Normalize to flat list
    if isinstance(data, list) and len(data) > 0 and "layer" in data[0]:
        results = data
    elif isinstance(data, list):
        results = []
        for s in data:
            results.extend(s.get("results", []))
    else:
        results = data.get("results", [])

    df = pd.DataFrame(results)

    # Melt locality_k columns into long format
    ks = [1, 2, 4, 8]
    k_cols = [f"attr_locality_k{k}" for k in ks]
    df_loc = df.melt(
        id_vars=["layer"],
        value_vars=k_cols,
        var_name="k",
        value_name="locality",
    )
    df_loc["k"] = df_loc["k"].str.replace("attr_locality_k", "k=")

    # Bar chart with k=8 in back, k=1 in front (overlapping, not stacked)
    df_loc_stats = df_loc.groupby(["layer", "k"])["locality"].agg(["mean"]).reset_index()
    # Order so k=8 is drawn first (back), k=1 last (front)
    k_order = ["k=8", "k=4", "k=2", "k=1"]
    df_loc_stats["k"] = pd.Categorical(df_loc_stats["k"], categories=k_order, ordered=True)
    df_loc_stats = df_loc_stats.sort_values("k")

    p_loc = (
        p9.ggplot(df_loc_stats, p9.aes(x="factor(layer)", y="mean", fill="k"))
        + p9.geom_col(position="identity", width=0.8)
        + p9.scale_fill_brewer(type="qual", palette="Set1")
        + p9.scale_y_continuous(limits=(0, 1))
        + p9.labs(x="Layer", y="Fraction of |attr| (excl BOS)", fill="")
        + p9.theme(figure_size=(4, 2.5))
    )

    # BOS fraction boxplot
    p_bos = (
        p9.ggplot(df, p9.aes(x="factor(layer)", y="attr_bos_frac"))
        + p9.geom_boxplot(outlier_shape="", alpha=0.6, fill="#984ea3")
        + p9.scale_y_continuous(limits=(0, 1))
        + p9.labs(x="Layer", y="Fraction of |attr| at BOS")
        + p9.theme(figure_size=(4, 2.5))
    )

    # Entropy boxplot
    p_ent = (
        p9.ggplot(df, p9.aes(x="factor(layer)", y="attr_entropy"))
        + p9.geom_boxplot(outlier_shape="", alpha=0.6, fill="#ff7f00")
        + p9.labs(x="Layer", y="Entropy of |attr| (excl BOS)")
        + p9.theme(figure_size=(4, 2.5))
    )

    output_path = Path(args.output or str(Path(args.results).parent / "locality_summary.pdf"))
    p_loc.save(output_path, dpi=150)
    print(f"Saved locality to {output_path}")

    bos_path = output_path.with_name(output_path.stem + "_bos.pdf")
    p_bos.save(bos_path, dpi=150)
    print(f"Saved BOS to {bos_path}")

    ent_path = output_path.with_name(output_path.stem + "_entropy.pdf")
    p_ent.save(ent_path, dpi=150)
    print(f"Saved entropy to {ent_path}")


if __name__ == "__main__":
    main()

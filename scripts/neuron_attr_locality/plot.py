#!/usr/bin/env python
"""Visualize neuron attribution locality results with per-token bar charts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


def main():
    parser = argparse.ArgumentParser(description="Plot neuron attr locality results.")
    parser.add_argument(
        "--results", default="outputs/neuron_attr_locality/results.json", help="Results JSON path"
    )
    parser.add_argument("--output", default=None, help="Output PDF path (default: next to results)")
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma-separated layer indices to plot (default: 8 evenly spaced)",
    )
    parser.add_argument(
        "--polarity", default="pos", choices=["pos", "neg"], help="Which polarity to plot"
    )
    parser.add_argument(
        "--top-n", type=int, default=1, help="Show top-n instances per neuron (by activation)"
    )
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    # New format: flat list of results (each has its own 'tokens' field)
    # Old format: list of samples or dict — normalize to flat list
    if isinstance(data, list) and len(data) > 0:
        if "attr_vec" in data[0]:
            # Already flat list of results
            results = data
        else:
            # List of samples with nested results
            results = []
            for s in data:
                for r in s["results"]:
                    r["tokens"] = s["tokens"]
                    r["sample_idx"] = s.get("sample_idx", 0)
                results.extend(s["results"])
    else:
        results = data.get("results", [])
        tokens_default = data.get("tokens", [])
        for r in results:
            r.setdefault("tokens", tokens_default)

    # Filter by polarity
    results = [r for r in results if r["polarity"] == args.polarity]

    # Pick layers to show
    if args.layers:
        selected_layers = [int(x) for x in args.layers.split(",")]
    else:
        all_layers = sorted(set(r["layer"] for r in results))
        n = min(8, len(all_layers))
        indices = np.linspace(0, len(all_layers) - 1, n, dtype=int)
        selected_layers = [all_layers[i] for i in indices]

    # Group by (layer, neuron), pick top-n by activation magnitude
    from collections import defaultdict

    grouped = defaultdict(list)
    for r in results:
        if r["layer"] in selected_layers:
            grouped[(r["layer"], r["neuron"])].append(r)

    plot_results = []
    for key in sorted(grouped.keys()):
        entries = sorted(grouped[key], key=lambda x: abs(x["activation"]), reverse=True)
        plot_results.extend(entries[: args.top_n])

    n_plots = len(plot_results)
    if n_plots == 0:
        print("No results to plot")
        return

    fig, axes = plt.subplots(n_plots, 1, figsize=(20, 2.2 * n_plots), constrained_layout=True)
    if n_plots == 1:
        axes = [axes]

    cmap = plt.cm.RdBu_r

    for i, r in enumerate(plot_results):
        layer = r["layer"]
        neuron = r["neuron"]
        token_pos = r["token_pos"]
        attr_vec = np.array(r["attr_vec"])
        tokens = r.get("tokens", [f"{j}" for j in range(len(attr_vec))])
        sample_idx = r.get("sample_idx", "?")

        ax = axes[i]
        vmax = max(abs(attr_vec.min()), abs(attr_vec.max())) or 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        ax.bar(range(len(attr_vec)), attr_vec, color=cmap(norm(attr_vec)), width=1.0)
        ax.axvline(token_pos, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_ylabel("attr", fontsize=8)
        bos_frac = r.get("attr_bos_frac", 0.0)
        loc_k1 = r.get("attr_locality_k1", r.get("attr_locality", 0.0))
        loc_k8 = r.get("attr_locality_k8", 0.0)
        ax.set_title(
            f"L{layer} N{neuron} {r['polarity']} [sample {sample_idx}] — "
            f"k1={loc_k1:.3f}, k8={loc_k8:.3f}, bos={bos_frac:.3f}, "
            f"ent={r['attr_entropy']:.2f}, act={r['activation']:.4f}",
            fontsize=9,
        )
        ax.set_xlim(-0.5, len(tokens) - 0.5)
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=4, ha="center")
        ax.tick_params(axis="y", labelsize=6)

    output_path = args.output or str(Path(args.results).parent / "locality_plot.pdf")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

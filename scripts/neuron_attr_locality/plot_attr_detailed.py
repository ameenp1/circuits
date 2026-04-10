#!/usr/bin/env python
"""Faceted per-sample attr_vec bar charts for specific (layer, neuron, polarity) tuples.

Requires --detailed output from run_attr.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Palatino", "P052", "serif"]


def plot_neuron(
    data: list[dict],
    layer: int,
    neuron: int,
    polarity: str,
    max_entries: int,
    output_dir: Path,
):
    entries = [
        r
        for r in data
        if r["layer"] == layer and r["neuron"] == neuron and r["polarity"] == polarity
    ]
    entries.sort(key=lambda r: -abs(r["activation"]))
    entries = entries[:max_entries]
    n = len(entries)
    if n == 0:
        print(f"No entries for L{layer}/N{neuron}/{polarity}")
        return

    fig, axes = plt.subplots(n, 1, figsize=(20, n * 1.8), constrained_layout=True)
    fig.suptitle(f"L{layer}/N{neuron}/{polarity} — top {n} activations", fontsize=12)
    if n == 1:
        axes = [axes]

    for i, (ax, r) in enumerate(zip(axes, entries)):
        av = np.array(r["attr_vec"])
        abs_total = np.abs(av).sum()
        av_norm = av / abs_total if abs_total > 0 else av
        seq_len = len(av)
        tokens = r["tokens"]

        vmax = np.max(np.abs(av_norm))
        if vmax == 0:
            vmax = 1e-6

        colors = ["#d73027" if v > 0 else "#4575b4" for v in av_norm]
        ax.bar(range(seq_len), av_norm, color=colors, width=1.0, edgecolor="none")
        ax.axvline(r["token_pos"], color="black", linewidth=1.2, linestyle="--", alpha=0.8)
        ax.set_ylim(-vmax * 1.1, vmax * 1.1)
        ax.set_xlim(-0.5, seq_len - 0.5)

        xlabels = [t.replace("\n", "\\n") for t in tokens]
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(xlabels, fontsize=3.5, rotation=90, ha="center")
        ax.set_ylabel(
            f"s{r['sample_idx']} p{r['token_pos']}\n"
            f"act={r['activation']:.3f}\n"
            f"k1={r['attr_locality_k1']:.2f}",
            fontsize=6,
        )
        ax.tick_params(axis="y", labelsize=5)

    out = output_dir / f"L{layer}_N{neuron}_{polarity}_faceted.pdf"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(description="Plot detailed attr_vec per sample.")
    parser.add_argument(
        "--results", required=True, help="Path to detailed results.json from run_attr.py"
    )
    parser.add_argument(
        "--neurons",
        required=True,
        help="Comma-separated layer:neuron:polarity tuples, e.g. '0:137:neg,31:5:pos'",
    )
    parser.add_argument("--max-entries", type=int, default=10, help="Max samples per neuron")
    parser.add_argument(
        "--output-dir", default=None, help="Output directory (default: same as results)"
    )
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.results).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for spec in args.neurons.split(","):
        parts = spec.strip().split(":")
        layer, neuron, polarity = int(parts[0]), int(parts[1]), parts[2]
        plot_neuron(data, layer, neuron, polarity, args.max_entries, output_dir)


if __name__ == "__main__":
    main()

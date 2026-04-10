#!/usr/bin/env python
"""Heatmap visualization for single-neuron analysis results.

For each exemplar, shows two rows:
- Input attribution: color each token by its attr_vec value (what caused the neuron to fire)
- Output contribution: color each future token by its contrib_vec value (what the neuron contributes to)

The firing position is marked with a box/indicator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


def make_heatmap(data: dict, output_path: Path):
    config = data["config"]
    results = data["results"]
    n = len(results)

    fig, axes = plt.subplots(n, 2, figsize=(20, 1.8 * n), squeeze=False)
    fig.suptitle(
        f"Layer {config['layer']}, Neuron {config['neuron']}, "
        f"Polarity {config['polarity']}\n"
        f"Model: {config['model_id']}",
        fontsize=12,
        fontweight="bold",
        y=1.0,
    )

    for i, r in enumerate(results):
        tokens = r["tokens"]
        attr_vec = np.array(r["attr_vec"])
        contrib_vec = np.array(r["contrib_vec"])
        token_pos = r["token_pos"]
        seq_len = r["seq_len"]
        contrib_positions = r["contrib_positions"]

        # --- Left: input attribution heatmap ---
        ax_attr = axes[i, 0]
        attr_2d = attr_vec.reshape(1, -1)
        vmax_attr = max(abs(attr_vec.min()), abs(attr_vec.max()), 1e-8)
        norm_attr = TwoSlopeNorm(vmin=-vmax_attr, vcenter=0, vmax=vmax_attr)
        im_attr = ax_attr.imshow(attr_2d, aspect="auto", cmap="RdBu_r", norm=norm_attr)

        ax_attr.set_yticks([])
        ax_attr.set_xticks(range(seq_len))
        ax_attr.set_xticklabels(
            [t.replace("\n", "\\n") for t in tokens],
            rotation=90,
            fontsize=5,
            fontfamily="monospace",
        )

        # Mark the firing position
        ax_attr.axvline(x=token_pos, color="black", linewidth=1.5, linestyle="--", alpha=0.7)

        title_attr = (
            f"#{r['rank']} input attr | sample={r['sample_idx']} pos={token_pos} "
            f"act={r['activation']:.3f} cr={r['credit_ratio']:.3f}"
        )
        ax_attr.set_title(title_attr, fontsize=7, loc="left")

        # --- Right: output contribution heatmap ---
        ax_contrib = axes[i, 1]

        # Build a full-sequence contrib array (NaN for non-future positions)
        contrib_full = np.full(seq_len, np.nan)
        for j, pos in enumerate(contrib_positions):
            # contrib_vec[j] corresponds to the contribution to the logit at pos predicting pos+1
            # Label with the predicted token (pos+1)
            contrib_full[pos] = contrib_vec[j]

        contrib_2d = contrib_full.reshape(1, -1)
        # Mask NaN for display
        masked = np.ma.masked_invalid(contrib_2d)
        valid = contrib_full[~np.isnan(contrib_full)]
        vmax_contrib = max(abs(valid.min()), abs(valid.max()), 1e-8) if len(valid) > 0 else 1e-8
        norm_contrib = TwoSlopeNorm(vmin=-vmax_contrib, vcenter=0, vmax=vmax_contrib)

        cmap_contrib = plt.cm.RdBu_r.copy()
        cmap_contrib.set_bad(color="0.9")
        im_contrib = ax_contrib.imshow(masked, aspect="auto", cmap=cmap_contrib, norm=norm_contrib)

        ax_contrib.set_yticks([])
        ax_contrib.set_xticks(range(seq_len))
        # Label x-axis with the *predicted* token (pos+1) for future positions
        xlabels = []
        for p in range(seq_len):
            if p in contrib_positions and p + 1 < seq_len:
                xlabels.append(tokens[p + 1].replace("\n", "\\n"))
            else:
                xlabels.append("")
        ax_contrib.set_xticklabels(xlabels, rotation=90, fontsize=5, fontfamily="monospace")

        ax_contrib.axvline(x=token_pos, color="black", linewidth=1.5, linestyle="--", alpha=0.7)

        if len(contrib_vec) > 0:
            title_contrib = (
                f"#{r['rank']} output contrib | "
                f"future_len={len(contrib_vec)} "
                f"max={max(contrib_vec):.3f} min={min(contrib_vec):.3f}"
            )
        else:
            title_contrib = f"#{r['rank']} output contrib | future_len=0 (last token)"
        ax_contrib.set_title(title_contrib, fontsize=7, loc="left")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Heatmap for single-neuron analysis results.")
    parser.add_argument("input", type=str, help="Path to JSON results file")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (default: same dir as input, .png)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    with open(input_path) as f:
        data = json.load(f)

    if args.output is None:
        output_path = input_path.with_suffix(".png")
    else:
        output_path = Path(args.output)

    make_heatmap(data, output_path)


if __name__ == "__main__":
    main()

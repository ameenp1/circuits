"""Plot per-layer similarity between attr and activation descriptions."""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from circuits.utils.constants import RESULTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comparison",
        type=str,
        default=str(RESULTS_DIR / "neuron_attr_locality/fineweb/comparison.json"),
    )
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--metric", type=str, default="max", choices=["max", "mean"])
    args = parser.parse_args()

    with open(args.comparison) as f:
        data = json.load(f)

    results = data["results"]
    key = "max_similarity" if args.metric == "max" else "mean_similarity"

    # Group by layer
    layer_sims: dict[int, list[float]] = {}
    for r in results:
        layer = r["layer"]
        layer_sims.setdefault(layer, []).append(r[key])

    layers = sorted(layer_sims.keys())
    means = [np.mean(layer_sims[l]) for l in layers]
    stds = [np.std(layer_sims[l]) for l in layers]
    counts = [len(layer_sims[l]) for l in layers]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(layers, means, yerr=stds, capsize=2, color="#4C72B0", alpha=0.8, edgecolor="white")

    # Add count labels
    for l, m, c in zip(layers, means, counts):
        ax.text(l, m + 0.02, str(c), ha="center", va="bottom", fontsize=6, color="gray")

    ax.set_xlabel("Layer")
    ax.set_ylabel(f"{'Best' if args.metric == 'max' else 'Mean'} cosine similarity")
    ax.set_title(f"Attr vs. activation description similarity by layer (n={len(results)})")
    ax.axhline(
        np.mean([r[key] for r in results]),
        color="red",
        linestyle="--",
        alpha=0.5,
        label="overall mean",
    )
    ax.legend()
    ax.set_xticks(layers)
    fig.tight_layout()

    if not args.output:
        args.output = args.comparison.replace(
            "comparison.json", f"similarity_by_layer_{args.metric}.png"
        )

    fig.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

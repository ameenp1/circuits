"""Compare attr-based vs activation-based descriptions from describe_neurons.py.

Loads descriptions.json (which has both attr_explanations and act_explanations per neuron)
and computes pairwise sentence embedding similarity between the two views.

Usage:
    python scripts/neuron_attr_locality/compare_descriptions.py \
        --descriptions /path/to/descriptions.json \
        --output /path/to/comparison.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from circuits.utils.constants import RESULTS_DIR
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(line_buffering=True)


def log(msg: str) -> None:
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {msg}")


def compute_similarity_matrix(
    texts_a: list[str], texts_b: list[str], model: SentenceTransformer
) -> np.ndarray:
    """Compute cosine similarity between each pair (a_i, b_j)."""
    emb_a = model.encode(texts_a, normalize_embeddings=True, show_progress_bar=False)
    emb_b = model.encode(texts_b, normalize_embeddings=True, show_progress_bar=False)
    return emb_a @ emb_b.T


def collect_explanations(data: dict, view: str) -> list[str]:
    """Collect all explanations for a view (pos + neg signs)."""
    expls = []
    expl_key = f"{view}_explanations"
    if expl_key not in data:
        # Fallback for old format
        if view == "attr" and "explanations" in data:
            expl_key = "explanations"
        else:
            return []
    for sign in ("pos", "neg"):
        for e in data[expl_key].get(sign, []):
            if isinstance(e, str) and e.strip():
                expls.append(e.strip())
    return expls


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare attr vs activation descriptions.")
    parser.add_argument(
        "--descriptions",
        type=str,
        default=str(RESULTS_DIR / "neuron_attr_locality/fineweb/descriptions.json"),
        help="Path to descriptions.json from describe_neurons.py",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model for similarity",
    )
    parser.add_argument("--output", type=str, default="", help="Output path for comparison JSON")
    args = parser.parse_args()

    # Load descriptions
    log(f"Loading descriptions from {args.descriptions}")
    with open(args.descriptions) as f:
        desc_data = json.load(f)

    log(f"Loaded descriptions for {len(desc_data)} neurons")

    # Load sentence transformer
    log(f"Loading sentence transformer: {args.embedding_model}")
    st_model = SentenceTransformer(args.embedding_model)

    # Compare attr vs act descriptions for each neuron
    results = []
    skipped = {"no_attr": 0, "no_act": 0}
    for key, data in sorted(desc_data.items()):
        nid = data["neuron_id"]

        attr_expls = collect_explanations(data, "attr")
        act_expls = collect_explanations(data, "act")

        if not attr_expls:
            skipped["no_attr"] += 1
            continue
        if not act_expls:
            skipped["no_act"] += 1
            continue

        # Compute similarity: each attr explanation vs each act explanation
        sim_matrix = compute_similarity_matrix(attr_expls, act_expls, st_model)

        # Best match: max over all pairs
        best_idx = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
        max_sim = float(sim_matrix[best_idx])
        mean_sim = float(np.mean(sim_matrix))

        # Per-attr best: for each attr expl, best match among act expls
        per_attr_max = sim_matrix.max(axis=1).tolist()
        # Per-act best: for each act expl, best match among attr expls
        per_act_max = sim_matrix.max(axis=0).tolist()

        result = {
            "neuron_key": key,
            "layer": nid["layer"],
            "neuron": nid["neuron"],
            "polarity": nid["polarity"],
            "attr_explanations": attr_expls,
            "act_explanations": act_expls,
            "mean_similarity": mean_sim,
            "max_similarity": max_sim,
            "best_attr_explanation": attr_expls[best_idx[0]],
            "best_act_explanation": act_expls[best_idx[1]],
            "per_attr_max": per_attr_max,
            "per_act_max": per_act_max,
        }
        results.append(result)

        log(
            f"  {key}: mean={mean_sim:.3f}, max={max_sim:.3f}"
            f"\n    Attr: {attr_expls[best_idx[0]][:100]}"
            f"\n    Act:  {act_expls[best_idx[1]][:100]}"
        )

    # Summary statistics
    if results:
        mean_sims = [r["mean_similarity"] for r in results]
        max_sims = [r["max_similarity"] for r in results]
        log(f"\n{'=' * 80}")
        log(f"Summary: {len(results)} neurons compared")
        if skipped["no_attr"]:
            log(f"  Skipped (no attr): {skipped['no_attr']}")
        if skipped["no_act"]:
            log(f"  Skipped (no act): {skipped['no_act']}")
        log(f"  Mean similarity (avg): {np.mean(mean_sims):.3f} +/- {np.std(mean_sims):.3f}")
        log(f"  Max similarity (avg):  {np.mean(max_sims):.3f} +/- {np.std(max_sims):.3f}")
        log(f"  Best single match: {max(max_sims):.3f}")
        log(f"  Worst single match: {min(max_sims):.3f}")
        log(f"{'=' * 80}")

        # Print ranked table
        results.sort(key=lambda r: r["max_similarity"], reverse=True)
        log(f"\n{'Neuron':<25} {'MaxSim':>7} {'MeanSim':>8} " f"{'Best Attr':<40} {'Best Act':<40}")
        log("-" * 125)
        for r in results:
            attr_short = r["best_attr_explanation"][:38]
            act_short = r["best_act_explanation"][:38]
            log(
                f"{r['neuron_key']:<25} {r['max_similarity']:>7.3f} "
                f"{r['mean_similarity']:>8.3f} {attr_short:<40} {act_short:<40}"
            )
    else:
        log("No neurons could be compared")

    # Save
    if not args.output:
        results_dir = Path(args.descriptions).parent
        args.output = str(results_dir / "comparison.json")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "embedding_model": args.embedding_model,
        "num_neurons": len(results),
        "mean_similarity_avg": float(np.mean(mean_sims)) if results else None,
        "max_similarity_avg": float(np.mean(max_sims)) if results else None,
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    log(f"Saved comparison to {output_path}")


if __name__ == "__main__":
    main()

"""
Check the normalization of attr_map and contrib_map in circuits.

This script loads a circuit and verifies:
1. attr_map normalization: sum(attr_map) should be ~1.0 after dividing by activation
2. contrib_map normalization: sum(contrib_map) should be ~1.0 after dividing by logit values

The normalization happens in Circuit._normalize_attr_contrib_maps() during __init__.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from circuits.analysis.circuit_ops import Circuit
from circuits.utils.constants import RESULTS_DIR

CIRCUIT_PICKLE = RESULTS_DIR / "case_studies/texas_circuit.pkl"


def load_raw_df_node(pickle_path: Path) -> pd.DataFrame:
    """Load the raw df_node from pickle without normalization."""
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    # Circuit pickles store df_node directly or in a dict
    if isinstance(data, dict):
        return data.get("df_node", pd.DataFrame())
    elif hasattr(data, "df_node"):
        return data.df_node
    else:
        return pd.DataFrame()


def main() -> None:
    if not CIRCUIT_PICKLE.exists():
        print(f"Missing circuit pickle at {CIRCUIT_PICKLE}")
        return

    # Load raw df_node (before normalization)
    print("Loading raw data from pickle...")
    df_raw = load_raw_df_node(CIRCUIT_PICKLE)
    print(f"Raw df_node: {len(df_raw)} entries")

    # Load circuit (with normalization)
    print("Loading circuit (with normalization)...")
    circuit = Circuit.load_from_pickle(str(CIRCUIT_PICKLE))
    df = circuit.df_node

    print(f"Loaded circuit with {len(df)} neuron entries")
    print(f"Columns: {list(df.columns)}")

    # Filter to MLP neurons (not embeddings or logits)
    num_layers = df["layer"].max()
    df_mlp = df[(df["layer"] >= 0) & (df["layer"] < num_layers)].copy()
    print(f"MLP neurons: {len(df_mlp)}")

    # Check attr_map and activation columns exist
    if "attr_map" not in df_mlp.columns or "activation" not in df_mlp.columns:
        print("Missing attr_map or activation columns")
        return

    # Debug: check what types are in attr_map
    print("\nDebug: attr_map types and samples")
    sample_attr_maps = df_mlp["attr_map"].head(5).tolist()
    for i, am in enumerate(sample_attr_maps):
        if hasattr(am, "shape"):
            print(f"  [{i}] type={type(am).__name__}, shape={am.shape}")
        else:
            print(f"  [{i}] type={type(am).__name__}, value={am}")

    # Debug: check activation types
    print("\nDebug: activation types and samples")
    sample_acts = df_mlp["activation"].head(5).tolist()
    for i, act in enumerate(sample_acts):
        if hasattr(act, "shape"):
            print(f"  [{i}] type={type(act).__name__}, shape={act.shape}, mean={act.mean():.4f}")
        else:
            print(f"  [{i}] type={type(act).__name__}, value={act}")

    # Compute sum of attr_map for each neuron
    results = []
    for _, row in df_mlp.iterrows():
        attr_map = row["attr_map"]
        activation = row["activation"]

        # Handle different types
        if attr_map is None:
            continue
        if isinstance(attr_map, (list, tuple)):
            attr_map = np.array(attr_map)
        if not isinstance(attr_map, np.ndarray):
            # Try converting torch tensor
            if hasattr(attr_map, "numpy"):
                attr_map = attr_map.numpy()
            elif hasattr(attr_map, "cpu"):
                attr_map = attr_map.cpu().numpy()
            else:
                continue

        # Handle activation - might be tensor with batch dim
        if hasattr(activation, "numpy"):
            activation = activation.numpy()
        elif hasattr(activation, "cpu"):
            activation = activation.cpu().numpy()

        # If attr_map has batch dimension (batch, src_tokens), handle per-batch
        if attr_map.ndim == 2:
            # Sum over src_tokens dimension, keep batch
            attr_sums = np.sum(attr_map, axis=-1)  # (batch,)
            activations = np.atleast_1d(activation)  # (batch,)

            for b in range(len(attr_sums)):
                act_val = (
                    float(activations[b]) if b < len(activations) else float(activations.mean())
                )
                attr_sum_val = float(attr_sums[b])
                results.append(
                    {
                        "layer": row["layer"],
                        "token": row["token"],
                        "neuron": row["neuron"],
                        "label": row.get("label", ""),
                        "batch": b,
                        "activation": act_val,
                        "attr_sum": attr_sum_val,
                        "ratio": attr_sum_val / (act_val + 1e-10) if act_val != 0 else float("nan"),
                        "diff": attr_sum_val - act_val,
                    }
                )
        else:
            # 1D case - single value
            attr_sum = float(np.sum(attr_map))
            act_val = float(activation) if np.isscalar(activation) else float(np.mean(activation))
            results.append(
                {
                    "layer": row["layer"],
                    "token": row["token"],
                    "neuron": row["neuron"],
                    "label": row.get("label", ""),
                    "batch": 0,
                    "activation": act_val,
                    "attr_sum": attr_sum,
                    "ratio": attr_sum / (act_val + 1e-10) if act_val != 0 else float("nan"),
                    "diff": attr_sum - act_val,
                }
            )

    df_results = pd.DataFrame(results)
    print(f"\nAnalyzed {len(df_results)} neurons with valid attr_map")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    print(f"\nActivation stats:")
    print(f"  mean: {df_results['activation'].mean():.6f}")
    print(f"  std:  {df_results['activation'].std():.6f}")
    print(f"  min:  {df_results['activation'].min():.6f}")
    print(f"  max:  {df_results['activation'].max():.6f}")

    print(f"\nattr_sum stats:")
    print(f"  mean: {df_results['attr_sum'].mean():.6f}")
    print(f"  std:  {df_results['attr_sum'].std():.6f}")
    print(f"  min:  {df_results['attr_sum'].min():.6f}")
    print(f"  max:  {df_results['attr_sum'].max():.6f}")

    print(f"\nRatio (attr_sum / activation) stats:")
    # Filter out extreme ratios from division by near-zero activations
    valid_ratios = df_results[df_results["activation"].abs() > 0.01]["ratio"]
    print(f"  mean: {valid_ratios.mean():.6f}")
    print(f"  std:  {valid_ratios.std():.6f}")
    print(f"  min:  {valid_ratios.min():.6f}")
    print(f"  max:  {valid_ratios.max():.6f}")

    print(f"\nDiff (attr_sum - activation) stats:")
    print(f"  mean: {df_results['diff'].mean():.6f}")
    print(f"  std:  {df_results['diff'].std():.6f}")
    print(f"  min:  {df_results['diff'].min():.6f}")
    print(f"  max:  {df_results['diff'].max():.6f}")

    # Correlation
    corr = df_results["activation"].corr(df_results["attr_sum"])
    print(f"\nCorrelation(activation, attr_sum): {corr:.6f}")

    # Show some examples
    print("\n" + "=" * 60)
    print("SAMPLE NEURONS (sorted by |activation|)")
    print("=" * 60)

    df_sorted = df_results.reindex(
        df_results["activation"].abs().sort_values(ascending=False).index
    )
    print(
        df_sorted[["layer", "token", "neuron", "activation", "attr_sum", "ratio"]]
        .head(20)
        .to_string()
    )

    # Check if ratio is consistently ~1 (would mean attr_sum ≈ activation)
    print("\n" + "=" * 60)
    print("RATIO DISTRIBUTION (for |activation| > 0.01)")
    print("=" * 60)

    valid_df = df_results[df_results["activation"].abs() > 0.01]
    ratio_bins = [0, 0.5, 0.9, 1.0, 1.1, 2.0, float("inf")]
    ratio_labels = ["<0.5", "0.5-0.9", "0.9-1.0", "1.0-1.1", "1.1-2.0", ">2.0"]
    valid_df = valid_df.copy()
    valid_df["ratio_bin"] = pd.cut(valid_df["ratio"], bins=ratio_bins, labels=ratio_labels)
    print(valid_df["ratio_bin"].value_counts().sort_index())

    # ========================================
    # CONTRIB_MAP ANALYSIS
    # ========================================
    print("\n")
    print("=" * 60)
    print("CONTRIB_MAP ANALYSIS")
    print("=" * 60)

    # Get logit layer info
    logit_layer = df["layer"].max()
    df_logits = df[df["layer"] == logit_layer]
    print(f"\nLogit layer: {logit_layer}")
    print(f"Logit nodes: {len(df_logits)}")

    # Build logit lookup by label
    logit_by_label: dict[str, list[float]] = {}
    for label in df_logits["label"].unique():
        label_logits = df_logits[df_logits["label"] == label].sort_values("token")
        logit_by_label[label] = label_logits["activation"].tolist()

    # Show sample logit activations
    print("\nSample logit nodes (these are used to normalize contrib_map):")
    for _, row in df_logits.head(5).iterrows():
        print(
            f"  label={row['label']}, token={row['token']}, neuron={row['neuron']}, act={row['activation']:.4f}"
        )

    # Show raw vs normalized attr_map and contrib_map for a few neurons
    print("\n" + "-" * 60)
    print("RAW VS NORMALIZED MAPS (first 5 MLP neurons)")
    print("-" * 60)

    df_mlp_raw_maps = df_raw[(df_raw["layer"] >= 0) & (df_raw["layer"] < logit_layer)].head(5)

    for _, raw_row in df_mlp_raw_maps.iterrows():
        label = str(raw_row["label"])
        # Find corresponding normalized row
        norm_rows = df_mlp[
            (df_mlp["layer"] == raw_row["layer"])
            & (df_mlp["token"] == raw_row["token"])
            & (df_mlp["neuron"] == raw_row["neuron"])
            & (df_mlp["label"] == label)
        ]
        if len(norm_rows) == 0:
            continue
        norm_row = norm_rows.iloc[0]

        activation = raw_row["activation"]
        logit_vals = np.asarray(logit_by_label.get(label, []))

        print(f"\n  L{raw_row['layer']}/T{raw_row['token']}/N{raw_row['neuron']} ({label[:20]}...)")
        print(f"    activation: {activation:.6f}")

        # attr_map comparison
        raw_attr = raw_row.get("attr_map")
        norm_attr = norm_row.get("attr_map")
        if raw_attr is not None and norm_attr is not None:
            raw_attr = np.asarray(raw_attr)
            norm_attr = np.asarray(norm_attr)
            print(f"    --- attr_map (normalized by activation={activation:.4f}) ---")
            print(f"    raw:  sum={np.sum(raw_attr):.6f}, first5={raw_attr[:5]}")
            print(f"    norm: sum={np.sum(norm_attr):.6f}, first5={norm_attr[:5]}")

        # contrib_map comparison
        raw_contrib = raw_row.get("contrib_map")
        norm_contrib = norm_row.get("contrib_map")
        if raw_contrib is not None and norm_contrib is not None:
            raw_contrib = np.asarray(raw_contrib)
            norm_contrib = np.asarray(norm_contrib)
            print(f"    --- contrib_map (normalized by logits={logit_vals}) ---")
            print(f"    raw:  sum={np.sum(raw_contrib):.6f}, vals={raw_contrib}")
            print(f"    norm: sum={np.sum(norm_contrib):.6f}, vals={norm_contrib}")

    # Analyze contrib_map for MLP neurons
    contrib_results = []
    for _, row in df_mlp.iterrows():
        contrib_map = row.get("contrib_map")
        if contrib_map is None:
            continue

        # Convert to numpy array
        if isinstance(contrib_map, (list, tuple)):
            contrib_map = np.array(contrib_map)
        if not isinstance(contrib_map, np.ndarray):
            if hasattr(contrib_map, "numpy"):
                contrib_map = contrib_map.numpy()
            elif hasattr(contrib_map, "cpu"):
                contrib_map = contrib_map.cpu().numpy()
            else:
                continue

        contrib_sum = float(np.sum(contrib_map))
        contrib_abs_sum = float(np.sum(np.abs(contrib_map)))

        contrib_results.append(
            {
                "layer": row["layer"],
                "token": row["token"],
                "neuron": row["neuron"],
                "label": row.get("label", ""),
                "activation": row["activation"],
                "contrib_sum": contrib_sum,
                "contrib_abs_sum": contrib_abs_sum,
                "contrib_len": len(contrib_map),
            }
        )

    df_contrib = pd.DataFrame(contrib_results)
    print(f"\nAnalyzed {len(df_contrib)} neurons with valid contrib_map")

    if len(df_contrib) > 0:
        print(f"\ncontrib_sum stats (should be ~1.0 if normalized):")
        print(f"  mean: {df_contrib['contrib_sum'].mean():.6f}")
        print(f"  std:  {df_contrib['contrib_sum'].std():.6f}")
        print(f"  min:  {df_contrib['contrib_sum'].min():.6f}")
        print(f"  max:  {df_contrib['contrib_sum'].max():.6f}")

        print(f"\ncontrib_abs_sum stats:")
        print(f"  mean: {df_contrib['contrib_abs_sum'].mean():.6f}")
        print(f"  std:  {df_contrib['contrib_abs_sum'].std():.6f}")
        print(f"  min:  {df_contrib['contrib_abs_sum'].min():.6f}")
        print(f"  max:  {df_contrib['contrib_abs_sum'].max():.6f}")

        print(f"\ncontrib_len (number of target logits):")
        print(f"  unique values: {sorted(df_contrib['contrib_len'].unique())}")

        # Show sample contrib_maps
        print("\n" + "=" * 60)
        print("SAMPLE CONTRIB_MAPS (sorted by |activation|)")
        print("=" * 60)

        df_contrib_sorted = df_contrib.sort_values(
            by="activation", key=lambda x: x.abs(), ascending=False
        )
        print(
            df_contrib_sorted[
                ["layer", "token", "neuron", "activation", "contrib_sum", "contrib_abs_sum"]
            ]
            .head(20)
            .to_string()
        )

        # Distribution of contrib_sum
        print("\n" + "=" * 60)
        print("CONTRIB_SUM DISTRIBUTION")
        print("=" * 60)

        contrib_bins = [-float("inf"), -1.0, -0.1, 0.0, 0.1, 1.0, 2.0, float("inf")]
        contrib_labels = ["<-1", "-1 to -0.1", "-0.1 to 0", "0 to 0.1", "0.1 to 1", "1 to 2", ">2"]
        df_contrib_copy = df_contrib.copy()
        df_contrib_copy["contrib_bin"] = pd.cut(
            df_contrib_copy["contrib_sum"], bins=contrib_bins, labels=contrib_labels
        )
        print(df_contrib_copy["contrib_bin"].value_counts().sort_index())


if __name__ == "__main__":
    main()

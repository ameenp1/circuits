"""Generate a LaTeX table comparing human vs automatic cluster descriptions.

Takes one or more human-scored JSONs and one or more automatic (v2) JSONs, and
produces a table with attr and contrib scores side by side.

Usage:
    # Defaults: latest human + latest auto (VLLM + API) from hardcoded results folders
    python scripts/ablations/descriptions/3_table_human_vs_auto.py

    # Explicit paths
    python scripts/ablations/descriptions/3_table_human_vs_auto.py \
        --human Human=/path/to/human_vllm.json \
        --human Human-API=/path/to/human_api.json \
        --auto V2-VLLM=/path/to/auto_vllm.json \
        --auto V2-API=/path/to/auto_api.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from circuits.utils.constants import RESULTS_DIR

HUMAN_DIR = RESULTS_DIR / "case_studies/capitals/human_descriptions"
AUTO_DIR = RESULTS_DIR / "case_studies/capitals/human_vs_auto"


def best_score(sign_data: dict) -> float | None:
    """Extract best score across pos/neg/combined signs."""
    best = None
    for sign in ("pos", "neg", "combined"):
        for expl in sign_data.get(sign, []):
            s = expl.get("score") if isinstance(expl, dict) else None
            if s is not None and (best is None or s > best):
                best = s
    return best


def load_human(path: Path) -> dict[str, dict[str, float | None]]:
    """Load human description scores. Returns {cluster: {attr: score, contrib: score}}."""
    with open(path) as f:
        d = json.load(f)
    out: dict[str, dict[str, float | None]] = {}
    for cname, cdata in d.get("results", {}).items():
        out[cname] = {
            "attr": cdata.get("attr_score"),
            "contrib": cdata.get("contrib_score"),
        }
    return out


def load_auto(path: Path) -> dict[str, dict[str, float | None]]:
    """Load automatic v2 description scores. Returns {cluster: {attr: score, contrib: score}}."""
    with open(path) as f:
        d = json.load(f)
    out: dict[str, dict[str, float | None]] = {}
    attr_data = d.get("attr", {})
    contrib_data = d.get("contrib", {})
    all_clusters = set(attr_data.keys()) | set(contrib_data.keys())
    for cname in all_clusters:
        out[cname] = {
            "attr": best_score(attr_data.get(cname, {})),
            "contrib": best_score(contrib_data.get(cname, {})),
        }
    return out


def fmt(val: float | None, bold: bool = False) -> str:
    if val is None:
        return "---"
    s = f"{val:.3f}"
    return rf"\textbf{{{s}}}" if bold else s


def best_idx(vals: list[float | None]) -> int | None:
    """Return index of the highest non-None value, or None if all are None."""
    best_i = None
    best_v = None
    for i, v in enumerate(vals):
        if v is not None and (best_v is None or v > best_v):
            best_i = i
            best_v = v
    return best_i


def latest_json(directory: Path, prefix: str = "") -> Path:
    """Return the most recently modified .json file in a directory, optionally filtered by prefix."""
    pattern = f"{prefix}*.json" if prefix else "*.json"
    jsons = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not jsons:
        raise FileNotFoundError(f"No {pattern} files found in {directory}")
    return jsons[-1]


def latest_json_optional(directory: Path, prefix: str = "") -> Path | None:
    """Like latest_json but returns None instead of raising."""
    try:
        return latest_json(directory, prefix)
    except FileNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--human",
        nargs="*",
        default=None,
        help=(
            "name=path pairs for human scores "
            f"(default: Human=latest in {HUMAN_DIR}, "
            f"Human-API=latest human_descriptions_api_* if available)"
        ),
    )
    parser.add_argument(
        "--auto",
        nargs="*",
        default=None,
        help=(
            "name=path pairs for auto scores "
            f"(default: VLLM=latest auto_descriptions_vllm_*, "
            f"API=latest auto_descriptions_api_* in {AUTO_DIR})"
        ),
    )
    parser.add_argument("--exclude", nargs="*", default=["__unclustered__"])
    args = parser.parse_args()

    # Load human scores
    human_names: list[str] = []
    human_scores_list: list[dict[str, dict[str, float | None]]] = []
    if args.human:
        for item in args.human:
            name, path = item.split("=", 1)
            human_names.append(name)
            human_scores_list.append(load_human(Path(path)))
            print(f"Human ({name}): {path}", file=sys.stderr)
    else:
        # Default: load latest human (VLLM-scored), and API-scored if available
        human_path = latest_json(HUMAN_DIR, prefix="human_descriptions_2")
        human_names.append("Human")
        human_scores_list.append(load_human(human_path))
        print(f"Human: {human_path}", file=sys.stderr)

        human_api_path = latest_json_optional(HUMAN_DIR, prefix="human_descriptions_api_")
        if human_api_path is not None:
            human_names.append("Human-API")
            human_scores_list.append(load_human(human_api_path))
            print(f"Human-API: {human_api_path}", file=sys.stderr)

    # Load auto scores
    auto_names: list[str] = []
    auto_scores_list: list[dict[str, dict[str, float | None]]] = []
    if args.auto:
        for item in args.auto:
            name, path = item.split("=", 1)
            auto_names.append(name)
            auto_scores_list.append(load_auto(Path(path)))
            print(f"Auto ({name}): {path}", file=sys.stderr)
    else:
        # Default: try to load both VLLM and API auto results
        vllm_path = latest_json_optional(AUTO_DIR, prefix="auto_descriptions_vllm_")
        api_path = latest_json_optional(AUTO_DIR, prefix="auto_descriptions_api_")

        if vllm_path is None and api_path is None:
            # Fall back to any auto_descriptions file
            fallback_path = latest_json(AUTO_DIR, prefix="auto_descriptions_")
            auto_names.append("V2")
            auto_scores_list.append(load_auto(fallback_path))
            print(f"Auto:  {fallback_path}", file=sys.stderr)
        else:
            if vllm_path is not None:
                auto_names.append("VLLM")
                auto_scores_list.append(load_auto(vllm_path))
                print(f"Auto (VLLM): {vllm_path}", file=sys.stderr)
            if api_path is not None:
                auto_names.append("API")
                auto_scores_list.append(load_auto(api_path))
                print(f"Auto (API):  {api_path}", file=sys.stderr)

    # Use first human scores to determine cluster list
    first_human = human_scores_list[0] if human_scores_list else {}

    # Collect clusters, excluding specified ones
    clusters = sorted(c for c in first_human if c not in args.exclude)

    # Column headers
    all_attr_names = human_names + auto_names
    all_contrib_names = human_names + auto_names
    n_attr = len(all_attr_names)
    n_contrib = len(all_contrib_names)
    n_cols = 1 + n_attr + n_contrib  # cluster + attr cols + contrib cols

    # LaTeX output
    lines: list[str] = []
    col_spec = "l" + "r" * n_attr + "|" + "r" * n_contrib
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row 1: category spans
    lines.append(
        r" & \multicolumn{"
        + str(n_attr)
        + r"}{c}{Attr} & \multicolumn{"
        + str(n_contrib)
        + r"}{c}{Contrib} \\"
    )
    lines.append(r"\cmidrule(lr){2-" + str(1 + n_attr) + "}")
    lines.append(r"\cmidrule(lr){" + str(2 + n_attr) + "-" + str(n_cols) + "}")

    # Header row 2: column names
    header = "Cluster"
    for name in all_attr_names:
        header += f" & {name}"
    for name in all_contrib_names:
        header += f" & {name}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Data rows
    attr_means: dict[str, list[float]] = {name: [] for name in all_attr_names}
    contrib_means: dict[str, list[float]] = {name: [] for name in all_contrib_names}

    for cluster in clusters:
        row = cluster.replace("_", r"\_")

        # Collect attr scores for this cluster
        h_attrs = [hs.get(cluster, {}).get("attr") for hs in human_scores_list]
        a_attrs = [auto_scores.get(cluster, {}).get("attr") for auto_scores in auto_scores_list]
        all_attr = h_attrs + a_attrs
        attr_best = best_idx(all_attr)

        for i, (name, val) in enumerate(zip(human_names, h_attrs)):
            row += f" & {fmt(val, bold=(attr_best == i))}"
            if val is not None:
                attr_means[name].append(val)
        for i, (name, a_attr) in enumerate(zip(auto_names, a_attrs)):
            row += f" & {fmt(a_attr, bold=(attr_best == len(human_names) + i))}"
            if a_attr is not None:
                attr_means[name].append(a_attr)

        # Collect contrib scores for this cluster
        h_contribs = [hs.get(cluster, {}).get("contrib") for hs in human_scores_list]
        a_contribs = [
            auto_scores.get(cluster, {}).get("contrib") for auto_scores in auto_scores_list
        ]
        all_contrib = h_contribs + a_contribs
        contrib_best = best_idx(all_contrib)

        for i, (name, val) in enumerate(zip(human_names, h_contribs)):
            row += f" & {fmt(val, bold=(contrib_best == i))}"
            if val is not None:
                contrib_means[name].append(val)
        for i, (name, a_contrib) in enumerate(zip(auto_names, a_contribs)):
            row += f" & {fmt(a_contrib, bold=(contrib_best == len(human_names) + i))}"
            if a_contrib is not None:
                contrib_means[name].append(a_contrib)

        row += r" \\"
        lines.append(row)

    # Mean row
    lines.append(r"\midrule")
    mean_row = r"\textbf{Mean}"

    attr_mean_vals = [
        float(np.mean(attr_means[n])) if attr_means[n] else None for n in all_attr_names
    ]
    contrib_mean_vals = [
        float(np.mean(contrib_means[n])) if contrib_means[n] else None for n in all_contrib_names
    ]
    attr_mean_best = best_idx(attr_mean_vals)
    contrib_mean_best = best_idx(contrib_mean_vals)

    for i, v in enumerate(attr_mean_vals):
        mean_row += f" & {fmt(v, bold=(i == attr_mean_best))}"
    for i, v in enumerate(contrib_mean_vals):
        mean_row += f" & {fmt(v, bold=(i == contrib_mean_best))}"
    mean_row += r" \\"
    lines.append(mean_row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    print("\n".join(lines))


if __name__ == "__main__":
    main()

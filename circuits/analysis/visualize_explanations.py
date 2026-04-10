"""
Visualize neuron explanations from JSON output.

HTML outputs default to outputs/ (gitignored).

Usage:
    uv run python -m circuits.analysis.visualize_explanations explanations.json
    uv run python -m circuits.analysis.visualize_explanations explanations.json --html custom.html
    uv run python -m circuits.analysis.visualize_explanations explanations.json --top 10
    uv run python -m circuits.analysis.visualize_explanations explanations.json --no-html
"""

import argparse
import json
import math
from collections import defaultdict
from html import escape as html_escape
from pathlib import Path


def parse_cluster_id(cluster_str: str) -> tuple[int, int, int, str]:
    """Parse cluster/neuron ID string into components.

    Handles formats:
    - Cluster format: 'C{cluster}' -> (cluster, 0, 0, "+")
    - Legacy cluster format: 'C{cluster}_{polarity}' -> (cluster, 0, 0, polarity)
    - Old neuron format: 'L{layer}_T{token}_N{neuron}_{polarity}'

    Returns:
        Tuple of (layer/cluster, token, neuron, polarity).
    """
    parts = cluster_str.split("_")

    # Cluster format: C{cluster} or C{cluster}_{polarity}
    if parts[0].startswith("C"):
        cluster = int(parts[0][1:])
        polarity = parts[1] if len(parts) > 1 else "+"
        return cluster, 0, 0, polarity

    # Named cluster format (manual clusters): arbitrary string without L/T/N prefixes
    if not parts[0].startswith("L"):
        return 0, 0, 0, "+"

    # Old neuron format: L{layer}_T{token}_N{neuron}_{polarity}
    layer = int(parts[0][1:])  # Remove 'L'
    try:
        token = int(parts[1][1:])  # Remove 'T'
    except Exception:
        token = parts[1] if len(parts) > 1 else 0
    try:
        neuron = int(parts[2][1:])  # Remove 'N'
    except Exception:
        neuron = 0
    try:
        polarity = parts[3]
    except Exception:
        polarity = "+"
    return layer, token, neuron, polarity


def print_summary(data: dict) -> None:
    """Print summary statistics."""
    metadata = data.get("metadata", {})

    print("=" * 70)
    print("EXPLANATION SUMMARY")
    print("=" * 70)
    print(f"Circuit: {metadata.get('circuit_pickle', 'N/A')}")
    print(f"Timestamp: {metadata.get('timestamp', 'N/A')}")
    print(f"Scored: {metadata.get('scored', False)}")
    print(f"Explanations per neuron: {metadata.get('num_expl_samples', 'N/A')}")
    print()
    n_attr = len(data.get("attr", {}))
    n_contrib = len(data.get("contrib", {}))
    print(f"Total neurons: {max(n_attr, n_contrib)}")
    print(f"Total attr explanations: {metadata.get('total_attr_explanations', 'N/A')}")
    print(f"Total contrib explanations: {metadata.get('total_contrib_explanations', 'N/A')}")
    print()

    # Count by sign
    attr_pos = sum(len(v.get("pos", [])) for v in data.get("attr", {}).values())
    attr_neg = sum(len(v.get("neg", [])) for v in data.get("attr", {}).values())
    contrib_pos = sum(len(v.get("pos", [])) for v in data.get("contrib", {}).values())
    contrib_neg = sum(len(v.get("neg", [])) for v in data.get("contrib", {}).values())

    print("Breakdown by type:")
    print(f"  attr_pos:    {attr_pos:,}")
    print(f"  attr_neg:    {attr_neg:,}")
    print(f"  contrib_pos: {contrib_pos:,}")
    print(f"  contrib_neg: {contrib_neg:,}")
    print()


def get_top_explanations(data: dict, n: int = 10) -> list[dict]:
    """Get top N explanations by score across all types."""
    all_explanations = []

    for category in ["attr", "contrib"]:
        for neuron_id, sign_data in data.get(category, {}).items():
            layer, token, neuron, polarity = parse_cluster_id(neuron_id)
            for sign in ["pos", "neg", "combined"]:
                for expl in sign_data.get(sign, []):
                    if expl.get("score") is not None:
                        all_explanations.append(
                            {
                                "neuron_id": neuron_id,
                                "layer": layer,
                                "token": token,
                                "neuron": neuron,
                                "polarity": polarity,
                                "category": category,
                                "sign": sign,
                                "explanation": expl["explanation"],
                                "score": expl["score"],
                                "rsquared": expl.get("rsquared"),
                            }
                        )

    # Sort by score descending
    all_explanations.sort(key=lambda x: x["score"], reverse=True)
    return all_explanations[:n]


def print_top_explanations(data: dict, n: int = 10) -> None:
    """Print top N explanations."""
    top = get_top_explanations(data, n)

    print("=" * 70)
    print(f"TOP {n} EXPLANATIONS BY SCORE")
    print("=" * 70)

    for i, expl in enumerate(top, 1):
        print(f"\n{i}. {expl['neuron_id']} " f"[{expl['category']}_{expl['sign']}]")
        r2_str = f"  R²: {expl['rsquared']:.4f}" if expl.get("rsquared") is not None else ""
        print(f"   Score: {expl['score']:.4f}{r2_str}")
        print(f"   {expl['explanation'][:100]}{'...' if len(expl['explanation']) > 100 else ''}")


def print_neuron_details(data: dict, neuron_id: str) -> None:
    """Print all explanations for a specific neuron."""
    print(f"\n{'=' * 70}")
    print(f"NEURON: {neuron_id}")
    print("=" * 70)

    for category in ["attr", "contrib"]:
        if neuron_id not in data.get(category, {}):
            continue
        sign_data = data[category][neuron_id]

        for sign in ["pos", "neg", "combined"]:
            explanations = sign_data.get(sign, [])
            if not explanations:
                continue

            print(f"\n{category.upper()} ({sign}):")
            print("-" * 50)

            for i, expl in enumerate(explanations, 1):
                score_str = f"{expl['score']:.4f}" if expl["score"] else "N/A"
                rsq_str = f"{expl['rsquared']:.4f}" if expl["rsquared"] else "N/A"
                print(f"  {i}. [score={score_str}, R²={rsq_str}]")
                print(f"     {expl['explanation']}")
                print()


def generate_html(data: dict, output_path: Path) -> None:
    """Generate an interactive HTML visualization with collapsible clusters."""
    # Aggregate by cluster - collect all explanations and exemplars per cluster
    clusters: dict[str, dict] = {}
    # For counting total scored (clusters × types)
    max_scores_by_type: dict[str, list[float]] = defaultdict(list)

    # Get exemplars data (may not exist in older JSON files)
    attr_exemplars = data.get("attr_exemplars", {})
    contrib_exemplars = data.get("contrib_exemplars", {})
    cluster_neurons = data.get("cluster_neurons", {})

    all_cluster_ids = set()
    if "attr" in data:
        all_cluster_ids.update(data["attr"].keys())
    if "contrib" in data:
        all_cluster_ids.update(data["contrib"].keys())

    for cluster_id in all_cluster_ids:
        layer, token, neuron, polarity = parse_cluster_id(cluster_id)

        cluster_explanations: dict[str, list] = {
            "attr_pos": [],
            "attr_neg": [],
            "attr_combined": [],
            "contrib_pos": [],
            "contrib_neg": [],
            "contrib_combined": [],
            "attr_paired": [],  # For paired scoring results
            "contrib_paired": [],
        }
        cluster_exemplars = {
            "attr_pos": [],
            "attr_neg": [],
            "contrib_pos": [],
            "contrib_neg": [],
        }

        # Collect explanations (including paired results)
        for category in ["attr", "contrib"]:
            if cluster_id not in data.get(category, {}):
                continue
            sign_data = data[category][cluster_id]
            # Standard pos/neg explanations and combined
            for sign in ["pos", "neg", "combined"]:
                type_key = f"{category}_{sign}"
                for expl in sign_data.get(sign, []):
                    cluster_explanations[type_key].append(expl)
            # Paired explanations (if present)
            for pair in sign_data.get("paired", []):
                cluster_explanations[f"{category}_paired"].append(pair)

        # Collect exemplars
        if cluster_id in attr_exemplars:
            cluster_exemplars["attr_pos"] = attr_exemplars[cluster_id].get("pos", [])
            cluster_exemplars["attr_neg"] = attr_exemplars[cluster_id].get("neg", [])
        if cluster_id in contrib_exemplars:
            cluster_exemplars["contrib_pos"] = contrib_exemplars[cluster_id].get("pos", [])
            cluster_exemplars["contrib_neg"] = contrib_exemplars[cluster_id].get("neg", [])

        # Compute scores (handle both standard and paired formats)
        type_max_scores: dict[str, float] = {}
        for type_key, expls in cluster_explanations.items():
            if not expls:
                continue
            # Paired results have different structure
            if type_key.endswith("_paired"):
                type_scores = [e["score"] for e in expls if e.get("score") is not None]
            else:
                type_scores = [e["score"] for e in expls if e.get("score") is not None]
            if type_scores:
                max_type_score = max(type_scores)
                type_max_scores[type_key] = max_type_score
                max_scores_by_type[type_key].append(max_type_score)

        # Compute contrib last-token best scores
        for sign in ["pos", "neg"]:
            lt_key = f"contrib_{sign}"
            lt_scores = [
                e["score_last_token"]
                for e in cluster_explanations.get(lt_key, [])
                if e.get("score_last_token") is not None
            ]
            if lt_scores:
                type_max_scores[f"contrib_lt_{sign}"] = max(lt_scores)
                max_scores_by_type[f"contrib_lt_{sign}"].append(max(lt_scores))

        # Compute best score per category (max of pos vs neg vs combined) and track winner
        category_best: dict[str, tuple[float, str]] = {}  # category -> (best_score, winner)
        for category in ["attr", "contrib"]:
            candidates: list[tuple[float, str]] = []
            for sign in ["pos", "neg", "combined"]:
                s = type_max_scores.get(f"{category}_{sign}")
                if s is not None:
                    candidates.append((s, sign))
            if candidates:
                category_best[category] = max(candidates, key=lambda x: x[0])

        # Average and max of category bests (not all 4 types)
        best_scores = [v[0] for v in category_best.values()]
        avg_score = sum(best_scores) / len(best_scores) if best_scores else 0
        max_score = max(best_scores) if best_scores else 0

        # Get neurons in this cluster
        neurons_in_cluster = cluster_neurons.get(cluster_id, [])

        clusters[cluster_id] = {
            "cluster_id": cluster_id,
            "layer": layer,
            "token": token,
            "neuron": neuron,
            "polarity": polarity,
            "explanations": cluster_explanations,
            "exemplars": cluster_exemplars,
            "neurons": neurons_in_cluster,
            "avg_score": avg_score,
            "max_score": max_score,
            "num_explanations": sum(len(v) for v in cluster_explanations.values()),
            "num_exemplars": sum(len(v) for v in cluster_exemplars.values()),
            "num_neurons": len(neurons_in_cluster),
            "type_max_scores": type_max_scores,
            "category_best": category_best,  # {category: (score, "pos"|"neg")}
        }

    # Sort clusters by average score
    sorted_clusters = sorted(clusters.values(), key=lambda x: x["avg_score"], reverse=True)

    # Total scored count (clusters × types with scores)
    total_scored = sum(len(s) for s in max_scores_by_type.values())

    metadata = data.get("metadata", {})

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Neuron Explanations - Capitals Circuit</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; padding: 20px; background: #f5f5f5;
        }}
        .header {{
            background: white; padding: 20px; border-radius: 8px;
            margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .header h1 {{ margin: 0 0 10px 0; }}
        .stats {{ display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 20px; }}
        .stat {{ background: #f0f0f0; padding: 10px 15px; border-radius: 4px; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .stat-label {{ font-size: 12px; color: #666; }}
        .charts {{
            display: flex; gap: 30px; flex-wrap: wrap; margin-top: 20px;
            padding-top: 20px; border-top: 1px solid #eee;
        }}
        .chart-section {{ flex: 1; min-width: 400px; }}
        .chart-section h3 {{ margin: 0 0 15px 0; font-size: 14px; color: #666; }}
        .chart-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }}
        .chart-item {{ background: #fafafa; padding: 10px; border-radius: 6px; }}
        .chart-item svg {{ display: block; }}
        .chart-item svg circle:hover {{ r: 4; opacity: 1; cursor: pointer; }}
        .chart-label {{ font-size: 12px; margin-bottom: 8px; color: #333; }}
        .chart-label .type-tag {{ margin-right: 8px; }}
        .filters {{
            background: white; padding: 15px; border-radius: 8px;
            margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            display: flex; gap: 15px; flex-wrap: wrap; align-items: center;
        }}
        .filters input, .filters select {{
            padding: 8px 12px; border: 1px solid #ddd; border-radius: 4px;
        }}
        .filters input[type="text"] {{ width: 300px; }}
        .table-container {{
            background: white; border-radius: 8px; overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{
            background: #f8f8f8; padding: 12px; text-align: left;
            border-bottom: 2px solid #ddd; cursor: pointer;
            position: sticky; top: 0; z-index: 10;
        }}
        th:hover {{ background: #eee; }}
        td {{ padding: 12px; border-bottom: 1px solid #eee; vertical-align: top; }}
        tr:hover {{ background: #f8f8f8; }}
        .type-tag {{
            display: inline-block; padding: 2px 8px; border-radius: 4px;
            font-size: 11px; font-weight: 500; margin-right: 5px;
        }}
        .attr_pos {{ background: #e3f2fd; color: #1565c0; }}
        .attr_neg {{ background: #fce4ec; color: #c2185b; }}
        .contrib_pos {{ background: #e8f5e9; color: #2e7d32; }}
        .contrib_neg {{ background: #fff3e0; color: #ef6c00; }}
        .attr_combined {{ background: #ede7f6; color: #4527a0; }}
        .contrib_combined {{ background: #e0f2f1; color: #00695c; }}
        .contrib_lt_pos {{ background: #f3e5f5; color: #7b1fa2; }}
        .contrib_lt_neg {{ background: #fce4ec; color: #880e4f; }}
        .score {{ font-family: monospace; }}
        .score-high {{ color: #2e7d32; font-weight: bold; }}
        .score-mid {{ color: #f57c00; }}
        .score-low {{ color: #c62828; }}
        .cluster-id {{ font-family: monospace; font-size: 13px; white-space: nowrap; }}
        .hidden {{ display: none; }}
        .expl-section {{ margin-bottom: 10px; }}
        .expl-section:last-child {{ margin-bottom: 0; }}
        .expl-header {{ font-weight: 600; font-size: 12px; margin-bottom: 5px; }}
        .expl-item {{
            font-size: 13px; padding: 5px 8px; background: #f8f8f8;
            border-radius: 4px; margin-bottom: 4px; line-height: 1.4;
        }}
        .expl-item:last-child {{ margin-bottom: 0; }}
        .expl-score {{ font-family: monospace; font-size: 11px; color: #666; }}
        .expl-text {{ }}
        .expl-text b {{ background: #fff3cd; padding: 0 2px; border-radius: 2px; }}
        /* Collapsible styles */
        .cluster-row {{ cursor: pointer; }}
        .cluster-row:hover {{ background: #f0f0f0; }}
        .expand-icon {{ font-size: 12px; color: #666; margin-right: 8px; }}
        .collapsed-content {{ }}
        .expanded-content {{ display: none; }}
        .cluster-row.expanded .collapsed-content {{ display: none; }}
        .cluster-row.expanded .expanded-content {{ display: block; }}
        /* Nested collapsible exemplars */
        .exemplars-container {{ margin-top: 15px; padding-top: 10px; border-top: 1px dashed #ddd; }}
        .exemplars-toggle {{
            cursor: pointer; font-weight: 600; font-size: 12px; color: #666;
            padding: 5px 0; user-select: none;
        }}
        .exemplars-toggle:hover {{ color: #333; }}
        .exemplars-toggle .expand-icon {{ font-size: 10px; }}
        .exemplars-content {{ display: none; margin-top: 8px; }}
        .exemplars-container.expanded .exemplars-content {{ display: block; }}
        .exemplar-section {{ margin-bottom: 12px; }}
        .exemplar-header {{ font-weight: 600; font-size: 11px; color: #888; margin-bottom: 5px; }}
        .exemplar-item {{
            font-size: 12px; padding: 4px 6px; background: #fafafa;
            border-radius: 3px; margin-bottom: 3px; line-height: 1.3;
            font-family: monospace; white-space: pre-wrap; word-break: break-all;
        }}
        .exemplar-item b {{ background: #fff3cd; padding: 0 2px; border-radius: 2px; }}
        .exemplar-list {{
            margin-top: 8px; padding-top: 8px; border-top: 1px dashed #ddd;
        }}
        .exemplar-list-header {{
            font-size: 10px; color: #888; margin-bottom: 6px; font-weight: 600;
        }}
        .exemplar-row {{
            display: flex; gap: 12px; margin-bottom: 6px; align-items: flex-start;
        }}
        .exemplar-highlighted {{
            flex: 1; font-size: 11px; padding: 4px 6px; background: #fafafa;
            border-radius: 3px; font-family: monospace; white-space: pre-wrap;
            word-break: break-all; line-height: 1.3;
        }}
        .exemplar-highlighted b {{ background: #fff3cd; padding: 0 2px; border-radius: 2px; }}
        .exemplar-tokens {{
            flex: 1; font-size: 11px; padding: 4px 6px; background: #f5f5f5;
            border-radius: 3px; font-family: monospace; white-space: pre-wrap;
            word-break: break-all; line-height: 1.3;
        }}
        .token-with-score {{
            cursor: help; padding: 0 1px; border-radius: 2px;
        }}
        .token-with-score:hover {{ outline: 1px solid #999; }}
        /* Clickable explanations with predictions */
        .expl-item-container {{ margin-bottom: 4px; }}
        .expl-item.clickable {{ cursor: pointer; }}
        .expl-item.clickable:hover {{ background: #f0f0f0; }}
        .pred-toggle {{
            font-size: 10px; color: #888; margin-left: 8px;
            padding: 2px 6px; background: #f5f5f5; border-radius: 3px;
        }}
        .predictions-content {{
            display: none; margin: 6px 0 10px 0; padding: 8px;
            background: #fafafa; border-radius: 4px; border: 1px solid #e0e0e0;
        }}
        .expl-item-container.expanded .predictions-content {{ display: block; }}
        .expl-item-container.expanded .pred-toggle {{ background: #e0e0e0; }}
        .prediction-row {{
            display: flex; gap: 12px; margin-bottom: 8px; align-items: flex-start;
            padding: 4px 0; border-bottom: 1px solid #f0f0f0;
        }}
        .prediction-col {{
            flex: 1; font-size: 11px; font-family: monospace;
            white-space: pre-wrap; word-break: break-all; line-height: 1.4;
        }}
        .prediction-col.highlighted-col {{
            flex: 1.2; background: #fafafa; padding: 4px 6px; border-radius: 3px;
        }}
        .pred-label {{
            font-weight: bold; font-size: 10px; color: #666; margin-right: 4px;
        }}
        .prediction-header {{
            border-bottom: 2px solid #ddd; margin-bottom: 8px; padding-bottom: 4px;
        }}
        .prediction-header .pred-label {{ font-size: 11px; color: #444; }}
        .neuron-list {{ font-family: monospace; font-size: 11px; color: #555; line-height: 1.6; }}
        /* Scatter plot styles */
        .scatter-container {{ margin: 10px 0; padding: 10px; background: #fafafa; border-radius: 6px; }}
        .scatter-title {{ font-size: 12px; font-weight: bold; color: #444; margin-bottom: 8px; }}
        .scatter-stats {{ font-size: 11px; color: #666; margin-top: 6px; }}
        .scatter-row {{ display: flex; gap: 20px; flex-wrap: wrap; align-items: flex-start; }}
        .summary-text {{ font-size: 13px; color: #333; }}
        .summary-text b {{ background: #fff3cd; padding: 0 2px; border-radius: 2px; }}
        /* Tooltip */
        #tooltip {{
            display: none; position: fixed; z-index: 1000;
            background: #333; color: #fff; padding: 6px 10px; border-radius: 4px;
            font-size: 12px; font-family: monospace; pointer-events: none;
            white-space: pre-wrap; word-break: break-all; box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            max-width: 500px;
        }}
        /* Aligned label rows */
        .desc-row {{ display: flex; align-items: baseline; margin-bottom: 4px; gap: 8px; }}
        .desc-row:last-child {{ margin-bottom: 0; }}
        .desc-row.best-desc {{ background: rgba(76, 175, 80, 0.1); border-radius: 3px; padding: 2px 4px; margin-left: -4px; }}
        .desc-label {{ width: 100px; flex-shrink: 0; display: flex; justify-content: flex-end; }}
        .desc-content {{ flex: 1; min-width: 0; }}
        .desc-score {{ font-family: monospace; font-size: 12px; color: #666; }}
        /* Histograms */
        .histograms-row {{ display: flex; gap: 20px; margin-bottom: 16px; }}
        .histogram-container {{ flex: 1; }}
        .histogram-title {{ font-size: 12px; font-weight: bold; color: #444; margin-bottom: 6px; }}
        /* Paired explanations */
        .pair-pos {{ color: #2e7d32; }}
        .pair-neg {{ color: #c62828; margin-left: 12px; }}
        .type-tag.attr_paired {{ background: linear-gradient(90deg, #e8f5e9 50%, #ffebee 50%); color: #333; }}
        .type-tag.contrib_paired {{ background: linear-gradient(90deg, #e3f2fd 50%, #fff3e0 50%); color: #333; }}
    </style>
</head>
<body>
    <div id="tooltip"></div>
    <div class="header">
        <h1>Cluster Explanations - Capitals Circuit</h1>
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{len(clusters):,}</div>
                <div class="stat-label">Clusters</div>
            </div>
            <div class="stat">
                <div class="stat-value">{metadata.get('total_attr_explanations', 0):,}</div>
                <div class="stat-label">Attr Explanations</div>
            </div>
            <div class="stat">
                <div class="stat-value">{metadata.get('total_contrib_explanations', 0):,}</div>
                <div class="stat-label">Contrib Explanations</div>
            </div>
            <div class="stat">
                <div class="stat-value">{total_scored:,}</div>
                <div class="stat-label">Clusters × Types</div>
            </div>
        </div>

        <div class="charts">
            <div class="chart-section">
                <h3>Best Score per Category (max of pos/neg)</h3>
                <div class="chart-grid">
                    <div class="chart-item">
                        <div class="chart-label" id="hist-label-attr_best">attr_best</div>
                        <div id="hist-attr_best"></div>
                    </div>
                    <div class="chart-item">
                        <div class="chart-label" id="hist-label-contrib_combined_best">contrib_combined_best</div>
                        <div id="hist-contrib_combined_best"></div>
                    </div>
                    <div class="chart-item">
                        <div class="chart-label" id="scatter-label-attr-vs-contrib">attr_best vs contrib_combined</div>
                        <div id="scatter-attr-vs-contrib"></div>
                    </div>
                </div>
            </div>


        </div>
    </div>

    <div class="filters">
        <input type="text" id="search" placeholder="Search explanations..." oninput="filterTable()">
        <select id="layerFilter" onchange="filterTable()">
            <option value="">All Layers</option>
        </select>
        <select id="polarityFilter" onchange="filterTable()">
            <option value="">All Polarities</option>
            <option value="+">+ (positive)</option>
            <option value="-">- (negative)</option>
        </select>
        <label><input type="checkbox" id="highScoreOnly" onchange="filterTable()"> Avg Score > 0.7</label>
        <span id="visibleCount" style="color: #666; font-size: 13px;"></span>
    </div>

    <div class="table-container">
        <table id="explanationsTable">
            <thead>
                <tr>
                    <th onclick="sortTable(0)" style="width: 100px;">Cluster</th>
                    <th onclick="sortTable(1)" style="width: 50px;">ID</th>
                    <th onclick="sortTable(2)" style="width: 70px;">Neurons</th>
                    <th onclick="sortTable(3)" style="width: 70px;">Avg Score</th>
                    <th onclick="sortTable(4)" style="width: 70px;">Max Score</th>
                    <th>Explanations & Exemplars (click to expand)</th>
                </tr>
            </thead>
            <tbody>
"""

    # Add rows - one per cluster
    def escape_html_text(text: str) -> str:
        """Escape HTML entities."""
        return html_escape(text)

    def get_top_explanation(expls: list[dict], type_key: str, is_best: bool = False) -> str:
        """Get the top-1 explanation formatted as summary with aligned columns."""
        if not expls:
            return ""
        sorted_expls = sorted(
            expls, key=lambda e: e["score"] if e["score"] is not None else -999, reverse=True
        )
        top = sorted_expls[0]
        score_str = f"{top['score']:.3f}" if top["score"] is not None else "N/A"
        text = escape_html_text(top["explanation"])
        best_class = " best-desc" if is_best else ""
        best_marker = " ★" if is_best else ""
        return (
            f'<div class="desc-row{best_class}">'
            f'<div class="desc-label"><span class="type-tag {type_key}">{type_key}{best_marker}</span></div>'
            f'<div class="desc-content"><span class="desc-score">[{score_str}]</span> {text}</div>'
            f"</div>"
        )

    def get_top_explanation_by_lt(expls: list[dict], type_key: str, is_best: bool = False) -> str:
        """Get the top-1 explanation by score_last_token, formatted as summary."""
        scored = [e for e in expls if e.get("score_last_token") is not None]
        if not scored:
            return ""
        sorted_expls = sorted(scored, key=lambda e: e["score_last_token"], reverse=True)
        top = sorted_expls[0]
        lt_score = top["score_last_token"]
        text = escape_html_text(top["explanation"])
        best_class = " best-desc" if is_best else ""
        best_marker = " ★" if is_best else ""
        return (
            f'<div class="desc-row{best_class}">'
            f'<div class="desc-label"><span class="type-tag {type_key}">{type_key}{best_marker}</span></div>'
            f'<div class="desc-content"><span class="desc-score">[{lt_score:.3f}]</span> {text}</div>'
            f"</div>"
        )

    def get_top_paired_explanation(pairs: list[dict], type_key: str) -> str:
        """Get the top-1 paired explanation formatted as summary."""
        if not pairs:
            return ""
        sorted_pairs = sorted(
            pairs, key=lambda e: e["score"] if e.get("score") is not None else -999, reverse=True
        )
        top = sorted_pairs[0]
        score_str = f"{top['score']:.3f}" if top.get("score") is not None else "N/A"
        pos_text = escape_html_text(top.get("pos_explanation", ""))
        neg_text = escape_html_text(top.get("neg_explanation", ""))
        pair_text = f'<span class="pair-pos">+: {pos_text}</span> <span class="pair-neg">−: {neg_text}</span>'
        return (
            f'<div class="desc-row">'
            f'<div class="desc-label"><span class="type-tag {type_key}">{type_key}</span></div>'
            f'<div class="desc-content"><span class="desc-score">[{score_str}]</span> {pair_text}</div>'
            f"</div>"
        )

    def count_highlighted_exemplars(predictions: list[dict] | None, exemplars: list | None) -> int:
        """Count how many predictions have matching highlighted exemplars."""
        if not predictions or not exemplars:
            return 0
        # Build lookup from tokens to exemplar's formatted text
        exemplar_tokens = set()
        for ex in exemplars:
            if isinstance(ex, dict):
                exemplar_tokens.add(tuple(ex.get("tokens", [])))
        # Count matches
        count = 0
        for pred in predictions:
            tokens_key = tuple(pred.get("tokens", []))
            if tokens_key in exemplar_tokens:
                count += 1
        return count

    def format_predictions(
        predictions: list[dict] | None, exemplars: list | None, is_neg: bool = False
    ) -> str:
        """Format simulator predictions for display with highlighted text."""
        if not predictions:
            return ""

        # Build lookup from tokens to exemplar's formatted text (with {{highlights}})
        exemplar_lookup: dict[tuple, str] = {}
        if exemplars:
            for ex in exemplars:
                if isinstance(ex, dict):
                    tokens_key = tuple(ex.get("tokens", []))
                    exemplar_lookup[tokens_key] = ex.get("text", "")

        # Sort predictions by max true activation (descending)
        predictions = sorted(
            predictions,
            key=lambda p: max([-a for a in p.get("true", [0])] if is_neg else p.get("true", [0])),
            reverse=True,
        )

        # Compute ranges for adaptive coloring
        all_true = [a for pred in predictions for a in pred.get("true", [])]
        if is_neg:
            all_true = [-a for a in all_true]
        all_pred = [a for pred in predictions for a in pred.get("predicted", [])]
        true_max = max(all_true) if all_true else 1.0
        pred_max = max(all_pred) if all_pred else 1.0

        def get_color(val: float, max_val: float, high_color: str, low_color: str) -> str:
            """Get background color based on value relative to max."""
            if max_val <= 0:
                return "transparent"
            ratio = val / max_val
            if ratio > 0.5:
                return high_color
            elif ratio > 0.1:
                return low_color
            return "transparent"

        # Header row
        items = """<div class="prediction-row prediction-header">
            <div class="prediction-col highlighted-col"><span class="pred-label">Highlighted</span></div>
            <div class="prediction-col"><span class="pred-label">True</span></div>
            <div class="prediction-col"><span class="pred-label">Predicted</span></div>
        </div>"""
        for pred in predictions:
            tokens = pred.get("tokens", [])
            true_acts = pred.get("true", [])
            pred_acts = pred.get("predicted", [])

            # For neg descriptions, negate true activations so high magnitude = highlighted
            if is_neg:
                true_acts = [-a for a in true_acts]

            # Try to find matching exemplar by tokens - show formatted text with {{highlights}}
            tokens_key = tuple(tokens)
            if tokens_key in exemplar_lookup:
                # escape_html_text converts {{token}} to <b>token</b>
                highlighted_text = escape_html_text(exemplar_lookup[tokens_key])
            else:
                highlighted_text = ""

            # Build side-by-side token display: true | predicted
            true_spans = []
            pred_spans = []
            for tok, true_act, pred_act in zip(tokens, true_acts, pred_acts):
                escaped_tok = html_escape(tok)
                # True activation coloring (green, based on true range)
                true_bg = get_color(true_act, true_max, "#c8e6c9", "#e8f5e9")
                true_spans.append(
                    f'<span class="token-with-score" title="true={true_act:.3f}" '
                    f'style="background:{true_bg}">{escaped_tok}</span>'
                )
                # Predicted activation coloring (blue, based on pred range)
                pred_bg = get_color(pred_act, pred_max, "#bbdefb", "#e3f2fd")
                pred_spans.append(
                    f'<span class="token-with-score" title="pred={pred_act:.3f}" '
                    f'style="background:{pred_bg}">{escaped_tok}</span>'
                )

            items += f"""<div class="prediction-row">
                <div class="prediction-col highlighted-col">{highlighted_text}</div>
                <div class="prediction-col">{"".join(true_spans)}</div>
                <div class="prediction-col">{"".join(pred_spans)}</div>
            </div>"""
        return items

    def format_predictions_minibatch_contrib(predictions: list[dict] | None) -> str:
        """Format minibatch contrib predictions (tokens + continuations format)."""
        if not predictions:
            return ""

        # Collect all true/pred values for adaptive coloring
        all_true = [
            abs(c.get("true", 0)) for pred in predictions for c in pred.get("continuations", [])
        ]
        all_pred = [
            abs(c.get("predicted", 0))
            for pred in predictions
            for c in pred.get("continuations", [])
        ]
        true_max = max(all_true) if all_true else 1.0
        pred_max = max(all_pred) if all_pred else 1.0

        def score_color(val: float, max_val: float, pos_color: str, neg_color: str) -> str:
            if max_val <= 0:
                return "transparent"
            ratio = abs(val) / max_val
            if ratio < 0.1:
                return "transparent"
            return pos_color if val >= 0 else neg_color

        items = ""
        for pred in predictions:
            tokens = pred.get("tokens", [])
            continuations = pred.get("continuations", [])
            prompt_text = html_escape("".join(tokens))

            cont_rows = ""
            for c in continuations:
                tok = html_escape(c.get("token", "?"))
                true_val = c.get("true", 0)
                pred_val = c.get("predicted", 0)
                true_bg = score_color(true_val, true_max, "#c8e6c9", "#ffcdd2")
                pred_bg = score_color(pred_val, pred_max, "#bbdefb", "#ffcdd2")
                cont_rows += (
                    f"<tr>"
                    f'<td style="padding:2px 6px;font-family:monospace;font-size:11px;'
                    f'font-weight:600;white-space:nowrap;">{tok}</td>'
                    f'<td style="padding:2px 6px;font-family:monospace;font-size:11px;'
                    f'text-align:right;background:{true_bg};">{true_val:.1f}</td>'
                    f'<td style="padding:2px 6px;font-family:monospace;font-size:11px;'
                    f'text-align:right;background:{pred_bg};">{pred_val:.1f}</td>'
                    f"</tr>"
                )

            cont_table = (
                f'<table style="border-collapse:collapse;width:100%;">'
                f'<tr style="border-bottom:1px solid #ddd;">'
                f'<th style="padding:2px 6px;text-align:left;font-size:10px;color:#888;">Token</th>'
                f'<th style="padding:2px 6px;text-align:right;font-size:10px;color:#888;">True</th>'
                f'<th style="padding:2px 6px;text-align:right;font-size:10px;color:#888;">Pred</th>'
                f"</tr>{cont_rows}</table>"
            )

            items += f"""<div style="display:flex;gap:12px;margin-bottom:10px;padding:8px;
                background:#fafafa;border-radius:4px;border:1px solid #e0e0e0;">
                <div style="flex:2;font-family:monospace;font-size:11px;white-space:pre-wrap;
                    word-break:break-all;line-height:1.3;overflow:auto;max-height:120px;">{prompt_text}</div>
                <div style="flex:1;">{cont_table}</div>
            </div>"""
        return items

    def format_expl_section_with_exemplars(
        expls: list[dict], exemplars: list, type_key: str
    ) -> str:
        """Format explanations with their exemplars shown below."""
        if not expls and not exemplars:
            return ""

        # Extract highlight threshold info from exemplars (if available)
        threshold_info = ""
        if exemplars and isinstance(exemplars[0], dict):
            thresh = exemplars[0].get("highlight_threshold")
            if thresh is not None:
                threshold_info = (
                    f' <span style="font-size:10px;color:#888;">[thresh={thresh:.9f}]</span>'
                )

        # Format explanations (clickable to show predictions with exemplars)
        expl_items = ""
        if expls:
            sorted_expls = sorted(
                expls, key=lambda e: e["score"] if e["score"] is not None else -999, reverse=True
            )
            for i, e in enumerate(sorted_expls):
                score_str = f"{e['score']:.3f}" if e["score"] is not None else "N/A"
                text = escape_html_text(e["explanation"])
                predictions = e.get("predictions")

                # Only show predictions for top explanation (first after sorting)
                is_minibatch = (
                    predictions
                    and isinstance(predictions[0], dict)
                    and "continuations" in predictions[0]
                )
                if predictions and i == 0:
                    if is_minibatch:
                        pred_html = format_predictions_minibatch_contrib(predictions)
                    else:
                        pred_html = format_predictions(
                            predictions, exemplars, is_neg="neg" in type_key
                        )
                    pred_count = len(predictions)
                    # Count how many have highlighted text available
                    highlighted_count = (
                        pred_count
                        if is_minibatch
                        else count_highlighted_exemplars(predictions, exemplars)
                    )
                    count_str = (
                        f"{highlighted_count}/{pred_count}"
                        if highlighted_count < pred_count
                        else str(pred_count)
                    )
                    expl_items += f"""<div class="expl-item-container">
                        <div class="expl-item clickable" onclick="togglePredictions(event, this)">
                            <span class="expl-score">[{score_str}]</span>
                            <span class="expl-text">{text}</span>
                            <span class="pred-toggle">▶ {count_str} exemplars</span>
                        </div>
                        <div class="predictions-content">{pred_html}</div>
                    </div>"""
                else:
                    expl_items += f'<div class="expl-item"><span class="expl-score">[{score_str}]</span> <span class="expl-text">{text}</span></div>'

        return f'<div class="expl-section"><div class="expl-header"><span class="type-tag {type_key}">{type_key}</span>{threshold_info}</div>{expl_items}</div>'

    def format_expl_section_by_lt(
        expls: list[dict], type_key: str, exemplars: list | None = None
    ) -> str:
        """Format explanations sorted by score_last_token, with exemplars for top-1."""
        if not expls:
            return ""
        # Sort: scored first (by score_last_token desc), then unscored
        scored = [e for e in expls if e.get("score_last_token") is not None]
        unscored = [e for e in expls if e.get("score_last_token") is None]
        sorted_expls = sorted(scored, key=lambda e: e["score_last_token"], reverse=True) + unscored
        if not sorted_expls:
            return ""
        expl_items = ""
        for i, e in enumerate(sorted_expls):
            lt_score = e.get("score_last_token")
            score_str = f"{lt_score:.3f}" if lt_score is not None else "N/A"
            text = escape_html_text(e["explanation"])

            # Show predictions with exemplars for top explanation
            predictions = e.get("predictions")
            if predictions and i == 0 and exemplars:
                pred_html = format_predictions(predictions, exemplars, is_neg="neg" in type_key)
                pred_count = len(predictions)
                highlighted_count = count_highlighted_exemplars(predictions, exemplars)
                count_str = (
                    f"{highlighted_count}/{pred_count}"
                    if highlighted_count < pred_count
                    else str(pred_count)
                )
                expl_items += f"""<div class="expl-item-container">
                    <div class="expl-item clickable" onclick="togglePredictions(event, this)">
                        <span class="expl-score">[{score_str}]</span>
                        <span class="expl-text">{text}</span>
                        <span class="pred-toggle">▶ {count_str} exemplars</span>
                    </div>
                    <div class="predictions-content">{pred_html}</div>
                </div>"""
            else:
                expl_items += f'<div class="expl-item"><span class="expl-score">[{score_str}]</span> <span class="expl-text">{text}</span></div>'
        return f'<div class="expl-section"><div class="expl-header"><span class="type-tag {type_key}">{type_key}</span></div>{expl_items}</div>'

    def format_paired_predictions(predictions: list[dict] | None, exemplars: list | None) -> str:
        """Format paired predictions showing pos_pred, neg_pred, and combined."""
        if not predictions:
            return ""

        # Build lookup from tokens to exemplar's formatted text
        exemplar_lookup: dict[tuple, str] = {}
        if exemplars:
            for ex in exemplars:
                if isinstance(ex, dict):
                    tokens_key = tuple(ex.get("tokens", []))
                    exemplar_lookup[tokens_key] = ex.get("text", "")

        # Sort predictions by max absolute true activation (descending)
        predictions = sorted(
            predictions,
            key=lambda p: max(abs(a) for a in p.get("true", [0])),
            reverse=True,
        )

        # Compute ranges for adaptive coloring
        all_true = [a for pred in predictions for a in pred.get("true", [])]
        all_pos = [a for pred in predictions for a in pred.get("pos_pred", [])]
        all_neg = [a for pred in predictions for a in pred.get("neg_pred", [])]
        # For true, track both positive and negative max magnitudes
        true_pos_max = max([a for a in all_true if a > 0], default=0.0)
        true_neg_max = abs(min([a for a in all_true if a < 0], default=0.0))
        pos_max = max(all_pos) if all_pos else 1.0
        neg_max = max(all_neg) if all_neg else 1.0

        def get_color(val: float, max_val: float, high_color: str, low_color: str) -> str:
            """Get background color based on value relative to max."""
            if max_val <= 0:
                return "transparent"
            ratio = val / max_val
            if ratio > 0.5:
                return high_color
            elif ratio > 0.1:
                return low_color
            return "transparent"

        # Header row with 4 columns
        items = """<div class="prediction-row prediction-header">
            <div class="prediction-col highlighted-col"><span class="pred-label">Highlighted</span></div>
            <div class="prediction-col"><span class="pred-label">True</span></div>
            <div class="prediction-col"><span class="pred-label">Pos Pred</span></div>
            <div class="prediction-col"><span class="pred-label">Neg Pred</span></div>
        </div>"""
        for pred in predictions:
            tokens = pred.get("tokens", [])
            true_acts = pred.get("true", [])
            pos_preds = pred.get("pos_pred", [])
            neg_preds = pred.get("neg_pred", [])

            # Get highlighted text from exemplars
            tokens_key = tuple(tokens)
            if tokens_key in exemplar_lookup:
                highlighted_text = escape_html_text(exemplar_lookup[tokens_key])
            else:
                highlighted_text = ""

            # Build token spans for each column
            true_spans = []
            pos_spans = []
            neg_spans = []
            for tok, true_act, pos_pred, neg_pred in zip(tokens, true_acts, pos_preds, neg_preds):
                escaped_tok = html_escape(tok)
                # True activation coloring (green for positive, red for negative)
                if true_act >= 0:
                    true_bg = get_color(true_act, true_pos_max, "#c8e6c9", "#e8f5e9")
                else:
                    true_bg = get_color(abs(true_act), true_neg_max, "#ffcdd2", "#ffebee")
                true_spans.append(
                    f'<span class="token-with-score" title="true={true_act:.3f}" '
                    f'style="background:{true_bg}">{escaped_tok}</span>'
                )
                # Pos pred coloring (green, based on pos range)
                pos_bg = get_color(pos_pred, pos_max, "#c8e6c9", "#e8f5e9")
                pos_spans.append(
                    f'<span class="token-with-score" title="pos={pos_pred:.3f}" '
                    f'style="background:{pos_bg}">{escaped_tok}</span>'
                )
                # Neg pred coloring (red, based on neg range)
                neg_bg = get_color(neg_pred, neg_max, "#ffcdd2", "#ffebee")
                neg_spans.append(
                    f'<span class="token-with-score" title="neg={neg_pred:.3f}" '
                    f'style="background:{neg_bg}">{escaped_tok}</span>'
                )

            items += f"""<div class="prediction-row">
                <div class="prediction-col highlighted-col">{highlighted_text}</div>
                <div class="prediction-col">{"".join(true_spans)}</div>
                <div class="prediction-col">{"".join(pos_spans)}</div>
                <div class="prediction-col">{"".join(neg_spans)}</div>
            </div>"""
        return items

    def format_paired_section(pairs: list[dict], exemplars: list, type_key: str) -> str:
        """Format paired explanation results."""
        if not pairs:
            return ""

        # Extract highlight threshold info from exemplars (if available)
        threshold_info = ""
        if exemplars and isinstance(exemplars[0], dict):
            thresh = exemplars[0].get("highlight_threshold")
            quantile = exemplars[0].get("quantile_used")
            if thresh is not None and quantile is not None:
                quantile_pct = f"{quantile * 100:.9f}%"
                threshold_info = f' <span style="font-size:10px;color:#888;">[thresh={thresh:.9f}, q={quantile_pct}]</span>'

        expl_items = ""
        sorted_pairs = sorted(
            pairs, key=lambda e: e["score"] if e.get("score") is not None else -999, reverse=True
        )
        for i, pair in enumerate(sorted_pairs):
            score_str = f"{pair['score']:.3f}" if pair.get("score") is not None else "N/A"
            pos_text = escape_html_text(pair.get("pos_explanation", ""))
            neg_text = escape_html_text(pair.get("neg_explanation", ""))
            predictions = pair.get("predictions")

            # Combined display: "POS: ... | NEG: ..."
            pair_text = f'<span class="pair-pos">+: {pos_text}</span> <span class="pair-neg">−: {neg_text}</span>'

            # Only show predictions for top pair
            if predictions and i == 0:
                pred_html = format_paired_predictions(predictions, exemplars)
                pred_count = len(predictions)
                # Count how many have highlighted text available
                highlighted_count = count_highlighted_exemplars(predictions, exemplars)
                count_str = (
                    f"{highlighted_count}/{pred_count}"
                    if highlighted_count < pred_count
                    else str(pred_count)
                )
                expl_items += f"""<div class="expl-item-container">
                    <div class="expl-item clickable" onclick="togglePredictions(event, this)">
                        <span class="expl-score">[{score_str}]</span>
                        <span class="expl-text">{pair_text}</span>
                        <span class="pred-toggle">▶ {count_str} exemplars</span>
                    </div>
                    <div class="predictions-content">{pred_html}</div>
                </div>"""
            else:
                expl_items += f'<div class="expl-item"><span class="expl-score">[{score_str}]</span> <span class="expl-text">{pair_text}</span></div>'

        return f'<div class="expl-section"><div class="expl-header"><span class="type-tag {type_key}">{type_key}</span>{threshold_info}</div>{expl_items}</div>'

    def get_true_acts_from_expls(expls: list[dict]) -> list[float]:
        """Extract all true activations from top explanation's predictions."""
        if not expls:
            return []
        sorted_expls = sorted(
            expls, key=lambda e: e["score"] if e["score"] is not None else -999, reverse=True
        )
        top = sorted_expls[0]
        predictions = top.get("predictions", [])
        if not predictions:
            return []
        all_acts = []
        for pred in predictions:
            all_acts.extend(pred.get("true", []))
        return all_acts

    def make_histogram_svg(
        pos_acts: list[float],
        neg_acts: list[float],
        title: str,
        width: int = 300,
        height: int = 120,
        show_legend: bool = False,
    ) -> str:
        """Create SVG histogram with pos (green) and neg (red) overlaid."""
        if not pos_acts and not neg_acts:
            return f'<div class="histogram-container"><div class="histogram-title">{title}</div><em>No data</em></div>'

        # Combine to find range
        all_acts = pos_acts + neg_acts
        min_val = min(all_acts) if all_acts else 0
        max_val = max(all_acts) if all_acts else 1
        if max_val == min_val:
            max_val = min_val + 1

        # Create bins
        num_bins = 30
        bin_width = (max_val - min_val) / num_bins

        def bin_counts(acts: list[float]) -> list[int]:
            counts = [0] * num_bins
            for a in acts:
                idx = min(int((a - min_val) / bin_width), num_bins - 1)
                counts[idx] += 1
            return counts

        pos_counts = bin_counts(pos_acts) if pos_acts else [0] * num_bins
        neg_counts = bin_counts(neg_acts) if neg_acts else [0] * num_bins
        max_count = max(max(pos_counts), max(neg_counts), 1)

        # Log scale helper (log(count + 1) to handle zeros)
        def log_scale(count: int) -> float:
            return math.log(count + 1)

        max_log = log_scale(max_count)

        # SVG dimensions
        margin = 25
        plot_width = width - 2 * margin
        plot_height = height - 2 * margin
        bar_width = plot_width / num_bins

        # Build bars (log y-axis)
        bars = ""
        for i in range(num_bins):
            x = margin + i * bar_width
            # Neg bars (red, behind)
            if neg_counts[i] > 0:
                h = (log_scale(neg_counts[i]) / max_log) * plot_height
                y = margin + plot_height - h
                bars += f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{h:.1f}" fill="rgba(239,83,80,0.5)" />'
            # Pos bars (green, in front)
            if pos_counts[i] > 0:
                h = (log_scale(pos_counts[i]) / max_log) * plot_height
                y = margin + plot_height - h
                bars += f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{h:.1f}" fill="rgba(102,187,106,0.5)" />'

        # Axes
        axes = f"""<line x1="{margin}" y1="{margin + plot_height}" x2="{margin + plot_width}" y2="{margin + plot_height}" stroke="#333" stroke-width="1"/>
            <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{margin + plot_height}" stroke="#333" stroke-width="1"/>"""

        # Labels
        labels = f"""<text x="{margin}" y="{height - 5}" font-size="9" fill="#666">{min_val:.2f}</text>
            <text x="{width - margin}" y="{height - 5}" font-size="9" fill="#666" text-anchor="end">{max_val:.2f}</text>"""

        # Legend
        legend = ""
        if show_legend:
            legend = f"""<rect x="{width - 80}" y="5" width="10" height="10" fill="rgba(102,187,106,0.5)"/>
                <text x="{width - 65}" y="14" font-size="9" fill="#666">pos</text>
                <rect x="{width - 40}" y="5" width="10" height="10" fill="rgba(239,83,80,0.5)"/>
                <text x="{width - 25}" y="14" font-size="9" fill="#666">neg</text>"""

        svg = f"""<svg width="{width}" height="{height}" style="background:#fafafa; border-radius:4px;">
            {bars}{axes}{labels}{legend}
        </svg>"""

        return f'<div class="histogram-container"><div class="histogram-title">{title}</div>{svg}</div>'

    def make_scatter_svg(
        true_acts: list[float],
        pred_acts: list[float],
        title: str = "True vs Predicted",
        width: int = 200,
        height: int = 200,
        color: str = "#1976d2",
        threshold: float | None = None,
        token_labels: list[str] | None = None,
    ) -> str:
        """Create SVG scatter plot of true vs predicted activations with R² and correlation."""
        if not true_acts or not pred_acts or len(true_acts) != len(pred_acts):
            return ""

        n = len(true_acts)

        # Compute stats
        mean_true = sum(true_acts) / n
        mean_pred = sum(pred_acts) / n

        # Correlation
        cov = sum((t - mean_true) * (p - mean_pred) for t, p in zip(true_acts, pred_acts)) / n
        var_true = sum((t - mean_true) ** 2 for t in true_acts) / n
        var_pred = sum((p - mean_pred) ** 2 for p in pred_acts) / n
        if var_true > 0 and var_pred > 0:
            corr = cov / (var_true**0.5 * var_pred**0.5)
        else:
            corr = 0.0

        # R² (coefficient of determination)
        ss_res = sum((t - p) ** 2 for t, p in zip(true_acts, pred_acts))
        ss_tot = sum((t - mean_true) ** 2 for t in true_acts)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Find ranges separately for x (true) and y (pred)
        x_min, x_max = min(true_acts), max(true_acts)
        y_min, y_max = min(pred_acts), max(pred_acts)
        if x_max == x_min:
            x_max = x_min + 1
        if y_max == y_min:
            y_max = y_min + 1
        # Add padding
        x_padding = (x_max - x_min) * 0.05
        y_padding = (y_max - y_min) * 0.05
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding
        x_range = x_max - x_min
        y_range = y_max - y_min

        # SVG dimensions
        margin = 30
        plot_width = width - 2 * margin
        plot_height = height - 2 * margin

        # Build points
        points = []
        for i, (t, p) in enumerate(zip(true_acts, pred_acts)):
            x = margin + ((t - x_min) / x_range) * plot_width
            y = margin + plot_height - ((p - y_min) / y_range) * plot_height  # Flip y
            tok_str = ""
            if token_labels and i < len(token_labels):
                tok_str = f"\ntok: {token_labels[i]}"
            tip = f"true: {t:.4f}\npred: {p:.4f}{tok_str}"
            points.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}" fill-opacity="0.5" '
                f'class="tip" data-tip="{html_escape(tip)}"/>'
            )

        # Axes
        axes = f"""<line x1="{margin}" y1="{margin + plot_height}" x2="{margin + plot_width}" y2="{margin + plot_height}" stroke="#333" stroke-width="1"/>
            <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{margin + plot_height}" stroke="#333" stroke-width="1"/>"""

        # Threshold line + highlighted region
        threshold_line = ""
        if threshold is not None and x_min <= threshold <= x_max:
            thresh_x = margin + ((threshold - x_min) / x_range) * plot_width
            # Shade the included region: x >= threshold (pos) or x <= threshold (neg)
            if threshold >= 0:
                # Positive threshold: highlight right side (x >= threshold)
                rect_x = thresh_x
                rect_w = margin + plot_width - thresh_x
            else:
                # Negative threshold: highlight left side (x <= threshold)
                rect_x = margin
                rect_w = thresh_x - margin
            threshold_line = f"""<rect x="{rect_x:.1f}" y="{margin}" width="{rect_w:.1f}" height="{plot_height}" fill="#ff5722" fill-opacity="0.08"/>"""
            threshold_line += f"""<line x1="{thresh_x:.1f}" y1="{margin}" x2="{thresh_x:.1f}" y2="{margin + plot_height}" stroke="#ff5722" stroke-width="1" stroke-dasharray="4,2" class="tip" data-tip="highlight threshold: {threshold:.4f}"/>"""

        # Labels
        labels = f"""<text x="{margin + plot_width / 2}" y="{height - 2}" font-size="10" fill="#666" text-anchor="middle">True</text>
            <text x="10" y="{margin + plot_height / 2}" font-size="10" fill="#666" text-anchor="middle" transform="rotate(-90, 10, {margin + plot_height / 2})">Pred</text>
            <text x="{margin}" y="{height - 12}" font-size="8" fill="#999">{x_min:.2f}</text>
            <text x="{margin + plot_width}" y="{height - 12}" font-size="8" fill="#999" text-anchor="end">{x_max:.2f}</text>
            <text x="{margin - 2}" y="{margin + plot_height}" font-size="8" fill="#999" text-anchor="end">{y_min:.2f}</text>
            <text x="{margin - 2}" y="{margin + 4}" font-size="8" fill="#999" text-anchor="end">{y_max:.2f}</text>"""

        svg = f"""<svg width="{width}" height="{height}" style="background:white; border-radius:4px; border: 1px solid #e0e0e0;">
            {axes}{threshold_line}{"".join(points)}{labels}
        </svg>"""

        stats_html = f'<div class="scatter-stats">n={n} | r={corr:.3f} | R²={r_squared:.3f}</div>'

        return f'<div class="scatter-container"><div class="scatter-title">{title}</div>{svg}{stats_html}</div>'

    def get_true_pred_from_expls(
        expls: list[dict],
    ) -> tuple[list[float], list[float], list[str]]:
        """Extract all (true, predicted, token) triples from top explanation's predictions."""
        if not expls:
            return [], [], []
        sorted_expls = sorted(
            expls, key=lambda e: e["score"] if e["score"] is not None else -999, reverse=True
        )
        top = sorted_expls[0]
        predictions = top.get("predictions", [])
        if not predictions:
            return [], [], []
        true_acts: list[float] = []
        pred_acts: list[float] = []
        token_labels: list[str] = []
        for pred_idx, pred in enumerate(predictions):
            # Minibatch contrib format: tokens + continuations
            if "continuations" in pred:
                tokens = pred.get("tokens", [])
                prompt_preview = "".join(tokens) if tokens else "?"
                for c in pred["continuations"]:
                    true_acts.append(c.get("true", 0))
                    pred_acts.append(c.get("predicted", 0))
                    token_labels.append(f"[ex{pred_idx}] {c.get('token', '?')}\n{prompt_preview}")
            else:
                # Standard format: tokens, true, predicted arrays
                tokens = pred.get("tokens", [])
                true_vals = pred.get("true", [])
                pred_vals = pred.get("predicted", [])
                prompt_preview = "".join(tokens) if tokens else "?"
                for i, (t, p) in enumerate(zip(true_vals, pred_vals)):
                    true_acts.append(t)
                    pred_acts.append(p)
                    tok = tokens[i] if i < len(tokens) else "?"
                    token_labels.append(f"[ex{pred_idx}] {tok}\n{prompt_preview}")
        return true_acts, pred_acts, token_labels

    def get_true_pred_last_token(
        expls: list[dict],
    ) -> tuple[list[float], list[float], list[str]]:
        """Extract last-token (true, predicted, token) from top explanation's predictions."""
        if not expls:
            return [], [], []
        sorted_expls = sorted(
            expls, key=lambda e: e["score"] if e["score"] is not None else -999, reverse=True
        )
        top = sorted_expls[0]
        predictions = top.get("predictions", [])
        if not predictions:
            return [], [], []
        true_acts: list[float] = []
        pred_acts: list[float] = []
        token_labels: list[str] = []
        for pred_idx, pred in enumerate(predictions):
            true_vals = pred.get("true", [])
            pred_vals = pred.get("predicted", [])
            tokens = pred.get("tokens", [])
            if true_vals and pred_vals:
                true_acts.append(true_vals[-1])
                pred_acts.append(pred_vals[-1])
                tok = tokens[-1] if tokens else "?"
                prompt_preview = "".join(tokens) if tokens else "?"
                token_labels.append(f"[ex{pred_idx}] {tok}\n{prompt_preview}")
        return true_acts, pred_acts, token_labels

    def get_true_pred_from_paired(
        pairs: list[dict],
    ) -> tuple[list[float], list[float], list[float]]:
        """Extract (true, pos_pred, neg_pred) from top paired explanation's predictions."""
        if not pairs:
            return [], [], []
        sorted_pairs = sorted(
            pairs, key=lambda e: e["score"] if e.get("score") is not None else -999, reverse=True
        )
        top = sorted_pairs[0]
        predictions = top.get("predictions", [])
        if not predictions:
            return [], [], []
        true_acts = []
        pos_preds = []
        neg_preds = []
        for pred in predictions:
            true_acts.extend(pred.get("true", []))
            pos_preds.extend(pred.get("pos_pred", []))
            neg_preds.extend(pred.get("neg_pred", []))
        return true_acts, pos_preds, neg_preds

    for c in sorted_clusters:
        avg_class = (
            "score-high"
            if c["avg_score"] > 0.7
            else ("score-mid" if c["avg_score"] > 0.4 else "score-low")
        )
        max_class = (
            "score-high"
            if c["max_score"] > 0.8
            else ("score-mid" if c["max_score"] > 0.5 else "score-low")
        )

        # Collapsed view: show top-1 explanation per type
        # Check if we have paired results (show all three: pos-only, neg-only, paired)
        has_paired = c["explanations"].get("attr_paired") or c["explanations"].get("contrib_paired")

        collapsed_summaries = []
        # Get best sign for each category
        category_best = c.get("category_best", {})
        attr_best_sign = category_best.get("attr", (None, None))[1]
        contrib_best_sign = category_best.get("contrib", (None, None))[1]

        if has_paired:
            # Show paired + pos-only + neg-only for comparison
            # Attr: paired, then pos-only, then neg-only
            attr_paired = get_top_paired_explanation(
                c["explanations"].get("attr_paired", []), "attr_paired"
            )
            attr_pos = get_top_explanation(
                c["explanations"].get("attr_pos", []), "attr_pos", is_best=(attr_best_sign == "pos")
            )
            attr_neg = get_top_explanation(
                c["explanations"].get("attr_neg", []), "attr_neg", is_best=(attr_best_sign == "neg")
            )
            # Contrib: paired, then pos-only, then neg-only
            contrib_paired = get_top_paired_explanation(
                c["explanations"].get("contrib_paired", []), "contrib_paired"
            )
            contrib_pos = get_top_explanation(
                c["explanations"].get("contrib_pos", []),
                "contrib_pos",
                is_best=(contrib_best_sign == "pos"),
            )
            contrib_neg = get_top_explanation(
                c["explanations"].get("contrib_neg", []),
                "contrib_neg",
                is_best=(contrib_best_sign == "neg"),
            )
            # Group by category: attr first, then contrib
            for s in [attr_paired, attr_pos, attr_neg, contrib_paired, contrib_pos, contrib_neg]:
                if s:
                    collapsed_summaries.append(s)
        else:
            # Standard pos/neg view
            for type_key in [
                "attr_pos",
                "attr_neg",
                "attr_combined",
                "contrib_pos",
                "contrib_neg",
                "contrib_combined",
            ]:
                if not c["explanations"].get(type_key):
                    continue
                # Determine if this type is the best for its category
                category = type_key.split("_")[0]  # "attr" or "contrib"
                sign = "_".join(type_key.split("_")[1:])  # "pos", "neg", or "combined"
                best_sign = attr_best_sign if category == "attr" else contrib_best_sign
                is_best = sign == best_sign
                summary = get_top_explanation(
                    c["explanations"][type_key], type_key, is_best=is_best
                )
                if summary:
                    collapsed_summaries.append(summary)

            # Contrib last-token summaries
            for sign, lt_type_key in [("pos", "contrib_lt_pos"), ("neg", "contrib_lt_neg")]:
                lt_summary = get_top_explanation_by_lt(
                    c["explanations"].get(f"contrib_{sign}", []), lt_type_key
                )
                if lt_summary:
                    collapsed_summaries.append(lt_summary)
        # Each summary is already a div with flexbox layout, so just concatenate
        collapsed_html = (
            "".join(collapsed_summaries) if collapsed_summaries else "<em>No explanations</em>"
        )

        # Histograms of true token scores (attr and contrib, pos only)
        attr_pos_acts = get_true_acts_from_expls(c["explanations"].get("attr_pos", []))
        contrib_pos_acts = get_true_acts_from_expls(
            c["explanations"].get("contrib_pos", [])
            or c["explanations"].get("contrib_combined", [])
        )

        histograms_html = '<div class="histograms-row">'
        histograms_html += make_histogram_svg(attr_pos_acts, [], "Attribution Scores")
        histograms_html += make_histogram_svg(contrib_pos_acts, [], "Contribution Scores")
        histograms_html += "</div>"

        # Scatter plots of true vs predicted for top explanations
        scatter_html = '<div class="scatter-row">'
        for type_key, color in [
            ("attr_pos", "#1565c0"),
            ("attr_neg", "#c2185b"),
            ("attr_combined", "#4527a0"),
            ("contrib_pos", "#2e7d32"),
            ("contrib_neg", "#ef6c00"),
            ("contrib_combined", "#00695c"),
        ]:
            if not c["explanations"].get(type_key):
                continue
            true_acts, pred_acts, tok_labels = get_true_pred_from_expls(c["explanations"][type_key])
            # Extract threshold from exemplars if available
            exemplars = c["exemplars"].get(type_key, [])
            thresh = exemplars[0].get("highlight_threshold") if exemplars else None
            # Negate threshold for neg types (stored as positive, but true acts are negative)
            if thresh is not None and "neg" in type_key:
                thresh = -thresh
            scatter_html += make_scatter_svg(
                true_acts,
                pred_acts,
                f"{type_key}",
                color=color,
                threshold=thresh,
                token_labels=tok_labels,
            )

        # Add contrib last-token scatter plots
        for type_key, color in [
            ("contrib_pos", "#558b2f"),
            ("contrib_neg", "#e65100"),
        ]:
            if not c["explanations"].get(type_key):
                continue
            true_acts, pred_acts, tok_labels = get_true_pred_last_token(c["explanations"][type_key])
            exemplars = c["exemplars"].get(type_key, [])
            thresh = exemplars[0].get("highlight_threshold") if exemplars else None
            if thresh is not None and "neg" in type_key:
                thresh = -thresh
            scatter_html += make_scatter_svg(
                true_acts,
                pred_acts,
                f"{type_key} (last-tok)",
                color=color,
                threshold=thresh,
                token_labels=tok_labels,
            )

        # Add paired scatter plots if available (pos_pred and neg_pred shown separately)
        # Note: neg_pred is negated so both plots show positive correlation for good explanations
        # (simulator outputs are always positive, but neg explains negative activations)
        for paired_key, pos_key, neg_key, pos_color, neg_color in [
            ("attr_paired", "attr_pos", "attr_neg", "#1565c0", "#c2185b"),
            ("contrib_paired", "contrib_pos", "contrib_neg", "#2e7d32", "#ef6c00"),
        ]:
            if c["explanations"].get(paired_key):
                true_acts, pos_preds, neg_preds = get_true_pred_from_paired(
                    c["explanations"][paired_key]
                )
                # Get thresholds from corresponding pos/neg exemplars
                pos_exemplars = c["exemplars"].get(pos_key, [])
                neg_exemplars = c["exemplars"].get(neg_key, [])
                pos_thresh = pos_exemplars[0].get("highlight_threshold") if pos_exemplars else None
                neg_thresh = neg_exemplars[0].get("highlight_threshold") if neg_exemplars else None
                if neg_thresh is not None:
                    neg_thresh = -neg_thresh
                scatter_html += make_scatter_svg(
                    true_acts,
                    pos_preds,
                    f"{paired_key} (pos)",
                    color=pos_color,
                    threshold=pos_thresh,
                )
                # Negate neg_preds: high neg_pred should correlate with negative true
                neg_preds_negated = [-n for n in neg_preds]
                scatter_html += make_scatter_svg(
                    true_acts,
                    neg_preds_negated,
                    f"{paired_key} (-neg)",
                    color=neg_color,
                    threshold=neg_thresh,
                )
        scatter_html += "</div>"

        # Expanded view: show explanations with their exemplars under each type
        expanded_html = histograms_html + scatter_html

        # Check if we have paired results
        has_paired = c["explanations"].get("attr_paired") or c["explanations"].get("contrib_paired")

        if has_paired:
            # Show paired sections + pos-only + neg-only (combine pos+neg exemplars for lookup)
            attr_all_exemplars = c["exemplars"]["attr_pos"] + c["exemplars"]["attr_neg"]
            contrib_all_exemplars = c["exemplars"]["contrib_pos"] + c["exemplars"]["contrib_neg"]

            # Attr: paired, then pos-only, then neg-only
            if c["explanations"].get("attr_paired"):
                expanded_html += format_paired_section(
                    c["explanations"]["attr_paired"], attr_all_exemplars, "attr_paired"
                )
            if c["explanations"].get("attr_pos"):
                expanded_html += format_expl_section_with_exemplars(
                    c["explanations"]["attr_pos"], c["exemplars"]["attr_pos"], "attr_pos"
                )
            if c["explanations"].get("attr_neg"):
                expanded_html += format_expl_section_with_exemplars(
                    c["explanations"]["attr_neg"], c["exemplars"]["attr_neg"], "attr_neg"
                )

            # Contrib: paired, then pos-only, then neg-only
            if c["explanations"].get("contrib_paired"):
                expanded_html += format_paired_section(
                    c["explanations"]["contrib_paired"], contrib_all_exemplars, "contrib_paired"
                )
            if c["explanations"].get("contrib_pos"):
                expanded_html += format_expl_section_with_exemplars(
                    c["explanations"]["contrib_pos"], c["exemplars"]["contrib_pos"], "contrib_pos"
                )
            if c["explanations"].get("contrib_neg"):
                expanded_html += format_expl_section_with_exemplars(
                    c["explanations"]["contrib_neg"], c["exemplars"]["contrib_neg"], "contrib_neg"
                )
        else:
            # Standard pos/neg/combined sections
            for type_key in [
                "attr_pos",
                "attr_neg",
                "attr_combined",
                "contrib_pos",
                "contrib_neg",
                "contrib_combined",
            ]:
                if not c["explanations"].get(type_key):
                    continue
                expanded_html += format_expl_section_with_exemplars(
                    c["explanations"][type_key], c["exemplars"].get(type_key, []), type_key
                )

            # Contrib last-token sections (same explanations, sorted by score_last_token)
            for sign in ["pos", "neg"]:
                expanded_html += format_expl_section_by_lt(
                    c["explanations"].get(f"contrib_{sign}", []),
                    f"contrib_lt_{sign}",
                    exemplars=c["exemplars"].get(f"contrib_{sign}", []),
                )

        # Build neurons list (collapsible)
        neurons = c["neurons"]
        if neurons:
            neurons_list = ", ".join(
                f'<a href="https://neurons.transluce.org/{n["layer"]}/{n["neuron"]}/{n.get("polarity", "+")}" '
                f'target="_blank" style="color: #1565c0; text-decoration: none;" '
                f'onclick="event.stopPropagation();">'
                f'L{n["layer"]}N{n["neuron"]}{n.get("polarity", "+")}</a>'
                for n in neurons[:50]
            )
            if len(neurons) > 50:
                neurons_list += f" ... (+{len(neurons) - 50} more)"
            expanded_html += f"""<div class="exemplars-container" onclick="toggleExemplars(event, this)">
                <div class="exemplars-toggle"><span class="expand-icon">▶</span> Neurons ({len(neurons)} total)</div>
                <div class="exemplars-content"><div class="neuron-list">{neurons_list}</div></div>
            </div>"""

        html += f"""                <tr class="cluster-row" data-layer="{c['layer']}" data-polarity="{c['polarity']}" data-avg="{c['avg_score']:.4f}" onclick="toggleRow(this)">
                    <td class="cluster-id"><span class="expand-icon">+</span>{c['cluster_id']}</td>
                    <td>{c['layer']}</td>
                    <td>{c['num_neurons']}</td>
                    <td class="score {avg_class}">{c['avg_score']:.3f}</td>
                    <td class="score {max_class}">{c['max_score']:.3f}</td>
                    <td>
                        <div class="collapsed-content summary-text">{collapsed_html}</div>
                        <div class="expanded-content">{expanded_html}</div>
                    </td>
                </tr>
"""

    # Export cluster data for JS
    cluster_data_for_js = [
        {
            "layer": c["layer"],
            "polarity": c["polarity"],
            "type_max_scores": c["type_max_scores"],
            "cluster_id": c["cluster_id"],
            "best_descs": {
                cat: (
                    lambda e: {"explanation": e["explanation"], "score": e["score"]} if e else None
                )(
                    max(
                        (
                            e
                            for e in c["explanations"].get(
                                f"{cat}_{c['category_best'][cat][1]}", []
                            )
                            if e.get("score") is not None
                        ),
                        key=lambda e: e["score"],
                        default=None,
                    )
                )
                for cat in ["attr", "contrib"]
                if cat in c.get("category_best", {})
            },
        }
        for c in clusters.values()
    ]
    import json as json_module

    cluster_json = json_module.dumps(cluster_data_for_js)

    html += f"""            </tbody>
        </table>
    </div>

    <script>
        const clusterData = {cluster_json};
        const typeColors = {{
            'attr_pos': '#1565c0',
            'attr_neg': '#c2185b',
            'contrib_pos': '#2e7d32',
            'contrib_neg': '#ef6c00'
        }};
        const types = ['attr_pos', 'attr_neg', 'contrib_pos', 'contrib_neg'];

        // Toggle row expansion
        function toggleRow(row) {{
            row.classList.toggle('expanded');
            const icon = row.querySelector('.expand-icon');
            icon.textContent = row.classList.contains('expanded') ? '−' : '+';
        }}

        // Toggle exemplars section (nested within expanded row)
        function toggleExemplars(event, container) {{
            event.stopPropagation();  // Prevent row toggle
            container.classList.toggle('expanded');
            const icon = container.querySelector('.expand-icon');
            icon.textContent = container.classList.contains('expanded') ? '▼' : '▶';
        }}

        // Toggle predictions for an explanation
        function togglePredictions(event, explItem) {{
            event.stopPropagation();  // Prevent row toggle
            const container = explItem.parentElement;
            container.classList.toggle('expanded');
            const toggle = explItem.querySelector('.pred-toggle');
            if (toggle) {{
                const isExpanded = container.classList.contains('expanded');
                toggle.textContent = toggle.textContent.replace(/[▶▼]/, isExpanded ? '▼' : '▶');
            }}
        }}

        // Populate layer/cluster filter
        const layers = [...new Set([...document.querySelectorAll('tr[data-layer]')].map(r => r.dataset.layer))].sort((a,b) => a-b);
        const layerSelect = document.getElementById('layerFilter');
        layers.forEach(l => {{
            const opt = document.createElement('option');
            opt.value = l;
            opt.textContent = 'Cluster ' + l;
            layerSelect.appendChild(opt);
        }});

        function getFilteredData(layerFilter) {{
            return clusterData.filter(c => !layerFilter || c.layer == layerFilter);
        }}

        function computeStats(scores) {{
            if (!scores.length) return {{ n: 0, mean: 0, median: 0, std: 0, min: 0, max: 0 }};
            const sorted = [...scores].sort((a, b) => a - b);
            const n = scores.length;
            const mean = scores.reduce((a, b) => a + b, 0) / n;
            const median = sorted[Math.floor(n / 2)];
            const variance = scores.reduce((sum, s) => sum + (s - mean) ** 2, 0) / n;
            const std = Math.sqrt(variance);
            return {{ n, mean, median, std, min: sorted[0], max: sorted[n - 1] }};
        }}

        function makeHistogram(scores, bins = 20) {{
            // Fixed range 0 to 1
            const minS = 0, maxS = 1;
            const binWidth = (maxS - minS) / bins;
            const hist = new Array(bins).fill(0);
            scores.forEach(s => {{
                // Clamp to [0, 1] range
                const clamped = Math.max(0, Math.min(1, s));
                const idx = Math.min(Math.floor((clamped - minS) / binWidth), bins - 1);
                hist[idx]++;
            }});
            return hist.map((count, i) => [minS + i * binWidth, minS + (i + 1) * binWidth, count]);
        }}

        function renderHistogram(containerId, scores, color) {{
            const container = document.getElementById(containerId);
            const width = 200, height = 120;
            if (!scores.length) {{
                container.innerHTML = `<svg width="${{width}}" height="${{height}}"><text x="50" y="60" font-size="12" fill="#999">No data</text></svg>`;
                return;
            }}
            const hist = makeHistogram(scores, 20);
            const maxCount = Math.max(...hist.map(h => h[2]));
            const total = hist.reduce((sum, h) => sum + h[2], 0);
            const barW = width / hist.length;
            let bars = '';
            hist.forEach(([low, high, count], i) => {{
                const barHeight = maxCount > 0 ? (count / maxCount) * (height - 20) : 0;
                const x = i * barW;
                const pct = total > 0 ? (count / total * 100).toFixed(1) : 0;
                const tooltip = `Range: [${{low.toFixed(2)}}, ${{high.toFixed(2)}})&#10;Count: ${{count}} (${{pct}}%)`;
                bars += `<rect x="${{x}}" y="${{height - 15 - barHeight}}" width="${{barW - 1}}" height="${{barHeight}}" fill="${{color}}"><title>${{tooltip}}</title></rect>`;
            }});
            container.innerHTML = `<svg width="${{width}}" height="${{height}}" viewBox="0 0 ${{width}} ${{height}}">${{bars}}<text x="0" y="${{height - 2}}" font-size="9" fill="#666">0</text><text x="${{width}}" y="${{height - 2}}" font-size="9" fill="#666" text-anchor="end">1</text></svg>`;
        }}

        function renderBoxplot(containerId, scoresByLayer, color) {{
            const container = document.getElementById(containerId);
            const width = 280, height = 120;
            const layerData = Object.entries(scoresByLayer)
                .filter(([_, scores]) => scores.length > 0)
                .map(([layer, scores]) => [parseInt(layer), [...scores].sort((a, b) => a - b)])
                .sort((a, b) => a[0] - b[0]);

            if (!layerData.length) {{
                container.innerHTML = `<svg width="${{width}}" height="${{height}}"><text x="50" y="60" font-size="12" fill="#999">No data</text></svg>`;
                return;
            }}

            const yMin = -1, yMax = 1, yRange = yMax - yMin;
            const plotHeight = height - 25, plotTop = 5;
            const yToSvg = val => plotTop + plotHeight - ((val - yMin) / yRange) * plotHeight;
            const boxW = width / layerData.length;
            let elements = '';

            // Zero line
            const zeroY = yToSvg(0);
            elements += `<line x1="0" y1="${{zeroY}}" x2="${{width}}" y2="${{zeroY}}" stroke="#ccc" stroke-dasharray="2"/>`;

            layerData.forEach(([layer, scores], i) => {{
                const n = scores.length;
                const xCenter = i * boxW + boxW / 2;
                const boxHalf = boxW * 0.35;

                const q1Idx = Math.floor(n * 0.25), q2Idx = Math.floor(n * 0.5), q3Idx = Math.floor(n * 0.75);
                const q1 = scores[q1Idx], median = scores[q2Idx], q3 = scores[q3Idx];
                const iqr = q3 - q1;
                const whiskerLow = Math.max(scores[0], q1 - 1.5 * iqr);
                const whiskerHigh = Math.min(scores[n - 1], q3 + 1.5 * iqr);
                const mean = scores.reduce((a, b) => a + b, 0) / n;

                const yQ1 = yToSvg(q1), yQ3 = yToSvg(q3), yMed = yToSvg(median);
                const yWl = yToSvg(whiskerLow), yWh = yToSvg(whiskerHigh);

                const tooltip = `Layer ${{layer}}&#10;n=${{n}}&#10;median=${{median.toFixed(3)}}&#10;mean=${{mean.toFixed(3)}}&#10;Q1=${{q1.toFixed(3)}}, Q3=${{q3.toFixed(3)}}&#10;whiskers=[${{whiskerLow.toFixed(3)}}, ${{whiskerHigh.toFixed(3)}}]`;

                elements += `<rect x="${{xCenter - boxHalf}}" y="${{plotTop}}" width="${{boxHalf * 2}}" height="${{plotHeight}}" fill="transparent"><title>${{tooltip}}</title></rect>`;
                elements += `<rect x="${{xCenter - boxHalf}}" y="${{yQ3}}" width="${{boxHalf * 2}}" height="${{yQ1 - yQ3}}" fill="${{color}}" fill-opacity="0.3" stroke="${{color}}" pointer-events="none"/>`;
                elements += `<line x1="${{xCenter - boxHalf}}" y1="${{yMed}}" x2="${{xCenter + boxHalf}}" y2="${{yMed}}" stroke="${{color}}" stroke-width="2" pointer-events="none"/>`;
                elements += `<line x1="${{xCenter}}" y1="${{yQ1}}" x2="${{xCenter}}" y2="${{yWl}}" stroke="${{color}}" pointer-events="none"/>`;
                elements += `<line x1="${{xCenter}}" y1="${{yQ3}}" x2="${{xCenter}}" y2="${{yWh}}" stroke="${{color}}" pointer-events="none"/>`;
                elements += `<line x1="${{xCenter - boxHalf * 0.5}}" y1="${{yWl}}" x2="${{xCenter + boxHalf * 0.5}}" y2="${{yWl}}" stroke="${{color}}" pointer-events="none"/>`;
                elements += `<line x1="${{xCenter - boxHalf * 0.5}}" y1="${{yWh}}" x2="${{xCenter + boxHalf * 0.5}}" y2="${{yWh}}" stroke="${{color}}" pointer-events="none"/>`;

                if (i % 8 === 0) {{
                    elements += `<text x="${{xCenter}}" y="${{height - 2}}" text-anchor="middle" font-size="8" fill="#666">${{layer}}</text>`;
                }}
            }});

            container.innerHTML = `<svg width="${{width}}" height="${{height}}" viewBox="0 0 ${{width}} ${{height}}">${{elements}}</svg>`;
        }}

        function updateCharts(layerFilter) {{
            const filtered = getFilteredData(layerFilter);

            // Compute best scores (max of pos/neg) for each category
            const attrBestScores = filtered
                .map(n => {{
                    const pos = n.type_max_scores['attr_pos'];
                    const neg = n.type_max_scores['attr_neg'];
                    if (pos !== undefined && neg !== undefined) return Math.max(pos, neg);
                    if (pos !== undefined) return pos;
                    if (neg !== undefined) return neg;
                    return undefined;
                }})
                .filter(s => s !== undefined);

            const contribCombinedBestScores = filtered
                .map(n => {{
                    const score = n.type_max_scores['contrib_combined'];
                    return score !== undefined ? score : undefined;
                }})
                .filter(s => s !== undefined);

            // Render best score histograms
            renderHistogram('hist-attr_best', attrBestScores, '#1565c0');
            renderHistogram('hist-contrib_combined_best', contribCombinedBestScores, '#00695c');

            // Update best score labels
            const attrBestStats = computeStats(attrBestScores);
            const attrBestLabel = document.getElementById('hist-label-attr_best');
            attrBestLabel.innerHTML = `<b>attr_best</b> n=${{attrBestStats.n}} | mean=${{attrBestStats.mean.toFixed(3)}}`;
            attrBestLabel.title = `n=${{attrBestStats.n}} clusters\\nmean=${{attrBestStats.mean.toFixed(3)}}\\nmedian=${{attrBestStats.median.toFixed(3)}}\\nstd=${{attrBestStats.std.toFixed(3)}}\\nrange=[${{attrBestStats.min.toFixed(3)}}, ${{attrBestStats.max.toFixed(3)}}]`;

            const contribCombinedBestStats = computeStats(contribCombinedBestScores);
            const contribCombinedBestLabel = document.getElementById('hist-label-contrib_combined_best');
            contribCombinedBestLabel.innerHTML = `<b>contrib_combined</b> n=${{contribCombinedBestStats.n}} | mean=${{contribCombinedBestStats.mean.toFixed(3)}}`;
            contribCombinedBestLabel.title = `n=${{contribCombinedBestStats.n}} clusters\\nmean=${{contribCombinedBestStats.mean.toFixed(3)}}\\nmedian=${{contribCombinedBestStats.median.toFixed(3)}}\\nstd=${{contribCombinedBestStats.std.toFixed(3)}}\\nrange=[${{contribCombinedBestStats.min.toFixed(3)}}, ${{contribCombinedBestStats.max.toFixed(3)}}]`;

            // Render attr_best vs contrib_combined scatter plot
            const scatterPairs = filtered
                .map(n => {{
                    const attrPos = n.type_max_scores['attr_pos'];
                    const attrNeg = n.type_max_scores['attr_neg'];
                    const attr = (attrPos !== undefined && attrNeg !== undefined) ? Math.max(attrPos, attrNeg)
                               : (attrPos !== undefined) ? attrPos
                               : (attrNeg !== undefined) ? attrNeg : undefined;
                    const contrib = n.type_max_scores['contrib_combined'];
                    if (attr !== undefined && contrib !== undefined) {{
                        const descs = n.best_descs || {{}};
                        const attrDesc = descs.attr ? `attr: ${{descs.attr.explanation}} (${{descs.attr.score.toFixed(3)}})` : '';
                        const contribDesc = descs.contrib ? `contrib: ${{descs.contrib.explanation}} (${{descs.contrib.score.toFixed(3)}})` : '';
                        return {{ x: attr, y: contrib, name: n.cluster_id, attrDesc, contribDesc }};
                    }}
                    return undefined;
                }})
                .filter(p => p !== undefined);

            const scatterContainer = document.getElementById('scatter-attr-vs-contrib');
            const scatterLabel = document.getElementById('scatter-label-attr-vs-contrib');
            if (scatterPairs.length === 0) {{
                scatterContainer.innerHTML = '<em>No data</em>';
            }} else {{
                const width = 200, height = 120, pad = 25;
                const xs = scatterPairs.map(p => p.x), ys = scatterPairs.map(p => p.y);
                const xMin = Math.min(...xs), xMax = Math.max(...xs);
                const yMin = Math.min(...ys), yMax = Math.max(...ys);
                const xRange = xMax - xMin || 1, yRange = yMax - yMin || 1;
                const sx = x => pad + (x - xMin) / xRange * (width - 2 * pad);
                const sy = y => height - pad - (y - yMin) / yRange * (height - 2 * pad);

                let dots = '';
                scatterPairs.forEach(p => {{
                    const tipLines = [p.name, `attr_best=${{p.x.toFixed(3)}} contrib_combined=${{p.y.toFixed(3)}}`];
                    if (p.attrDesc) tipLines.push(p.attrDesc);
                    if (p.contribDesc) tipLines.push(p.contribDesc);
                    const tip = tipLines.join('&#10;').replace(/"/g, '&quot;');
                    dots += `<circle cx="${{sx(p.x)}}" cy="${{sy(p.y)}}" r="2.5" fill="#1565c0" opacity="0.5"><title>${{tip}}</title></circle>`;
                }});

                // Axes
                const axisColor = '#999';
                const axes = `
                    <line x1="${{pad}}" y1="${{height - pad}}" x2="${{width - pad}}" y2="${{height - pad}}" stroke="${{axisColor}}" stroke-width="0.5"/>
                    <line x1="${{pad}}" y1="${{pad}}" x2="${{pad}}" y2="${{height - pad}}" stroke="${{axisColor}}" stroke-width="0.5"/>
                    <text x="${{width / 2}}" y="${{height - 3}}" text-anchor="middle" font-size="9" fill="#666">attr_best</text>
                    <text x="3" y="${{height / 2}}" text-anchor="middle" font-size="9" fill="#666" transform="rotate(-90, 8, ${{height / 2}})">contrib_comb</text>
                    <text x="${{pad}}" y="${{height - pad + 12}}" font-size="8" fill="#999">${{xMin.toFixed(2)}}</text>
                    <text x="${{width - pad}}" y="${{height - pad + 12}}" text-anchor="end" font-size="8" fill="#999">${{xMax.toFixed(2)}}</text>
                    <text x="${{pad - 3}}" y="${{height - pad}}" text-anchor="end" font-size="8" fill="#999">${{yMin.toFixed(2)}}</text>
                    <text x="${{pad - 3}}" y="${{pad + 4}}" text-anchor="end" font-size="8" fill="#999">${{yMax.toFixed(2)}}</text>
                `;

                scatterContainer.innerHTML = `<svg width="${{width}}" height="${{height}}" viewBox="0 0 ${{width}} ${{height}}">${{axes}}${{dots}}</svg>`;

                // Compute correlation
                const n = scatterPairs.length;
                const meanX = xs.reduce((a, b) => a + b, 0) / n;
                const meanY = ys.reduce((a, b) => a + b, 0) / n;
                let cov = 0, varX = 0, varY = 0;
                for (let i = 0; i < n; i++) {{
                    const dx = xs[i] - meanX, dy = ys[i] - meanY;
                    cov += dx * dy; varX += dx * dx; varY += dy * dy;
                }}
                const r = (varX > 0 && varY > 0) ? (cov / Math.sqrt(varX * varY)) : 0;
                scatterLabel.innerHTML = `<b>attr_best vs contrib_combined</b> n=${{n}} | r=${{r.toFixed(3)}}`;
                scatterLabel.title = `n=${{n}} pairs\\nPearson r=${{r.toFixed(3)}}`;
            }}

        }}

        function updateCount() {{
            const visible = document.querySelectorAll('#explanationsTable tbody tr:not(.hidden)').length;
            const total = document.querySelectorAll('#explanationsTable tbody tr').length;
            document.getElementById('visibleCount').textContent = `Showing ${{visible}} of ${{total}} clusters`;
        }}

        function filterTable() {{
            const search = document.getElementById('search').value.toLowerCase();
            const layerFilter = document.getElementById('layerFilter').value;
            const polarityFilter = document.getElementById('polarityFilter').value;
            const highScoreOnly = document.getElementById('highScoreOnly').checked;

            document.querySelectorAll('#explanationsTable tbody tr').forEach(row => {{
                const text = row.textContent.toLowerCase();
                const layer = row.dataset.layer;
                const polarity = row.dataset.polarity;
                const avg = parseFloat(row.dataset.avg);

                const matchesSearch = text.includes(search);
                const matchesLayer = !layerFilter || layer === layerFilter;
                const matchesPolarity = !polarityFilter || polarity === polarityFilter;
                const matchesScore = !highScoreOnly || avg > 0.7;

                row.classList.toggle('hidden', !(matchesSearch && matchesLayer && matchesPolarity && matchesScore));
            }});
            updateCount();
            updateCharts(layerFilter);
        }}

        let sortDir = {{2: true}};  // Default sort by avg score descending
        function sortTable(col) {{
            const table = document.getElementById('explanationsTable');
            const tbody = table.querySelector('tbody');
            const rows = [...tbody.querySelectorAll('tr')];

            sortDir[col] = !sortDir[col];
            const dir = sortDir[col] ? 1 : -1;

            rows.sort((a, b) => {{
                let aVal = a.cells[col].textContent.trim();
                let bVal = b.cells[col].textContent.trim();

                // Try numeric sort
                const aNum = parseFloat(aVal);
                const bNum = parseFloat(bVal);
                if (!isNaN(aNum) && !isNaN(bNum)) {{
                    return (aNum - bNum) * dir;
                }}
                return aVal.localeCompare(bVal) * dir;
            }});

            rows.forEach(row => tbody.appendChild(row));
        }}

        // Tooltip handling
        const tooltip = document.getElementById('tooltip');
        document.addEventListener('mouseover', e => {{
            const el = e.target.closest('.tip');
            if (el && el.dataset.tip) {{
                tooltip.textContent = el.dataset.tip;
                tooltip.style.display = 'block';
            }}
        }});
        document.addEventListener('mouseout', e => {{
            const el = e.target.closest('.tip');
            if (el) tooltip.style.display = 'none';
        }});
        document.addEventListener('mousemove', e => {{
            if (tooltip.style.display === 'block') {{
                tooltip.style.left = (e.clientX + 12) + 'px';
                tooltip.style.top = (e.clientY - 10) + 'px';
            }}
        }});

        // Initial render
        updateCount();
        updateCharts('');
    </script>
</body>
</html>
"""

    output_path.write_text(html)
    print(f"Generated HTML: {output_path}")


def get_default_output_dir() -> Path:
    """Get default output directory (outputs/)."""
    # Navigate from this file's location to outputs/
    return Path(__file__).parent.parent.parent / "outputs"


def generate_multi_html(input_paths: list[Path], output_path: Path) -> None:
    """Generate a wrapper HTML with a dropdown to switch between multiple visualization files.

    Each input JSON gets its own HTML page; the wrapper uses an iframe + dropdown.
    """
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate individual HTML files
    child_files: list[tuple[str, str]] = []  # (label, filename)
    for input_path in input_paths:
        with open(input_path) as f:
            data = json.load(f)
        child_name = f"{input_path.stem}.html"
        child_path = output_dir / child_name
        generate_html(data, child_path)
        child_files.append((input_path.stem, child_name))

    # Generate wrapper
    options_html = "\n".join(
        f'            <option value="{fname}">{label}</option>' for label, fname in child_files
    )
    wrapper = f"""<!DOCTYPE html>
<html>
<head>
    <title>Model Grid Comparison</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        .toolbar {{
            position: fixed; top: 0; left: 0; right: 0; z-index: 100;
            background: #1a1a2e; color: white; padding: 12px 20px;
            display: flex; align-items: center; gap: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }}
        .toolbar h1 {{ font-size: 16px; font-weight: 600; }}
        .toolbar select {{
            padding: 6px 12px; border-radius: 4px; border: 1px solid #444;
            background: #16213e; color: white; font-size: 14px; cursor: pointer;
        }}
        .toolbar select:hover {{ border-color: #888; }}
        iframe {{
            position: fixed; top: 50px; left: 0; right: 0; bottom: 0;
            width: 100%; height: calc(100vh - 50px); border: none;
        }}
    </style>
</head>
<body>
    <div class="toolbar">
        <h1>Model Grid Comparison</h1>
        <select id="file-select" onchange="document.getElementById('viewer').src=this.value">
{options_html}
        </select>
    </div>
    <iframe id="viewer" src="{child_files[0][1]}"></iframe>
</body>
</html>"""
    output_path.write_text(wrapper)
    print(f"Generated multi-file HTML: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize neuron explanations")
    parser.add_argument("input", nargs="+", type=Path, help="Input JSON file(s)")
    parser.add_argument(
        "--html", type=Path, help="Generate HTML output (default: outputs/<input_stem>.html)"
    )
    parser.add_argument("--top", type=int, default=20, help="Show top N explanations")
    parser.add_argument("--neuron", type=str, help="Show details for specific neuron ID")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML generation")
    args = parser.parse_args()

    # Multi-file mode: generate wrapper with dropdown
    if len(args.input) > 1:
        if not args.no_html:
            if args.html:
                output_path = args.html
            else:
                output_dir = get_default_output_dir()
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / "model_grid.html"
            generate_multi_html(args.input, output_path)
        return

    # Single-file mode (original behavior)
    input_path = args.input[0]
    with open(input_path) as f:
        data = json.load(f)

    print_summary(data)

    if args.neuron:
        print_neuron_details(data, args.neuron)
    else:
        print_top_explanations(data, args.top)

    if not args.no_html:
        if args.html:
            output_path = args.html
        else:
            # Default to outputs/ directory
            output_dir = get_default_output_dir()
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{input_path.stem}.html"
        generate_html(data, output_path)


if __name__ == "__main__":
    main()

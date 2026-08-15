"""
report_counts.py — grouped/ungrouped counts for a directory of grouped graphs.

Emits the row for the "Grouped vs. ungrouped nodes per Capitals graph" table, in both
markdown and LaTeX, plus a --check mode that reports coverage gaps (neurons missing a
description or a group, and the per-graph neuron-count distribution).

Counts are means per graph; the fraction grouped is pooled across all graphs.

Usage:
    # verify an export before annotating it
    python3 custom_automation/sweep/report_counts.py --graphs-dir capital_neuron_graphs --check

    # the table row
    python3 custom_automation/sweep/report_counts.py \
        --graphs-dir capital_neuron_graphs --label "MLP neurons (150)"
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def _described(neuron: dict) -> bool:
    d = (neuron.get("generated_description") or "").strip()
    return bool(d) and d != "Error generating description"


def main() -> None:
    ap = argparse.ArgumentParser(description="Grouped/ungrouped counts for a grouped graphs dir.")
    ap.add_argument("--graphs-dir", type=Path, required=True)
    ap.add_argument("--label", default=None, help="Row label (default: the directory name).")
    ap.add_argument("--check", action="store_true", help="Also report coverage gaps.")
    args = ap.parse_args()

    paths = sorted(args.graphs_dir.glob("graph_*.json"))
    if not paths:
        raise SystemExit(f"No graph_*.json found in {args.graphs_dir}")

    grouped = ungrouped = 0
    sizes: list[int] = []
    no_desc: dict[str, int] = {}
    no_group: dict[str, int] = {}
    ungroupable = 0

    for p in paths:
        g = json.loads(p.read_text(encoding="utf-8"))
        neurons = g.get("neurons", [])
        sizes.append(len(neurons))
        grouped += sum(len(ids) for ids in g.get("supernodes", {}).values())
        ungrouped += len(g.get("ungrouped", []))

        nd = sum(1 for n in neurons if not _described(n))
        ng = sum(1 for n in neurons if not n.get("group"))
        if nd:
            no_desc[p.name] = nd
        if ng:
            no_group[p.name] = ng
        ungroupable += nd

    n = len(paths)
    total = grouped + ungrouped
    frac = grouped / total if total else 0.0
    label = args.label or args.graphs_dir.name

    if args.check:
        print(f"graphs: {n} | neurons: {sum(sizes)} | mean {sum(sizes)/n:.2f} "
              f"| min {min(sizes)} | max {max(sizes)}")
        print(f"neuron-count distribution: {dict(sorted(collections.Counter(sizes).items()))}")
        print(f"neurons missing a description: {ungroupable}"
              f"{' across ' + str(len(no_desc)) + ' graphs' if no_desc else ''}")
        for k, v in sorted(no_desc.items(), key=lambda kv: -kv[1])[:20]:
            print(f"    {v:4d}  {k}")
        print(f"neurons missing a group: {sum(no_group.values())}"
              f"{' across ' + str(len(no_group)) + ' graphs' if no_group else ''}")
        for k, v in sorted(no_group.items(), key=lambda kv: -kv[1])[:20]:
            print(f"    {v:4d}  {k}")
        print()

    print(f"| Method | Grouped | Ungrouped | Total | Fraction grouped |")
    print(f"| --- | ---: | ---: | ---: | ---: |")
    print(f"| {label} | {grouped/n:.1f} | {ungrouped/n:.1f} | {total/n:.1f} | {100*frac:.1f}% |")
    print()
    print(f"    {label} & {grouped/n:.1f} & {ungrouped/n:.1f} & {total/n:.1f} & {100*frac:.1f}\\% \\\\")
    print()
    print(f"Counts are means per graph over the {n} graphs; the fraction grouped is pooled.")


if __name__ == "__main__":
    main()

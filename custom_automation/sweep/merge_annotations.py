"""
merge_annotations.py — carry descriptions/groups from an annotated graphs dir into a
freshly exported one.

Use after re-tracing at a looser `percentage_threshold` to fill the graphs whose pruned
circuit held fewer than the top-N budget. Because the prune is an absolute cut
(`goal_value * percentage_threshold`, clja.py:262) applied after attributions are
computed, every neuron admitted by the looser threshold ranks BELOW the ones already
there — the existing top-N ordering does not reshuffle. So the old annotations are still
valid and only the newly-admitted neurons need work.

For each graph (matched on `ci_idx`) this copies, keyed by feature id:

  - `generated_description` onto any new neuron that already had one
  - `group` + `supernodes` + `ungrouped`, but ONLY when the graph's feature set is
    unchanged — a graph that gained neurons must be regrouped, since Phase 3 reconciles
    over the whole feature set

Graphs that gained neurons are written to --write-todo so the regroup can target just
those. Everything is written in place into --new-dir; --old-dir is read-only.

Usage:
    python3 custom_automation/sweep/merge_annotations.py \
        --old-dir capital_neuron_graphs \
        --new-dir capital_neuron_graphs_p003 \
        --write-todo custom_automation/sweep/regroup_todo.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def feature_id(neuron: dict) -> str:
    """L{layer}_N{neuron}_{polarity} — must match generate_supernodes._feature_id."""
    pol = neuron.get("polarity", "")
    return f"L{neuron['layer']}_N{neuron['neuron']}{('_' + pol) if pol else ''}"


def _described(neuron: dict) -> bool:
    d = (neuron.get("generated_description") or "").strip()
    return bool(d) and d != "Error generating description"


def _index_by_ci(graphs_dir: Path) -> dict[int, Path]:
    """Map ci_idx -> path, so renamed slugs don't break the pairing."""
    out: dict[int, Path] = {}
    for p in sorted(graphs_dir.glob("graph_*.json")):
        try:
            ci = json.loads(p.read_text(encoding="utf-8"))["ci_idx"]
        except (json.JSONDecodeError, KeyError):
            print(f"  ! unreadable or missing ci_idx: {p}")
            continue
        out[int(ci)] = p
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Port descriptions/groups into a re-exported graphs dir.")
    ap.add_argument("--old-dir", type=Path, required=True, help="Annotated graphs (read-only).")
    ap.add_argument("--new-dir", type=Path, required=True, help="Freshly exported graphs (written in place).")
    ap.add_argument("--write-todo", type=Path, default=None, help="File listing graphs that need regrouping.")
    ap.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    args = ap.parse_args()

    old_by_ci = _index_by_ci(args.old_dir)
    new_by_ci = _index_by_ci(args.new_dir)
    if not new_by_ci:
        raise SystemExit(f"No graph_*.json found in {args.new_dir}")

    todo: list[Path] = []
    n_unchanged = n_grew = n_unpaired = 0
    desc_carried = desc_todo = neurons_gained = 0

    for ci in sorted(new_by_ci):
        new_path = new_by_ci[ci]
        old_path = old_by_ci.get(ci)
        new_graph = json.loads(new_path.read_text(encoding="utf-8"))
        new_neurons = new_graph.get("neurons", [])

        if old_path is None:
            print(f"  ! ci={ci} has no counterpart in {args.old_dir} — left untouched")
            n_unpaired += 1
            todo.append(new_path)
            desc_todo += sum(1 for n in new_neurons if not _described(n))
            continue

        old_graph = json.loads(old_path.read_text(encoding="utf-8"))
        old_neurons = old_graph.get("neurons", [])
        old_desc = {feature_id(n): n["generated_description"] for n in old_neurons if _described(n)}
        old_group = {feature_id(n): n["group"] for n in old_neurons if n.get("group")}

        for n in new_neurons:
            fid = feature_id(n)
            if not _described(n) and fid in old_desc:
                n["generated_description"] = old_desc[fid]
                desc_carried += 1
        desc_todo += sum(1 for n in new_neurons if not _described(n))

        old_ids = {feature_id(n) for n in old_neurons}
        new_ids = {feature_id(n) for n in new_neurons}
        gained = new_ids - old_ids

        if not gained and new_ids == old_ids:
            # Identical feature set — the existing grouping still applies verbatim.
            for n in new_neurons:
                fid = feature_id(n)
                if fid in old_group:
                    n["group"] = old_group[fid]
            if "supernodes" in old_graph:
                new_graph["supernodes"] = old_graph["supernodes"]
            if "ungrouped" in old_graph:
                new_graph["ungrouped"] = old_graph["ungrouped"]
            n_unchanged += 1
        else:
            # Feature set changed — Phase 3 reconciles over the whole set, so regroup.
            neurons_gained += len(gained)
            n_grew += 1
            todo.append(new_path)
            print(f"  ci={ci:<3} {new_path.name}: {len(old_ids)} -> {len(new_ids)} features (+{len(gained)})")

        if not args.dry_run:
            new_path.write_text(json.dumps(new_graph, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(f"graphs unchanged (grouping carried over): {n_unchanged}")
    print(f"graphs that gained neurons (need regroup): {n_grew}  (+{neurons_gained} neurons)")
    if n_unpaired:
        print(f"graphs with no old counterpart:           {n_unpaired}")
    print(f"descriptions carried over:                {desc_carried}")
    print(f"neurons still needing a description:      {desc_todo}")
    if args.dry_run:
        print("\n(dry run — nothing written)")

    if args.write_todo and not args.dry_run:
        args.write_todo.parent.mkdir(parents=True, exist_ok=True)
        args.write_todo.write_text("\n".join(str(p) for p in todo) + "\n", encoding="utf-8")
        print(f"\nwrote {len(todo)} paths to {args.write_todo}")


if __name__ == "__main__":
    main()

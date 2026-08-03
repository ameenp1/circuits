"""
sweep_supernodes.py — sweep the supernode grouping over the feature budget N.

Graphs exported with `--top-n 150` already contain every feature a smaller budget
would have used (batch_export_neurons.py ranks by |attribution|), so an N-sweep is
grouping-only: no re-trace, no re-export, no re-description.

Phase 1 seeds from the top-K features, which are identical in every arm, so it runs
once per graph and is shared. The arms then differ only in how much Phase 2 does:

    N = 50   ->  0 batches      N = 100  ->  1 batch      N = 150  ->  2 batches

Shortcut: if the top-K neurons of a source graph already carry a `group` (i.e. it
has been grouped before — e.g. the 150-feature run you already have), that grouping
is reused as the Phase 1 seed and no Phase 1 call is made.

Everything else — prompts, schemas, all three phases — is imported from
generate_supernodes.py unchanged; the only difference here is that Phase 2 is fed
`features[K:N]` rather than `features[K:]`.

Writes one graph per arm to `<out-dir>/n<N>/`, truncated to N neurons, leaving the
source directory untouched. Appends a summary table to the report .md per arm.

Usage:
    export OPENAI_API_KEY=sk-...
    uv run python custom_automation/sweep/sweep_supernodes.py \
        --graphs-dir capital_neuron_graphs \
        -n 50 -n 100 -n 150 \
        --out-dir custom_automation/sweep/sweep_out \
        --report custom_automation/sweep/sweep_results.md
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # custom_automation/

from openai import AsyncOpenAI

from config import GROUPING_TOP_K_SEED, setup_logging
from generate_supernodes import (
    _feature_id,
    apply_phase1_output,
    apply_phase2_outputs,
    apply_phase3_actions,
    build_output_context,
    load_features,
    run_phase1,
    run_phase2_batches,
    run_phase3,
    write_supernodes_into_graph,
)

log = setup_logging()


# ---------------------------------------------------------------------------
# Reuse of an existing grouping as the Phase 1 seed
# ---------------------------------------------------------------------------

def existing_seed(
    graph: dict, features: list[dict], top_k_seed: int
) -> tuple[dict[str, str], dict[str, str]] | None:
    """Recover a Phase 1 seed from a graph that has already been grouped.

    Returns (active_groups, assignments), or None if any of the top-K features is
    missing a `group` — in which case Phase 1 has to run for real.

    Only group *names* are persisted by write_supernodes_into_graph, so rationales
    come from graph["supernode_rationales"] when present and are left empty
    otherwise. Phase 2 sees the names either way.
    """
    seed_feats = features[:top_k_seed]
    if not seed_feats:
        return None

    assignments: dict[str, str] = {}
    for f in seed_feats:
        gname = f["_neuron"].get("group")
        if not gname:
            return None
        assignments[f["id"]] = gname

    stored = graph.get("supernode_rationales") or {}
    active_groups = {
        g: stored.get(g, "") for g in set(assignments.values()) if g != "Ungrouped"
    }
    return active_groups, assignments


# ---------------------------------------------------------------------------
# One arm
# ---------------------------------------------------------------------------

def _counts(grouped_graph: dict) -> tuple[int, int, int]:
    """(grouped, ungrouped, total) for a graph that has been written back."""
    grouped = sum(len(ids) for ids in grouped_graph.get("supernodes", {}).values())
    ungrouped = len(grouped_graph.get("ungrouped", []))
    return grouped, ungrouped, grouped + ungrouped


def _truncate(graph: dict, keep: list[dict]) -> None:
    """Drop every neuron outside `keep` (the arm's top-N), preserving file order."""
    keep_ids = {f["id"] for f in keep}
    graph["neurons"] = [n for n in graph["neurons"] if _feature_id(n) in keep_ids]


async def run_arm(
    client: AsyncOpenAI,
    graph: dict,
    n: int,
    top_k_seed: int,
    seed_groups: dict[str, str],
    seed_assignments: dict[str, str],
    prompt_text: str,
    output_context: str,
    out_path: Path,
) -> tuple[int, int, int]:
    """Group one graph at budget N and write it to out_path. Returns the counts."""
    arm_graph = copy.deepcopy(graph)
    arm_feats = load_features(arm_graph)[:n]  # rebind _neuron refs to the copy

    active_groups = dict(seed_groups)
    final_assignments = {
        fid: g for fid, g in seed_assignments.items() if fid in {f["id"] for f in arm_feats}
    }

    # Phase 2 — the only thing that varies across arms.
    remaining = arm_feats[top_k_seed:]
    p2 = await run_phase2_batches(
        client, remaining, active_groups, prompt_text, output_context
    )
    apply_phase2_outputs(p2, active_groups, final_assignments)

    # Phase 3 — reconcile over exactly this arm's features.
    p3 = await run_phase3(client, final_assignments, arm_feats, prompt_text, output_context)
    apply_phase3_actions(p3, active_groups, final_assignments)

    write_supernodes_into_graph(arm_graph, arm_feats, final_assignments)
    _truncate(arm_graph, arm_feats)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(arm_graph, indent=2, ensure_ascii=False), encoding="utf-8")
    return _counts(arm_graph)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

async def process_graph(
    client: AsyncOpenAI,
    path: Path,
    arms: list[int],
    top_k_seed: int,
    out_dir: Path,
    force: bool,
    stats: dict[int, list[tuple[int, int, int]]],
) -> None:
    """Run every arm for one graph, sharing a single Phase 1 seed across them."""
    graph = json.loads(path.read_text(encoding="utf-8"))
    prompt_text = graph.get("prompt") or "Unknown prompt"
    features = load_features(graph)
    if not features:
        log.warning("No described features in %s — skipping.", path.name)
        return
    log.info("=== %s — %d described features — '%s' ===", path.name, len(features), prompt_text)

    # Which arms still need computing?
    todo: list[int] = []
    for n in arms:
        out_path = out_dir / f"n{n}" / path.name
        if not force and out_path.exists():
            try:
                done = json.loads(out_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                done = {}
            if "supernodes" in done and "ungrouped" in done:
                stats[n].append(_counts(done))
                log.info("  n=%d — reusing %s", n, out_path)
                continue
        todo.append(n)
    if not todo:
        return

    output_context = build_output_context(graph, features)

    # Phase 1 — shared across arms, and skipped entirely when the source graph has
    # already been grouped (its top-K assignments become the seed).
    seed = existing_seed(graph, features, top_k_seed)
    if seed is not None:
        seed_groups, seed_assignments = seed
        log.info("  Phase 1 skipped — reusing %d existing groups from source.", len(seed_groups))
    else:
        seed_groups, seed_assignments = {}, {}
        p1 = await run_phase1(client, features[:top_k_seed], prompt_text, output_context)
        if p1 is None:
            log.error("Phase 1 failed for %s — skipping.", path.name)
            return
        apply_phase1_output(p1, seed_groups, seed_assignments)

    for n in todo:
        counts = await run_arm(
            client, graph, n, top_k_seed, seed_groups, seed_assignments,
            prompt_text, output_context, out_dir / f"n{n}" / path.name,
        )
        stats[n].append(counts)
        log.info("  n=%d — %d grouped, %d ungrouped, %d total", n, *counts)


def append_report(
    report: Path, n: int, rows: list[tuple[int, int, int]], corpus: str
) -> None:
    """Append one `-n` section to the report .md."""
    if not rows:
        return
    n_graphs = len(rows)
    mean_g = sum(r[0] for r in rows) / n_graphs
    mean_u = sum(r[1] for r in rows) / n_graphs
    mean_t = sum(r[2] for r in rows) / n_graphs
    total_t = sum(r[2] for r in rows)
    frac = sum(r[0] for r in rows) / total_t if total_t else 0.0

    section = (
        f"-n count: {n}\n\n"
        "| Method | Grouped | Ungrouped | Total | Fraction grouped |\n"
        "| --- | ---: | ---: | ---: | ---: |\n"
        f"| n={n} | {mean_g:.1f} | {mean_u:.1f} | {mean_t:.1f} | {frac:.3f} |\n\n"
        f"Counts are means per graph over the {n_graphs} {corpus} graphs; the fraction\n"
        "grouped is pooled across all graphs.\n\n"
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    with open(report, "a", encoding="utf-8") as f:
        f.write(section)
    log.info("Appended -n %d section to %s", n, report)


async def main_async(args: argparse.Namespace, graphs: list[Path]) -> None:
    client = AsyncOpenAI()
    stats: dict[int, list[tuple[int, int, int]]] = {n: [] for n in args.n}
    for p in graphs:
        await process_graph(
            client, p, args.n, args.top_k_seed, args.out_dir, args.force, stats
        )
    for n in args.n:
        append_report(args.report, n, stats[n], args.corpus_name)


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        log.error("OPENAI_API_KEY not set.")
        sys.exit(1)

    ap = argparse.ArgumentParser(description="Sweep supernode grouping over the feature budget N.")
    ap.add_argument("--graphs-dir", type=Path, required=True, help="Folder of grouped graph_*.json.")
    ap.add_argument(
        "-n", type=int, action="append", required=True, metavar="N",
        help="A feature budget to sweep. Repeat: -n 50 -n 100 -n 150.",
    )
    ap.add_argument("--out-dir", type=Path, required=True, help="Arm outputs go to <out-dir>/n<N>/.")
    ap.add_argument("--report", type=Path, required=True, help="Markdown report to append to.")
    ap.add_argument(
        "--top-k-seed", type=int, default=GROUPING_TOP_K_SEED,
        help="Features seeding Phase 1, shared by all arms (default: %(default)s).",
    )
    ap.add_argument("--corpus-name", default="Capitals", help="Corpus name for the report caption.")
    ap.add_argument("--force", action="store_true", help="Recompute arms that already have output.")
    args = ap.parse_args()

    if args.top_k_seed <= 0:
        log.error("--top-k-seed must be a positive integer (got %d).", args.top_k_seed)
        sys.exit(1)

    graphs = sorted(args.graphs_dir.glob("graph_*.json"))
    if not graphs:
        log.error("No graph_*.json found in %s.", args.graphs_dir)
        sys.exit(1)

    asyncio.run(main_async(args, graphs))


if __name__ == "__main__":
    main()

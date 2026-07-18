"""
compare_mlp_validation.py — same-method control (false-positive floor) for compare_features_llm.py.

Calibrates the feature↔supernode judge by comparing MLP supernodes against the SAME MLP neurons,
re-described independently. Every supernode is built from those neurons, so a correct method
matches ~100%; anything reported "missing" is the judge failing to re-match a concept to its own
features under independent wording — the same difficulty as the real MLP↔SLT comparison. Whatever
floor this measures is the noise you subtract from the real "missing" counts.

It reuses compare_features_llm.py's helpers by import (same filter, judge, prompts) so this is a
faithful control, not a re-implementation that could drift.

  Source = original MLP graphs (--mlp-dir): supernodes + their members' ORIGINAL descriptions.
  Target = regenerated MLP graphs (--regen-dir): the SAME neurons, freshly described. Produce with:
      cp -r capital_neuron_graphs capital_neuron_graphs_regen
      generate_description.py --graphs-dir capital_neuron_graphs_regen --exemplars <...> --force

Direction: MLP supernodes → all ~150 regenerated MLP features (mirrors the real run's shape).
Filter: ON (same meaningfulness filter as the real run) so the floor is measured over the same
population the real comparison judges.

Usage:
    OPENAI_API_KEY=... uv run python custom_automation/compare_mlp_validation.py \
        --mlp-dir capital_neuron_graphs \
        --regen-dir capital_neuron_graphs_regen \
        --out-dir custom_automation/validation_out --concurrency 100
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

# Import the EXACT method used by the real comparison, so this control cannot drift from it.
from compare_features_llm import (
    JUDGE_MODEL,
    _compare_direction,
    load_mlp_graph,
    slug_from_mlp_path,
)


def write_reports(reports: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validation_mlp.json").write_text(
        json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")

    # Terminal-style false-positive lines (missing == FP), printed AND saved verbatim.
    fp_lines: list[str] = []
    for r in reports:
        for name in r["false_positives"]:
            fp_lines.append(
                f'  FALSE POSITIVE: MLP supernode "{name}" not matched to its own '
                f'regenerated features  [{r["slug"]}]')
    for line in fp_lines:
        print(line)
    (out_dir / "terminal_validation.md").write_text("\n".join(fp_lines) + "\n", encoding="utf-8")

    total_compared = sum(r["n_compared"] for r in reports)
    total_fp = sum(len(r["false_positives"]) for r in reports)
    rate = total_fp / total_compared if total_compared else 0.0

    L = ["# MLP→MLP validation — false-positive floor of the comparison", "",
         "Control for `compare_features_llm.py`: original MLP supernodes vs. the SAME neurons "
         "re-described independently. A `missing` here is a FALSE POSITIVE — the judge failed to "
         "re-match a supernode to its own features. Subtract this floor from the real MLP↔SLT "
         "'missing' counts. (Identical method as the real run — helpers imported, not copied.)", "",
         f"- Graphs: **{len(reports)}**",
         f"- Supernodes compared (meaningful, filter on): **{total_compared}**",
         f"- False positives (missing): **{total_fp}**",
         f"- **False-positive rate: {rate:.1%}**   (self-recovery = {1 - rate:.1%})", "",
         "| slug | compared | false positives |",
         "|---|---:|---|"]
    for r in reports:
        fp = "; ".join(r["false_positives"]) or "—"
        L.append(f"| {r['slug']} | {r['n_compared']} | {fp} |")

    L += ["", "## Per-graph detail", ""]
    for r in reports:
        L.append(f"### {r['slug']}")
        L.append("**MLP supernode → matched regenerated MLP features:**")
        for name, v in r["matches"].items():
            feats = v["matched_features"]
            if feats:
                L.append(f'  - "{name}" → {"; ".join(feats)}')
            else:
                L.append(f'  - "{name}" → **FALSE POSITIVE** ({v["reason"] or "no match"})')
        if r["filtered_out"]:
            L.append(f'  - _(filtered as non-meaningful: {", ".join(r["filtered_out"])})_')
        L.append("")
    (out_dir / "validation_mlp.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nWrote validation_mlp.json, .md, and terminal_validation.md ({len(reports)} graphs)")
    print(f"False-positive floor: {total_fp}/{total_compared} = {rate:.1%}  "
          f"(self-recovery {1 - rate:.1%})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mlp-dir", type=Path, required=True,
                    help="Original MLP graph_*.json (SOURCE: supernodes + original descriptions).")
    ap.add_argument("--regen-dir", type=Path, required=True,
                    help="MLP graph_*.json with REGENERATED descriptions (TARGET: the same neurons).")
    ap.add_argument("--out-dir", type=Path, default=Path("custom_automation/validation_out"))
    ap.add_argument("--slugs", nargs="*", help="Optional subset (default: all in --mlp-dir).")
    ap.add_argument("--concurrency", type=int, default=50,
                    help="Max in-flight gpt-5.4 calls. Tier-5 accounts can push this to 100+.")
    args = ap.parse_args()
    asyncio.run(_amain(args))


def _index(d: Path) -> dict[str, Path]:
    paths = {slug_from_mlp_path(p): p for p in sorted(d.glob("graph_*.json"))}
    return paths or {slug_from_mlp_path(p): p for p in sorted(d.glob("*.json"))}


async def _amain(args) -> None:
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)

    src_paths = _index(args.mlp_dir)
    regen_paths = _index(args.regen_dir)
    slugs = args.slugs or sorted(src_paths)

    entries: list[dict] = []
    for slug in slugs:
        if slug not in src_paths:
            print(f"  [skip {slug}] no MLP graph in {args.mlp_dir}")
            continue
        if slug not in regen_paths:
            print(f"  [skip {slug}] no regenerated graph in {args.regen_dir}")
            continue
        entries.append({"slug": slug,
                        "src": load_mlp_graph(src_paths[slug]),
                        "regen": load_mlp_graph(regen_paths[slug])})
    if not entries:
        write_reports([], args.out_dir)
        return

    async def run_entry(e: dict) -> dict:
        # Original supernodes (source) vs the SAME neurons re-described (target features).
        # NEUTRAL method labels ("Method A"/"Method B") in the prompt: if we told the judge both
        # sides are "MLP" it could shortcut "same method -> match", biasing the floor optimistically.
        # Neutral labels reproduce the real run's cross-method framing without that leak.
        matches, filtered = await _compare_direction(
            client, sem, e["src"], e["regen"], "Method A", "Method B")
        fps = [n for n, v in matches.items() if not v["matched_features"]]
        return {"slug": e["slug"], "matches": matches, "filtered_out": filtered,
                "false_positives": fps, "n_compared": len(matches)}

    print(f"Validating {len(entries)} graphs (MLP→MLP self-match, filter + judge, "
          f"concurrency={args.concurrency}, model={JUDGE_MODEL})...")
    reports = await asyncio.gather(*[run_entry(e) for e in entries])
    write_reports(list(reports), args.out_dir)


if __name__ == "__main__":
    main()

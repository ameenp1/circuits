"""
compare_features_llm.py — LLM-judge feature↔supernode correspondence, MLP vs SLT.

cross_graph_analysis.py matches supernode NAMES (embeddings). This asks a stricter, more
honest question with an LLM judge:

    For each supernode found by ONE method, does the OTHER method contain individual
    features that represent the SAME underlying concept?

Both directions. We deliberately compare a method-A *supernode* against method-B's flat list
of *individual* features (not B's supernodes): a supernode is missing in B only if NONE of B's
individual features carry the concept — so a reported gap is a real missing-feature difference,
not an artifact of how the two methods happened to label / cluster their groups.

Prints (and writes) the gaps each direction:
    MLP supernode missing in SLT: "<name>"      (no SLT feature represents it)
    SLT supernode missing in MLP: "<name>"

MLP  = raw down_proj neurons, circuits/ graphs (graph_*.json, --mlp-dir).
SLT  = single-layer transcoder features. Needs ONLY the Neuronpedia test_graphs dir (<slug>.json):
       the supernode grouping is in qParams.supernodes, and each described feature's
       generated_description is stored as its node `clerp`. No separate feature_descriptions or
       feature_groups files required.

The MLP loader + the Graph/Feature model are reused from cross_graph_analysis.py; the SLT graph
is built by load_slt_local (below), so "which side is which" stays defined in one place.

Usage (on the box — test_graphs is all you need for the SLT side):
    OPENAI_API_KEY=... uv run python custom_automation/compare_features_llm.py \
        --mlp-dir capital_neuron_graphs \
        --slt-graphs-dir test_graphs \
        --out-dir custom_automation/compare_out
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

from cross_graph_analysis import (
    Feature,
    Graph,
    load_mlp_graph,
    slug_from_mlp_path,
)


def load_slt_local(tg_path: Path, desc_path: Path | None = None) -> Graph:
    """Build the SLT (transcoder) Graph from a Neuronpedia test_graph ALONE — no separate
    feature_groups or feature_descriptions artifact needed:
      - supernode grouping is embedded in the graph's `qParams.supernodes`
        (a JSON string of [group_name, member_id, ...]);
      - each described feature's generated_description is stored on its node as `clerp`
        (only the described features carry a clerp — exactly the feature_descriptions set),
        and the node_id == the description id == the supernode member id, so it all lines up.
    If `desc_path` is given, that feature_descriptions file is used instead (identical text,
    plus promotes/demotes); otherwise the descriptions are read from the graph's node clerps."""
    tg = json.loads(tg_path.read_text(encoding="utf-8"))
    meta = tg.get("metadata", {})

    groups: dict[str, str] = {}
    raw = (tg.get("qParams") or {}).get("supernodes")
    if raw:
        for grp in json.loads(raw):
            if not grp:
                continue
            name, members = grp[0], grp[1:]
            for mid in members:
                groups[str(mid)] = name

    target = ""
    for n in tg.get("nodes", []):
        if n.get("is_target_logit") or n.get("feature_type") == "logit":
            target = (n.get("clerp") or n.get("token") or "").strip() or target
            if n.get("is_target_logit"):
                break

    feats: list[Feature] = []
    if desc_path and desc_path.exists():
        for e in json.loads(desc_path.read_text(encoding="utf-8")):
            desc = (e.get("generated_description") or "").strip()
            if not desc:
                continue
            fid = str(e["id"])
            feats.append(Feature(
                fid=fid, layer=int(e.get("layer", 0)), ctx=int(e.get("ctx_idx", 0)),
                score=float(e.get("influence_score", 0.0)), desc=desc,
                group=groups.get(fid, "Ungrouped"),
                promotes=list(e.get("promotes") or []), demotes=list(e.get("demotes") or []),
            ))
    else:
        # descriptions live in the graph itself, as the clerp of each described transcoder node
        for n in tg.get("nodes", []):
            if n.get("feature_type") != "cross layer transcoder":
                continue
            desc = (n.get("clerp") or "").strip()
            if not desc:                       # undescribed nodes have empty clerp -> skip
                continue
            fid = str(n.get("node_id", ""))
            feats.append(Feature(
                fid=fid, layer=int(n.get("layer", 0) or 0), ctx=int(n.get("ctx_idx", 0) or 0),
                score=float(n.get("influence", 0.0) or 0.0), desc=desc,
                group=groups.get(fid, "Ungrouped"), promotes=[], demotes=[],
            ))
    return Graph(
        side="transcoder", slug=meta.get("slug", tg_path.stem), prompt=meta.get("prompt", ""),
        target=target, features=feats, node_type_counts={}, n_token_nodes=len(tg.get("nodes", [])),
        n_edges=len(tg.get("links", [])), n_tokens=len(meta.get("prompt_tokens") or []),
    )

# The interesting-graphs judge model (custom_automation/analysis/explore_interesting_graphs.py).
JUDGE_MODEL = "gpt-5.4"

SYSTEM = (
    "You compare two decompositions of the SAME language-model computation on the SAME prompt. "
    "One method ('source') grouped its features into named supernodes (concepts). The other "
    "('target') gives a flat list of individual features, each with a description. "
    "For every source supernode, decide whether the target contains one or more individual "
    "features that represent the SAME underlying concept — the same thing the model is tracking "
    "(e.g. 'the state Alabama', 'capital-city relation', 'the token Birmingham'), not merely "
    "similar wording or the same broad topic. A partial match by a single target feature counts. "
    "Be strict: if nothing in the target genuinely carries the concept, report no match."
)

USER_TEMPLATE = """Prompt: {prompt}
Model's answer: {target}

SOURCE supernode ({source_method}): "{supernode}"
Its member features:
{members}

TARGET individual features ({target_method}), numbered:
{features}

List the numbers of the TARGET features that represent the SAME underlying concept as this
source supernode, or an empty list if none do.

Return ONLY JSON, no prose:
{{"feature_numbers": [<int>, ...], "reason": "<short>"}}
"""


def _dedup_descs(items: list[str]) -> list[str]:
    """Order-preserving de-dup of feature descriptions. NO cap: with low pruning the graphs are
    small, and every feature — including ungrouped ones — must be a candidate for the judge."""
    seen: set[str] = set()
    out: list[str] = []
    for d in items:
        d = (d or "").strip()
        if not d or d in seen:
            continue
        seen.add(d)
        out.append(d)
    return out


def _direction_plan(source: Graph, target: Graph, source_method: str, target_method: str
                    ) -> tuple[list[str], list[tuple[str, str]]]:
    """Build the per-supernode judge prompts for one direction, without calling the LLM.

    Returns (target_feats, [(supernode_name, user_prompt), ...]). `target.features` is the FULL
    feature list from the loader — grouped AND ungrouped — so an ungrouped feature on the other
    side can satisfy a supernode. Empty job list if there is nothing to compare."""
    supernodes = source.concept_supernodes()
    target_feats = _dedup_descs([f.desc for f in target.features])
    if not supernodes or not target_feats:
        return target_feats, []
    feat_block = "\n".join(f"{i}. {d}" for i, d in enumerate(target_feats))
    jobs: list[tuple[str, str]] = []
    for name, members in supernodes.items():
        member_block = "\n".join(f"  - {d}" for d in _dedup_descs([f.desc for f in members]))
        user = USER_TEMPLATE.format(
            prompt=source.prompt or target.prompt, target=source.target or target.target,
            source_method=source_method, target_method=target_method,
            supernode=name, members=member_block, features=feat_block,
        )
        jobs.append((name, user))
    return target_feats, jobs


async def _judge_one(client, sem, user: str, target_feats: list[str]) -> dict:
    """One judge call for one supernode, bounded by the semaphore. Never raises: a failed or
    unparseable response degrades to 'no match' so one bad call can't abort the whole gather."""
    try:
        async with sem:
            resp = await client.chat.completions.create(
                model=JUDGE_MODEL,
                # gpt-5.4 spends reasoning tokens against this budget; keep it well above the tiny
                # JSON answer so a long feature list never truncates the response.
                max_completion_tokens=16384,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            )
    except Exception as e:  # noqa: BLE001 — surface as a reason, keep the run alive
        return {"matched_features": [], "reason": f"judge error: {e}"}
    text = (resp.choices[0].message.content or "").strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {"matched_features": [], "reason": "no JSON in judge response"}
    try:
        entry = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"matched_features": [], "reason": "unparseable JSON"}
    nums = [n for n in entry.get("feature_numbers", [])
            if isinstance(n, int) and 0 <= n < len(target_feats)]
    return {
        "matched_features": [target_feats[n] for n in nums],
        "reason": str(entry.get("reason", "")).strip(),
    }


def write_reports(reports: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "compare_features_llm.json").write_text(
        json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")

    L = ["# LLM-judge feature↔supernode comparison (MLP vs SLT)", "",
         "One judge call per supernode, each shown ALL of the other method's individual features. "
         "A gap = a real missing feature, not a labeling artifact.", "",
         "| slug | MLP supernodes missing in SLT features | SLT supernodes missing in MLP features |",
         "|---|---|---|"]
    for r in reports:
        mm = "; ".join(r["mlp_supernode_missing_in_slt"]) or "—"
        sm = "; ".join(r["slt_supernode_missing_in_mlp"]) or "—"
        L.append(f"| {r['slug']} | {mm} | {sm} |")
    n_mlp = sum(len(r["mlp_supernode_missing_in_slt"]) for r in reports)
    n_slt = sum(len(r["slt_supernode_missing_in_mlp"]) for r in reports)
    L += ["", f"**Totals:** {n_mlp} MLP supernodes missing in SLT features, "
          f"{n_slt} SLT supernodes missing in MLP features, across {len(reports)} graphs.", ""]

    def _detail(matches: dict[str, dict]) -> list[str]:
        lines = []
        for name, v in matches.items():
            feats = v["matched_features"]
            if feats:
                lines.append(f'  - "{name}" → {"; ".join(feats)}')
            else:
                lines.append(f'  - "{name}" → **MISSING** ({v["reason"] or "no match"})')
        return lines or ["  - (none)"]

    L += ["## Per-graph detail", ""]
    for r in reports:
        L.append(f"### {r['slug']}")
        L.append("**MLP supernode → matched SLT features:**")
        L += _detail(r["mlp_to_slt"])
        L.append("")
        L.append("**SLT supernode → matched MLP features:**")
        L += _detail(r["slt_to_mlp"])
        L.append("")
    (out_dir / "compare_features_llm.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nWrote {out_dir/'compare_features_llm.json'} and .md ({len(reports)} graphs)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mlp-dir", type=Path, required=True, help="Dir of MLP graph_*.json (circuits/).")
    ap.add_argument("--slt-graphs-dir", type=Path, required=True,
                    help="Dir of SLT Neuronpedia test_graphs (<slug>.json). Grouping AND descriptions "
                         "are read from here (qParams.supernodes + node clerps) — this is all you need.")
    ap.add_argument("--slt-desc-dir", type=Path, default=None,
                    help="Optional. Separate feature_descriptions dir (<slug>.json). Only needed if "
                         "you want to override the descriptions embedded in the test_graphs.")
    ap.add_argument("--out-dir", type=Path, default=Path("custom_automation/compare_out"))
    ap.add_argument("--slugs", nargs="*", help="Optional subset of slugs (default: all in --mlp-dir).")
    ap.add_argument("--concurrency", type=int, default=50,
                    help="Max in-flight judge calls. Tier-5 accounts can push this to 100+.")
    args = ap.parse_args()
    asyncio.run(_amain(args))


async def _amain(args) -> None:
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)

    mlp_paths = {slug_from_mlp_path(p): p for p in sorted(args.mlp_dir.glob("graph_*.json"))}
    if not mlp_paths:
        mlp_paths = {slug_from_mlp_path(p): p for p in sorted(args.mlp_dir.glob("*.json"))}
    slugs = args.slugs or sorted(mlp_paths)

    # Plan: load every graph (fast, local) and build all judge prompts up front, so the whole
    # batch — every supernode, both directions, every slug — can fire concurrently in one gather.
    plan: dict[str, dict] = {}          # slug -> per-graph accumulator (results filled in below)
    coros, metas = [], []               # parallel lists: coros[i] fills metas[i] = (slug, dir, name)
    for slug in slugs:
        if slug not in mlp_paths:
            print(f"  [skip {slug}] no MLP graph in {args.mlp_dir}")
            continue
        tg_p = args.slt_graphs_dir / f"{slug}.json"
        if not tg_p.exists():
            print(f"  [skip {slug}] missing SLT graph {tg_p}")
            continue
        de_p = (args.slt_desc_dir / f"{slug}.json") if args.slt_desc_dir else None
        mlp = load_mlp_graph(mlp_paths[slug])
        slt = load_slt_local(tg_p, de_p)
        plan[slug] = {
            "prompt": mlp.prompt,
            "mlp_supernodes": list(mlp.concept_supernodes().keys()),
            "slt_supernodes": list(slt.concept_supernodes().keys()),
            "mlp_to_slt": {}, "slt_to_mlp": {},
        }
        for dkey, (src, tgt, sm, tm) in (
            ("mlp_to_slt", (mlp, slt, "MLP neurons", "SLT features")),
            ("slt_to_mlp", (slt, mlp, "SLT features", "MLP neurons")),
        ):
            target_feats, jobs = _direction_plan(src, tgt, sm, tm)
            for name, user in jobs:
                coros.append(_judge_one(client, sem, user, target_feats))
                metas.append((slug, dkey, name))

    if not plan:
        write_reports([], args.out_dir)
        return

    print(f"Dispatching {len(coros)} judge calls across {len(plan)} graphs "
          f"(concurrency={args.concurrency}, model={JUDGE_MODEL})...")
    results = await asyncio.gather(*coros)          # order matches `metas`
    for (slug, dkey, name), res in zip(metas, results):
        plan[slug][dkey][name] = res

    reports = []
    for slug, e in plan.items():
        mlp_missing = [n for n, v in e["mlp_to_slt"].items() if not v["matched_features"]]
        slt_missing = [n for n, v in e["slt_to_mlp"].items() if not v["matched_features"]]
        for name in mlp_missing:
            print(f'  MLP supernode missing in SLT features: "{name}"  [{slug}]')
        for name in slt_missing:
            print(f'  SLT supernode missing in MLP features: "{name}"  [{slug}]')
        reports.append({
            "slug": slug, "prompt": e["prompt"],
            "mlp_supernodes": e["mlp_supernodes"], "slt_supernodes": e["slt_supernodes"],
            "mlp_supernode_missing_in_slt": mlp_missing,
            "slt_supernode_missing_in_mlp": slt_missing,
            "mlp_to_slt": e["mlp_to_slt"], "slt_to_mlp": e["slt_to_mlp"],
        })

    write_reports(reports, args.out_dir)


if __name__ == "__main__":
    main()

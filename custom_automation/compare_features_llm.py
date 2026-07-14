"""
compare_features_llm.py — LLM-judge feature↔supernode correspondence, MLP vs SLT.

The question: for each supernode found by ONE method, does the OTHER method contain individual
features that represent the SAME underlying concept? Both directions. We compare a method-A
*supernode* against method-B's flat list of *individual* features (not B's supernodes), so a
reported gap is a genuinely missing feature, not an artifact of how the two methods labelled or
clustered their groups.

Two refinements:
  1. MEANINGFULNESS FILTER (per direction, on the SOURCE side only). Before comparing, gpt-5.4
     drops supernodes that are entirely generic/grammatical (e.g. "capital city", "economic
     capital", "say place name", "containment", "content noun") — only SPECIFIC labels (proper
     nouns, nationalities, named concepts) are worth comparing. A label keeps its slot if it
     contains ANY specific element (e.g. "suppress Texas"); it is dropped only if fully generic.
     The target feature list is never filtered.
  2. GROUPING FRACTION. We also report the fraction of each method's features that land in a
     concept supernode vs. stay Ungrouped — MLP tends to group a smaller share, one concrete
     sense in which it is harder to interpret.

Prints (and writes) the gaps each direction:
    MLP supernode missing in SLT features: "<name>"
    SLT supernode missing in MLP features: "<name>"

MLP  = raw down_proj neurons, circuits/ graphs (graph_*.json, --mlp-dir).
SLT  = single-layer transcoder features. Needs ONLY the Neuronpedia test_graphs dir (<slug>.json):
       supernode grouping is in qParams.supernodes, and each described feature's
       generated_description is its node `clerp`. No separate feature_descriptions/groups files.

Usage (on the box — test_graphs is all you need for the SLT side):
    OPENAI_API_KEY=... uv run python custom_automation/compare_features_llm.py \
        --mlp-dir capital_neuron_graphs \
        --slt-graphs-dir test_graphs \
        --out-dir custom_automation/compare_out \
        --concurrency 100
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


# ===========================================================================
# Graph / Feature model + loaders
# ===========================================================================

@dataclass
class Feature:
    fid: str            # id within its own graph
    layer: int
    ctx: int            # token position
    score: float        # |attribution| (MLP) or influence (transcoder)
    desc: str           # generated_description (the matching key)
    group: str          # supernode name, or "Ungrouped"
    promotes: list[str] = field(default_factory=list)
    demotes: list[str] = field(default_factory=list)


@dataclass
class Graph:
    side: str           # "mlp" | "transcoder"
    slug: str
    prompt: str
    target: str
    features: list[Feature]

    def concept_supernodes(self) -> dict[str, list[Feature]]:
        """name -> member features, concept groups only (drops Ungrouped / Emb / Output)."""
        out: dict[str, list[Feature]] = {}
        for f in self.features:
            if is_concept_group(f.group):
                out.setdefault(f.group, []).append(f)
        return out

    def grouped_counts(self) -> tuple[int, int]:
        """(#features in a concept supernode, #Ungrouped) over described features."""
        grouped = sum(1 for f in self.features if is_concept_group(f.group))
        return grouped, len(self.features) - grouped


def is_concept_group(name: str | None) -> bool:
    if not name or name == "Ungrouped":
        return False
    if name.startswith(("Emb:", "Output:")):
        return False
    return True


def _split_signed(contribs) -> tuple[list[str], list[str]]:
    """ADAG output_contributions [[tok, signed]] -> (promoted, suppressed) token lists."""
    pos, neg = [], []
    for item in contribs or []:
        try:
            tok, score = item[0], float(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        (pos if score >= 0 else neg).append(tok)
    return pos, neg


def slug_from_mlp_path(path: Path) -> str:
    g = json.loads(path.read_text(encoding="utf-8"))
    return (g.get("label") or g.get("slug") or path.stem).split("___")[0]


def load_mlp_graph(path: Path) -> Graph:
    g = json.loads(Path(path).read_text(encoding="utf-8"))
    slug = (g.get("label") or g.get("slug") or path.stem).split("___")[0]
    feats: list[Feature] = []
    for n in g.get("neurons", []):
        desc = (n.get("generated_description") or "").strip()
        if not desc or desc == "Error generating description":
            continue
        pol = n.get("polarity", "")
        fid = f"L{n['layer']}_N{n['neuron']}{('_' + pol) if pol else ''}"
        promotes, demotes = _split_signed(n.get("output_contributions"))
        feats.append(Feature(
            fid=fid, layer=int(n["layer"]), ctx=int(n.get("token", 0) or 0),
            score=abs(float(n.get("attribution", 0.0))), desc=desc,
            group=n.get("group", "Ungrouped"), promotes=promotes, demotes=demotes,
        ))
    return Graph(side="mlp", slug=slug, prompt=g.get("prompt", ""),
                 target=g.get("target", ""), features=feats)


def load_slt_local(tg_path: Path, desc_path: Path | None = None) -> Graph:
    """Build the SLT (transcoder) Graph from a Neuronpedia test_graph ALONE — no separate
    feature_groups or feature_descriptions artifact needed:
      - supernode grouping is embedded in the graph's `qParams.supernodes`;
      - each described feature's generated_description is stored on its node as `clerp`
        (only described features carry a clerp), and node_id == description id == supernode
        member id, so it all lines up.
    If `desc_path` is given, that feature_descriptions file is used instead (identical text)."""
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
    return Graph(side="transcoder", slug=meta.get("slug", tg_path.stem),
                 prompt=meta.get("prompt", ""), target=target, features=feats)


# ===========================================================================
# LLM prompts
# ===========================================================================

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

FILTER_SYSTEM = (
    "You filter supernode labels from a model's attribution graph, keeping only the ones specific "
    "enough to be worth comparing across two interpretability methods. KEEP a label if it names "
    "something specific: a proper noun (place, country, nationality, person, organization), a "
    "specific script/language, or a concrete named concept. DROP a label only if it is ENTIRELY "
    "generic or grammatical — a generic category (e.g. 'capital city', 'economic capital', 'city "
    "name', 'US state'), a parsing/positional role (e.g. 'say X', 'suppress X', 'containment', "
    "'content noun', 'proper noun'), or just articles/prepositions/stopwords. If a label contains "
    "ANY specific element (e.g. 'suppress Texas', 'say Karnataka district'), KEEP it — drop only "
    "when it is fully generic. When unsure, keep."
)

FILTER_USER = """Supernode label: "{label}"

Is this label specific enough to be worth comparing (KEEP), or entirely generic/grammatical (DROP)?

Return ONLY JSON, no prose: {{"keep": true or false}}
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


# ===========================================================================
# LLM calls
# ===========================================================================

async def _chat(client, sem, system: str, user: str) -> str:
    """One bounded gpt-5.4 call -> raw text. Never raises (errors surface as '')."""
    try:
        async with sem:
            resp = await client.chat.completions.create(
                model=JUDGE_MODEL,
                # gpt-5.4 spends reasoning tokens against this budget; keep it well above the tiny
                # JSON answer so a long feature list never truncates the response.
                max_completion_tokens=16384,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
        return (resp.choices[0].message.content or "").strip()
    except Exception:  # noqa: BLE001 — surface as empty, keep the run alive
        return ""


async def _is_meaningful(client, sem, name: str) -> bool:
    """ONE gpt-5.4 call for ONE supernode label -> keep (specific) or drop (fully generic).
    On any failure, KEEP (never silently drop a real comparison)."""
    text = await _chat(client, sem, FILTER_SYSTEM, FILTER_USER.format(label=name))
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return True
    try:
        return bool(json.loads(m.group(0)).get("keep", True))
    except json.JSONDecodeError:
        return True


async def _judge_one(client, sem, user: str, target_feats: list[str]) -> dict:
    """Which target features match one source supernode. Degrades to 'no match' on failure."""
    text = await _chat(client, sem, SYSTEM, user)
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


async def _compare_direction(client, sem, source: Graph, target: Graph,
                             source_method: str, target_method: str) -> tuple[dict, list[str]]:
    """Filter the SOURCE supernodes to the meaningful ones, then (in parallel) ask, for each,
    which TARGET features carry the same concept. Returns (matches, filtered_out_names)."""
    supernodes = source.concept_supernodes()
    names = list(supernodes.keys())
    if not names:
        return {}, []
    # Meaningfulness filter: ONE call per supernode (1-by-1), all in parallel.
    flags = await asyncio.gather(*[_is_meaningful(client, sem, n) for n in names])
    kept = [n for n, keep in zip(names, flags) if keep]
    filtered_out = [n for n, keep in zip(names, flags) if not keep]

    target_feats = _dedup_descs([f.desc for f in target.features])
    if not target_feats or not kept:
        return {}, filtered_out
    feat_block = "\n".join(f"{i}. {d}" for i, d in enumerate(target_feats))

    async def one(name: str) -> tuple[str, dict]:
        member_block = "\n".join(f"  - {d}" for d in _dedup_descs([f.desc for f in supernodes[name]]))
        user = USER_TEMPLATE.format(
            prompt=source.prompt or target.prompt, target=source.target or target.target,
            source_method=source_method, target_method=target_method,
            supernode=name, members=member_block, features=feat_block,
        )
        return name, await _judge_one(client, sem, user, target_feats)

    pairs = await asyncio.gather(*[one(n) for n in kept])
    return dict(pairs), filtered_out


# ===========================================================================
# Reporting
# ===========================================================================

def write_reports(reports: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "compare_features_llm.json").write_text(
        json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")

    L = ["# LLM-judge feature↔supernode comparison (MLP vs SLT)", "",
         "Only meaningful (specific) supernodes are compared — generic/grammatical labels are "
         "filtered out per direction on the source side. A gap = a real missing feature.", "",
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

    # ---- Grouping fraction: how much of each method lands in a concept supernode ----
    mg = sum(r["mlp_grouped"] for r in reports); mu = sum(r["mlp_ungrouped"] for r in reports)
    sg = sum(r["slt_grouped"] for r in reports); su = sum(r["slt_ungrouped"] for r in reports)
    mfrac = mg / (mg + mu) if (mg + mu) else 0.0
    sfrac = sg / (sg + su) if (sg + su) else 0.0
    L += ["## Grouping fraction (interpretability)",
          "Fraction of described features placed into a concept supernode vs. left Ungrouped. "
          "A lower fraction = more features the method could not group into a meaning — one sense "
          "in which a method is harder to interpret.", "",
          "| method | grouped | ungrouped | fraction grouped |",
          "|---|---:|---:|---:|",
          f"| **MLP** | {mg} | {mu} | {mfrac:.1%} |",
          f"| **SLT** | {sg} | {su} | {sfrac:.1%} |",
          "",
          (f"> MLP groups **{mfrac:.1%}** of its features vs SLT's **{sfrac:.1%}**"
           + (" — MLP leaves more Ungrouped, consistent with it being harder to interpret."
              if mfrac < sfrac else ".")),
          ""]

    def _detail(matches: dict[str, dict], filtered: list[str]) -> list[str]:
        lines = []
        for name, v in matches.items():
            feats = v["matched_features"]
            if feats:
                lines.append(f'  - "{name}" → {"; ".join(feats)}')
            else:
                lines.append(f'  - "{name}" → **MISSING** ({v["reason"] or "no match"})')
        if filtered:
            lines.append(f'  - _(filtered as non-meaningful: {", ".join(filtered)})_')
        return lines or ["  - (none)"]

    L += ["## Per-graph detail", ""]
    for r in reports:
        L.append(f"### {r['slug']}  (MLP {r['mlp_grouped']}/{r['mlp_grouped'] + r['mlp_ungrouped']} grouped)")
        L.append("**MLP supernode → matched SLT features:**")
        L += _detail(r["mlp_to_slt"], r["mlp_filtered_out"])
        L.append("")
        L.append("**SLT supernode → matched MLP features:**")
        L += _detail(r["slt_to_mlp"], r["slt_filtered_out"])
        L.append("")
    (out_dir / "compare_features_llm.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nWrote {out_dir/'compare_features_llm.json'} and .md ({len(reports)} graphs)")
    print(f"Grouping fraction — MLP: {mfrac:.1%} grouped, SLT: {sfrac:.1%} grouped")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mlp-dir", type=Path, required=True, help="Dir of MLP graph_*.json (circuits/).")
    ap.add_argument("--slt-graphs-dir", type=Path, required=True,
                    help="Dir of SLT Neuronpedia test_graphs (<slug>.json). Grouping AND descriptions "
                         "are read from here (qParams.supernodes + node clerps) — this is all you need.")
    ap.add_argument("--slt-desc-dir", type=Path, default=None,
                    help="Optional separate feature_descriptions dir (<slug>.json) to override the "
                         "descriptions embedded in the test_graphs.")
    ap.add_argument("--out-dir", type=Path, default=Path("custom_automation/compare_out"))
    ap.add_argument("--slugs", nargs="*", help="Optional subset of slugs (default: all in --mlp-dir).")
    ap.add_argument("--concurrency", type=int, default=50,
                    help="Max in-flight gpt-5.4 calls. Tier-5 accounts can push this to 100+.")
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

    # Load every graph (fast, local) up front.
    entries: list[dict] = []
    for slug in slugs:
        if slug not in mlp_paths:
            print(f"  [skip {slug}] no MLP graph in {args.mlp_dir}")
            continue
        tg_p = args.slt_graphs_dir / f"{slug}.json"
        if not tg_p.exists():
            print(f"  [skip {slug}] missing SLT graph {tg_p}")
            continue
        de_p = (args.slt_desc_dir / f"{slug}.json") if args.slt_desc_dir else None
        entries.append({"slug": slug,
                        "mlp": load_mlp_graph(mlp_paths[slug]),
                        "slt": load_slt_local(tg_p, de_p)})
    if not entries:
        write_reports([], args.out_dir)
        return

    async def run_entry(e: dict) -> dict:
        mlp, slt = e["mlp"], e["slt"]
        (mlp_to_slt, mlp_filtered), (slt_to_mlp, slt_filtered) = await asyncio.gather(
            _compare_direction(client, sem, mlp, slt, "MLP neurons", "SLT features"),
            _compare_direction(client, sem, slt, mlp, "SLT features", "MLP neurons"),
        )
        mg, mu = mlp.grouped_counts()
        sg, su = slt.grouped_counts()
        return {
            "slug": e["slug"], "prompt": mlp.prompt,
            "mlp_supernodes": list(mlp.concept_supernodes().keys()),
            "slt_supernodes": list(slt.concept_supernodes().keys()),
            "mlp_filtered_out": mlp_filtered, "slt_filtered_out": slt_filtered,
            "mlp_supernode_missing_in_slt": [n for n, v in mlp_to_slt.items() if not v["matched_features"]],
            "slt_supernode_missing_in_mlp": [n for n, v in slt_to_mlp.items() if not v["matched_features"]],
            "mlp_to_slt": mlp_to_slt, "slt_to_mlp": slt_to_mlp,
            "mlp_grouped": mg, "mlp_ungrouped": mu, "slt_grouped": sg, "slt_ungrouped": su,
        }

    print(f"Comparing {len(entries)} graphs (filter + judge, concurrency={args.concurrency}, "
          f"model={JUDGE_MODEL})...")
    reports = await asyncio.gather(*[run_entry(e) for e in entries])
    for r in reports:
        for name in r["mlp_supernode_missing_in_slt"]:
            print(f'  MLP supernode missing in SLT features: "{name}"  [{r["slug"]}]')
        for name in r["slt_supernode_missing_in_mlp"]:
            print(f'  SLT supernode missing in MLP features: "{name}"  [{r["slug"]}]')

    write_reports(list(reports), args.out_dir)


if __name__ == "__main__":
    main()

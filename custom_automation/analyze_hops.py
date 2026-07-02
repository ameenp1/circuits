"""
analyze_hops.py — intermediate-hop detection for MLP (ADAG) attribution graphs.

The MLP analogue of circuit-tracer-automation/custom_automation/analysis/analyze_hops.py
(which does this for SLT transcoder graphs). For each prompt in a ground-truth CSV it loads
the MLP graph and records the predicted token, whether it's correct, and whether the
intermediate hop (e.g. the state/country) is present — checking supernode group NAMES and the
per-neuron `generated_description` text.

MLP graph schema (from batch_export_neurons -> generate_description -> generate_supernodes):
  { "label": "<slug>", "prompt", "target",
    "neurons": [{"layer","neuron","polarity","attribution","generated_description","group"}],
    "supernodes": {"<group name>": ["L_N_pol", ...]}, "ungrouped": [...] }

The concept-matching logic (aliases, prefix match, answer match) is copied verbatim from the
SLT reference so hop detection is identical on both sides.

Usage:
    python custom_automation/analyze_hops.py \
        --graphs-dir capital_neuron_graphs/ \
        --ground-truth ../circuit-tracer-automation/prompts/ground_truth_capital.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

# ===========================================================================
# Concept matching — copied verbatim from the SLT analyze_hops.py so detection
# is identical across MLP and SLT.
# ===========================================================================

def concept_in_text(concept: str, text: str) -> bool:
    """Exact whole-word match first; prefix fallback (first 4 chars equal) for derived forms."""
    if not concept or not text:
        return False
    c = concept.lower()
    t = text.lower()
    if re.search(r'\b' + re.escape(c) + r'\b', t):
        return True
    if len(c) >= 4:
        for word in re.findall(r'\b\w+\b', t):
            if len(word) >= 4 and c[:4] == word[:4]:
                return True
    return False


_ALIASES: dict[str, list[str]] = {
    "united states of america": ["united states", "america", "usa", "us", "american"],
    "united kingdom": ["uk", "britain", "british", "england", "english", "great britain"],
    "south korea": ["korea", "korean"],
    "north korea": ["korea", "korean"],
    "new zealand": ["zealand", "kiwi"],
    "czech republic": ["czech", "czechia"],
    "soviet union": ["ussr", "soviet"],
    "people's republic of china": ["china", "chinese", "prc"],
    "republic of china": ["china", "chinese", "taiwan", "taiwanese"],
    "india": ["indian", "hindi"],
    "japan": ["japanese"],
    "brazil": ["brazilian", "brasil"],
    "portugal": ["portuguese"],
    "netherlands": ["dutch", "holland"],
    "belgium": ["belgian"],
    "austria": ["austrian"],
    "sweden": ["swedish"],
    "norway": ["norwegian"],
    "poland": ["polish"],
    "greece": ["greek"],
    "russia": ["russian"],
    "turkey": ["turkish"],
    "china": ["chinese"],
    "france": ["french"],
    "germany": ["german"],
    "italy": ["italian"],
    "spain": ["spanish"],
    "mexico": ["mexican"],
    "canada": ["canadian"],
    "australia": ["australian"],
    "egypt": ["egyptian"],
    "thailand": ["thai"],
    "vietnam": ["vietnamese"],
    "pakistan": ["pakistani"],
    "united arab emirates": ["uae", "emirates", "emirati", "dubai"],
    "colombia": ["colombian"],
    "peru": ["peruvian"],
    "ireland": ["irish"],
}


def _prefix_match_info(concept: str, text: str) -> tuple[bool, str]:
    if not concept or not text:
        return False, ""
    c = concept.lower()
    t = text.lower()
    if re.search(r'\b' + re.escape(c) + r'\b', t):
        return False, ""
    if len(c) >= 4:
        for word in re.findall(r'\b\w+\b', t):
            if len(word) >= 4 and c[:4] == word[:4]:
                return True, word
    return False, ""


def expand_concept(concept: str) -> list[str]:
    """Original concept + aliases + significant sub-words (>5 chars, non-stopword)."""
    terms = [concept]
    low = concept.lower()
    if low in _ALIASES:
        terms.extend(_ALIASES[low])
    _STOP_WORDS = {
        "with", "from", "that", "this", "have", "been", "were", "they", "their", "which",
        "where", "when", "what", "about", "being", "other", "there", "after", "before",
        "first", "under", "could", "would", "should", "these", "those", "through",
        "between", "current", "original", "country", "capital",
    }
    words = concept.split()
    if len(words) >= 2:
        for w in words:
            if len(w) > 5 and w.lower() not in _STOP_WORDS:
                terms.append(w)
    seen: set[str] = set()
    unique: list[str] = []
    for t in terms:
        tl = t.lower().strip()
        if tl and tl not in seen:
            seen.add(tl)
            unique.append(tl)
    return unique


def concept_matches_text(concept: str, text: str) -> bool:
    return any(concept_in_text(term, text) for term in expand_concept(concept))


def match_detail(concept: str, text: str) -> tuple[bool, str, str]:
    """Transparency helper: HOW a concept matched a piece of text.

    Returns (matched, term, kind). `kind` is 'exact' (the concept itself, whole word),
    'alias/subword' (an alias or significant sub-word, whole word), or 'prefix(fuzzy)' (only the
    first-4-chars heuristic fired — inspect these, they can be false positives like Ala->Alaska).
    """
    if not concept or not text:
        return (False, "", "")
    base = concept.lower().strip()
    terms = expand_concept(concept)
    tl = text.lower()
    for term in terms:                                    # whole-word matches first (higher trust)
        if re.search(r"\b" + re.escape(term) + r"\b", tl):
            return (True, term, "exact" if term == base else "alias/subword")
    for term in terms:                                    # 4-char prefix fallback (fuzzy)
        hit, word = _prefix_match_info(term, text)
        if hit:
            return (True, f"{term}~{word}", "prefix(fuzzy)")
    return (False, "", "")


def any_concept_matches(concepts: list[str], text: str) -> bool:
    return any(concept_matches_text(c, text) for c in concepts)


# ===========================================================================
# MLP graph loading + hop detection
# ===========================================================================

_SUPPRESS_PREFIXES = ("suppress ", "anti-", "anti ", "demote ", "inhibit ", "avoid ", "repress ")


def _norm_slug(s: str) -> str:
    """Canonical slug key. Handles the mismatches seen in the real data:
      - label carries a run suffix:      'grand-canyon-capital___2' -> 'grand-canyon-capital'
      - filenames use underscores:       'birmingham_capital'       -> 'birmingham-capital'
      - ground-truth uses hyphens:       'birmingham-capital'
    so all three forms collapse to the same key."""
    return s.split("___")[0].strip().lower().replace("_", "-")


def load_mlp_graphs(graphs_dir: Path) -> dict[str, dict]:
    """normalized slug -> MLP graph. Prefers the internal `label` (hyphenated, matches ground
    truth); falls back to the underscore filename. Both are normalized so they line up."""
    out: dict[str, dict] = {}
    for fp in sorted(graphs_dir.glob("graph_*.json")):
        g = json.loads(fp.read_text(encoding="utf-8"))
        raw = str(g.get("label", "")).strip() or fp.stem
        # strip a leading "graph_0001_" index prefix when we fall back to the filename
        raw = re.sub(r"^graph_\d+_", "", raw)
        key = _norm_slug(raw)
        if key:
            out[key] = g
    return out


def _clip(text: str, n: int = 90) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= n else t[: n - 1] + "…"


def detect_intermediate_hop(graph: dict, intermediate_concept: str) -> dict:
    """Is the INTERMEDIATE hop (e.g. the state/country between subject and capital) present?

    This is the whole point of the analysis, so it records the EVIDENCE, not just a boolean:
      (a) supernode group NAMES  (excluding Ungrouped / suppression groups)
      (b) every neuron's generated_description  (grouped AND ungrouped — ungrouped neurons are
          described and count as evidence)
    For each hit it keeps the text that matched, the matched term, and the match kind
    (exact / alias/subword / prefix(fuzzy)) so a reader can see exactly why we call the hop present
    and can discount fuzzy prefix matches. Mirrors the SLT detector's group + description scan.
    """
    concepts = [c.strip() for c in intermediate_concept.split("|")
                if c.strip() and c.strip().upper() != "N/A"]
    per_concept = {c: {"found_in_group": False, "found_in_desc": False} for c in concepts}

    neurons = graph.get("neurons", [])
    group_members: dict[str, list[dict]] = {}
    for n in neurons:
        g = (n.get("group") or "").strip()
        if g and g != "Ungrouped":
            group_members.setdefault(g, []).append(n)

    # (a) supernode group names — record which group, which concept/term, and how it matched
    group_evidence: list[dict] = []
    matching_neurons: dict[int, dict] = {}
    for name, members in group_members.items():
        if any(name.lower().startswith(p) for p in _SUPPRESS_PREFIXES):
            continue
        for concept in concepts:
            ok, term, kind = match_detail(concept, name)
            if ok:
                group_evidence.append({"group": name, "concept": concept, "term": term, "kind": kind})
                per_concept[concept]["found_in_group"] = True
                for m in members:
                    matching_neurons[id(m)] = m
                break

    # (b) per-neuron descriptions (grouped + ungrouped) — record the snippet that matched
    desc_evidence: list[dict] = []
    for n in neurons:
        desc = n.get("generated_description") or ""
        for concept in concepts:
            ok, term, kind = match_detail(concept, desc)
            if ok:
                per_concept[concept]["found_in_desc"] = True
                desc_evidence.append({
                    "layer": n.get("layer"), "neuron": n.get("neuron"),
                    "ungrouped": (n.get("group") or "Ungrouped").strip() in ("", "Ungrouped"),
                    "concept": concept, "term": term, "kind": kind, "desc": _clip(desc),
                })
                break

    influences = [abs(float(n.get("attribution", 0.0))) for n in matching_neurons.values()]
    total = len(neurons)
    matching_group_names = [e["group"] for e in group_evidence]
    hop_found_in_groups = len(group_evidence) > 0
    hop_found_in_desc = len(desc_evidence) > 0
    concepts_found = [c for c in concepts if per_concept[c]["found_in_group"]]
    concepts_found_either = [c for c in concepts
                             if per_concept[c]["found_in_group"] or per_concept[c]["found_in_desc"]]
    # any match that is NOT a fuzzy prefix -> a high-trust hit
    strong = any(e["kind"] != "prefix(fuzzy)" for e in group_evidence + desc_evidence)

    return {
        "total_neurons": total,
        "hop_feature_count": len(matching_neurons),
        "hop_feature_fraction": round(len(matching_neurons) / total, 4) if total else 0.0,
        "hop_mean_influence": round(sum(influences) / len(influences), 4) if influences else 0.0,
        "hop_max_influence": round(max(influences), 4) if influences else 0.0,
        "hop_desc_feature_count": len(desc_evidence),
        "hop_found_in_desc": hop_found_in_desc,
        "hop_groups": matching_group_names,
        "hop_found_in_groups": hop_found_in_groups,
        "hop_found": hop_found_in_groups,                       # headline: present as a supernode
        "hop_found_either": hop_found_in_groups or hop_found_in_desc,
        "hop_match_strong": strong,                            # at least one non-fuzzy match
        "group_evidence": group_evidence,
        "desc_evidence": desc_evidence,
        "concepts_found": concepts_found,
        "concepts_found_either": concepts_found_either,
        "concepts_missed": [c for c in concepts if c not in concepts_found_either],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Intermediate-hop detection for MLP (ADAG) graphs.")
    ap.add_argument("--graphs-dir", type=Path, required=True, help="MLP graph_*.json dir.")
    ap.add_argument("--ground-truth", type=Path, required=True,
                    help="CSV with columns: slug, intermediate_concept, correct_answer, ...")
    ap.add_argument("--out-dir", type=Path, default=Path("custom_automation/analysis/results"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    graphs = load_mlp_graphs(args.graphs_dir)
    with open(args.ground_truth, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    results: list[dict] = []
    evidence: dict[str, dict] = {}   # slug -> {group_evidence, desc_evidence} (kept out of the flat CSV)
    for row in rows:
        slug = row["slug"].strip()
        graph = graphs.get(_norm_slug(slug))
        if graph is None:
            print(f"  SKIP {slug} — no MLP graph found")
            continue
        # NOTE: graph['target'] is the SLUG, not the model's generated token, so we cannot judge
        # model correctness from these graphs. This analysis is only about middle-hop presence.
        det = detect_intermediate_hop(graph, row.get("intermediate_concept", ""))
        evidence[slug] = {"group_evidence": det["group_evidence"], "desc_evidence": det["desc_evidence"]}
        flat = {k: v for k, v in det.items() if k not in ("hop_groups", "group_evidence", "desc_evidence")}
        results.append({
            "slug": slug,
            "intermediate_concept": row.get("intermediate_concept", ""),
            "correct_answer": row.get("correct_answer", "").strip(),
            **flat,
            "hop_groups": "; ".join(det["hop_groups"]),
        })

    if not results:
        print("No results — check --graphs-dir labels match the ground-truth slugs.")
        return

    # CSV (flat scalar columns only)
    csv_path = args.out_dir / "mlp_hop_analysis.csv"
    fields = list(results[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    # MD summary
    n = len(results)
    n_hop = sum(1 for r in results if r["hop_found"])
    n_hop_either = sum(1 for r in results if r["hop_found_either"])
    n_strong = sum(1 for r in results if r["hop_match_strong"])
    n_missed = n - n_hop_either
    md = [
        "# MLP intermediate-hop (middle-hop) analysis", "",
        "The intermediate step (e.g. the STATE/COUNTRY between the subject and its capital) is the "
        "quantity of interest. For each prompt we report whether that concept is present in the MLP "
        "graph and show the exact evidence, so nothing is taken on faith.", "",
        "_(Model correctness is not reported: the MLP graphs store the slug in `target`, not the "
        "model's generated token, so it can't be judged from these files.)_", "",
        f"- Graphs analyzed: **{n}**",
        f"- Middle hop present as a **supernode**: **{n_hop}/{n}** ({n_hop/n:.0%})",
        f"- Present in a supernode **or** any (grouped/ungrouped) neuron description: "
        f"**{n_hop_either}/{n}** ({n_hop_either/n:.0%})",
        f"- Of those, backed by at least one **non-fuzzy** match: **{n_strong}/{n}** "
        "(the rest rely only on the 4-char prefix heuristic — treat as weak)",
        f"- Middle hop **entirely absent**: **{n_missed}/{n}**",
        "",
        "`hop`: ✓ supernode · ~ description-only · ✗ absent.  `strong`: ✓ has a non-fuzzy match.",
        "", "| slug | intermediate | hop | strong | #hop neurons | matched groups |",
        "|---|---|:--:|:--:|---:|---|",
    ]
    for r in results:
        hop = "✓" if r["hop_found"] else ("~" if r["hop_found_either"] else "✗")
        strong = "✓" if r["hop_match_strong"] else ("—" if r["hop_found_either"] else "")
        md.append(f"| {r['slug']} | {r['intermediate_concept']} | {hop} | {strong} | "
                  f"{r['hop_feature_count']} | {r['hop_groups']} |")

    # Per-prompt evidence — exactly why each hop was (or was not) called present
    md += ["", "## Per-prompt evidence", ""]
    for r in results:
        ev = evidence[r["slug"]]
        md.append(f"### {r['slug']} — intermediate: `{r['intermediate_concept']}`")
        if not ev["group_evidence"] and not ev["desc_evidence"]:
            md.append("- **Middle hop ABSENT** — no matching supernode or neuron description.")
            md.append("")
            continue
        for e in ev["group_evidence"]:
            md.append(f"- **supernode** `{e['group']}` — matched *{e['term']}* ({e['kind']})")
        for e in ev["desc_evidence"]:
            tag = "ungrouped neuron" if e["ungrouped"] else "grouped neuron"
            md.append(f"- **{tag}** L{e['layer']}N{e['neuron']} — matched *{e['term']}* "
                      f"({e['kind']}): \"{e['desc']}\"")
        md.append("")
    (args.out_dir / "mlp_hop_analysis.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote {csv_path} and mlp_hop_analysis.md")

    # Terminal summary — same hop-detection metrics as the SLT analyze_hops.py.
    # (SLT's "clerp" scan == MLP's neuron-description scan.) SLT also prints Model correct (top-1),
    # Correct in top-5, Mean rank score, and the correct/wrong x hop breakdown — those need the
    # model's prediction, which the MLP graphs do NOT store (nodes carry only
    # layer/token/neuron/attribution/activation), so they can't be computed here.
    n_hop_groups = n_hop
    n_hop_desc = sum(1 for r in results if r["hop_found_in_desc"])
    print()
    print("=" * 60)
    print(f"  Graphs analysed:            {n}")
    print(f"  --- Hop detection ---")
    print(f"  Groups only:                {n_hop_groups}/{n} ({n_hop_groups/n:.1%})")
    print(f"  Description only:           {n_hop_desc - n_hop_groups}/{n}  (description but not group)")
    print(f"  Groups OR description:      {n_hop_either}/{n} ({n_hop_either/n:.1%})")
    print("=" * 60)
    print("  (model-correct / top-5 / mean-rank-score need the model's prediction, which the")
    print("   MLP graphs don't store — omitted. Hop detection is the computable subset.)")


if __name__ == "__main__":
    main()

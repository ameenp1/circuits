"""
generate_supernodes.py — semantically group ADAG MLP neurons into supernodes.

Step 3 of the custom_automation pipeline (after generate_description.py). Operates
per graph, in place, on the same `graph_*.json` files produced by
`scripts/circuit_prep/batch_export_neurons.py` and annotated by
generate_description.py. Mirrors generate_description.py's CLI (`--graph` /
`--graphs-dir`).

For each graph it reads the described `neurons` (each carries `layer`, `neuron`,
`polarity`, `attribution`, and a `generated_description`) and clusters them into
supernodes with an LLM, in three phases:

  Phase 1: Discover groups from the top-K most influential features.
  Phase 2: Assign the remaining features to existing groups in concurrent batches.
  Phase 3: Reconciliation — merge duplicates, fix misassignments, drop noise.

Only the described MLP neurons are grouped — embedding (layer -1) and logit nodes
are left out (ADAG's exported `nodes` carry no token strings / probabilities).

Writes back into each graph_*.json, in place:
  - graph["supernodes"]  : {group_name: [feature_id, ...]} (excludes Ungrouped)
  - graph["ungrouped"]   : [feature_id, ...]
  - each neuron gains a "group" field (its group name, or "Ungrouped")

A feature_id is `L{layer}_N{neuron}_{polarity}` (same id generate_description uses).

Usage:
    export OPENAI_API_KEY=sk-...
    # one graph
    python generate_supernodes.py --graph ../capitals_neuron_graphs/graph_0000_austin.json
    # a whole folder (writes back in place)
    python generate_supernodes.py --graphs-dir ../capitals_neuron_graphs/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import config` works

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from config import (
    GROUPING_BATCH_SIZE,
    GROUPING_MAX_CONCURRENCY,
    GROUPING_MODEL,
    GROUPING_TOP_K_SEED,
    GROUPING_VARIANT,
    setup_logging,
)

log = setup_logging()


# ---------------------------------------------------------------------------
# Pydantic schemas (structured output)
# ---------------------------------------------------------------------------

class Assignment(BaseModel):
    feature_id: str = Field(description="The feature ID.")
    group_name: str = Field(description="The group name (or 'Ungrouped'). MUST be 5 words or fewer.")


class GroupDef(BaseModel):
    group_name: str = Field(description="Semantic name of the supernode group. MUST be 5 words or fewer. Simple, natural phrasing only.")
    rationale: str = Field(description="Brief rationale for this cluster.")


class Phase1Output(BaseModel):
    groups: list[GroupDef] = Field(description="High-level groups discovered.")
    assignments: list[Assignment] = Field(description="Feature-to-group assignments.")


class Phase2Output(BaseModel):
    assignments: list[Assignment] = Field(description="Feature-to-group assignments.")
    new_groups: list[GroupDef] = Field(description="Any NEW groups created (only if absolutely necessary).")


class RenameAction(BaseModel):
    old_name: str = Field(description="The current group name.")
    new_name: str = Field(description="The new group name (must be <= 5 words).")


class MergeAction(BaseModel):
    groups_to_merge: list[str] = Field(description="List of group names to merge together.")
    merged_name: str = Field(description="The name for the merged group (must be <= 5 words).")


class ReassignAction(BaseModel):
    feature_id: str = Field(description="The feature ID to reassign.")
    from_group: str = Field(description="Current group name.")
    to_group: str = Field(description="Target group name (can be 'Ungrouped' or a new name).")


class Phase3Output(BaseModel):
    renames: list[RenameAction] = Field(default_factory=list, description="Groups to rename.")
    merges: list[MergeAction] = Field(default_factory=list, description="Groups to merge together.")
    reassignments: list[ReassignAction] = Field(default_factory=list, description="Individual features to move between groups.")
    dropped_groups: list[str] = Field(default_factory=list, description="Groups to dissolve (members become Ungrouped). Use for: (1) Grammar kill — names describing syntactic roles, token patterns, or prefix fragments. (2) Relevance drop — concepts with no connection to the prompt's reasoning chain or predicted output. Do NOT dissolve suppression groups (suppress X, anti-X, etc.) — consolidate them instead.")


# ---------------------------------------------------------------------------
# Grouping prompt variants (a0–a3)
#
# All variants share the same say-X / X-itself hard constraint.
# a1–a3 each combine the same three properties at increasing smartness:
#   - borderline pull (assign plausible features rather than leaving Ungrouped)
#   - critical merge constraint (same-role groups that tell the same story → merge)
#   - description reading (scan member descriptions for specific named entities
#     before naming groups generically)
#
#   a0 — neutral baseline
#   a1 — structural rules: borderline pull + critical merge constraint
#   a2 — a1 + description-aware naming
#   a3 — a2 + first-principles reader test on every group decision
# ---------------------------------------------------------------------------

_SAY_X_STRICTNESS = (
    "Treat say-X / X-itself separation as a hard constraint. Every say-X group must contain only promoting features; "
    "every concept group must contain only content features. A single misplaced feature is enough to split or reassign."
)

_SPECIFICITY_BIAS_BASE = (
    "When in doubt between narrower and broader groups, use whichever granularity best explains the model's specific output and prompt. "
    "Consider both the prompt and the predicted output together: a distinction is worth keeping only if it is relevant to what was asked and (or) what the model predicted. "
)

_STRUCTURAL_RULES = (
    "BORDERLINE FEATURES: prefer assigning a borderline feature to a plausible existing group over Ungrouped — "
    "reserve Ungrouped for features with no meaningful connection to the prompt or output. "
    "MERGE CONSTRAINT: when two same-role groups tell the same part of the story and their separation does not help a reader understand the reasoning differently, merge them. "
    "Never merge a say-X group with a concept group — promotion and content always stay separate. Prefer keeping more proper noun specificity when relevant to the prompt and output"
)

_DESCRIPTION_READING = (
    "DESCRIPTION-AWARE NAMING: before naming any group, scan the member descriptions for recurring proper nouns or specific named entities. "
    "If a specific entity (a place, person, concept) appears consistently across descriptions, use that specific name — "
    "do not collapse to a generic category like 'a place' or 'a city' when the descriptions clearly name something specific. "
    "Apply the same rule to say-X groups: if descriptions consistently name a specific entity after the trigger, prefer 'say California' over 'say a place'. "
    "If features clearly relate to an alternate sense of a key prompt word, name the group with a sense qualifier (e.g. 'X (general)') rather than a domain label that is not applicable given the prompt and output context (e.g. 'economic X' or 'X (music)'). "
    "The last few words of a description often carry non-trivial specificity — use them as an additional signal when placing features into groups. "
    "When a description lists specific promoted tokens in portions like proper nouns, treat those tokens as a meaningful naming signal — prefer relevant specificity when existing over broader category labels. Of course, this is given the relevancy to the prompt and output, since features may be polysemantic."
    "Group names must come from member descriptions, not from the prompt and output context. Be faithful to feature descriptions in naming. "
)

_SPECIFICITY_BIAS: dict[str, str] = {
    "a0": _SPECIFICITY_BIAS_BASE + _SAY_X_STRICTNESS,
    "a1": _SPECIFICITY_BIAS_BASE + _STRUCTURAL_RULES + _SAY_X_STRICTNESS,
    "a2": _SPECIFICITY_BIAS_BASE + _STRUCTURAL_RULES + _DESCRIPTION_READING + _SAY_X_STRICTNESS,
    "a3": (
        _SPECIFICITY_BIAS_BASE
        + _STRUCTURAL_RULES
        + _DESCRIPTION_READING
        + "READER TEST: before finalising any group decision, ask — if a reader saw only this group name and its members, "
        "would they understand something specific about why the model predicted this output? "
        "A group that passes is worth keeping. A group that fails should be merged or dropped. "
        "For merges specifically: two groups that tell the same part of the reasoning story should become one; "
        "two groups that explain different steps or angles should stay separate even if semantically close. "
        + _SAY_X_STRICTNESS
    ),
}

if GROUPING_VARIANT not in _SPECIFICITY_BIAS:
    log.warning("Unknown GROUPING_VARIANT '%s' — falling back to a0 behaviour.", GROUPING_VARIANT)
_BIAS: str = _SPECIFICITY_BIAS.get(GROUPING_VARIANT, _SPECIFICITY_BIAS["a0"])

# Shared rules for all phases — specificity bias is injected per-phase below.
GROUPING_PHILOSOPHY = """
GOAL: Produce a cohesive attribution graph that highlights the main intent and meaning of the prompt.


RELEVANCE & UNGROUPED:
- Assign to "Ungrouped": weak, isolated, or noisy features; features not meaningfully connected to the main prompt and output semantics.
- Purely grammatical tokens (prepositions, articles, conjunctions, punctuation, copulas) go to "Ungrouped" unless they clearly promote a semantically meaningful role (this is very rare). Sentence structure-level grammar is mostly pointless, unless the prompt or output is about it.
- "say X" groups are ONLY valid when X is a meaningful content word or category (e.g., "say a color", "say a fruit"). Do NOT create "say X" groups when X is a function word, grammatical structure, or syntactic role — e.g., "say 'is'", "say a relative clause", "say 'of'", "say a preposition" are never valid groups. These belong in Ungrouped.
- A valid group should feel connected to at least one other group in the graph, not like an isolated curiosity.
- "Ungrouped" is not a failure.


GRANULARITY & SPECIFICITY:
- Create only groups clearly supported by the data. Prefer the most specific name the evidence supports over broad buckets.
- Preserve meaningful distinctions in abstraction level when relevant to the prompt — don't blindly merge a broad category with a narrower stable subtype.
- Distinctions that explain WHY the model chose one output over another should be preserved.
- The prompt commits to one active sense of every word in it. Features from alternate senses of a word (senses the prompt does not require) may form their own groups — Phase 3 will consolidate them.


SEMANTIC ROLE — "SAY X" vs "X ITSELF" (high-priority rule):
- A feature that promotes a concept is different from a feature that IS the concept. Keep them in separate groups.
  - Highlighted tokens are function/structural words → the feature sets up what follows; name it "say [what]".
  - Highlighted tokens are content words → the feature represents the concept directly; name the concept.
- say tags in descriptions are strong signals — respect them.
- "Say" is for genuine promoting. Do not use it when features directly represent a concept.
- Only use 'say' in a group name when member descriptions themselves use it. Do not add 'say' by inference.
- Surface overlap is never enough reason to merge groups with different semantic roles.


NAMING (STRICT):
- LIMIT: 5 words maximum. No exceptions.
- Natural, simple phrasing. If you need more than 5 words, the name is too specific.
- Before naming a group, read the member descriptions. If a specific named entity (place, person, concept) recurs across them, use that name — do not default to a generic category when the descriptions clearly point to something specific.
- This applies to say-X groups too: "say California" is better than "say a place" when descriptions consistently name California.
- Avoid parentheses and words like "mention", "reference", "entity", "concept", "topic", or "pattern" when a simpler phrase works.
- Prefer layman's vocabulary.


SPLITTING vs MERGING:
- Split when a group mixes semantic roles or abstraction levels.
- Small groups are fine if they are interpretable and prompt-relevant.
"""


# ---------------------------------------------------------------------------
# Feature loading from a graph_*.json
# ---------------------------------------------------------------------------

def _feature_id(neuron: dict) -> str:
    """Stable id for an MLP neuron: L{layer}_N{neuron}_{polarity} (matches generate_description)."""
    pol = neuron.get("polarity", "")
    return f"L{neuron['layer']}_N{neuron['neuron']}{('_' + pol) if pol else ''}"


def load_features(graph: dict) -> list[dict]:
    """Load described MLP neurons from a graph, most influential first.

    Each returned feature is {id, score, desc, _neuron} where `_neuron` is a
    reference back into graph["neurons"] so the chosen group can be written in place.
    Neurons without a generated_description are skipped (run generate_description first).
    """
    feats: list[dict] = []
    missing = 0
    for n in graph.get("neurons", []):
        desc = n.get("generated_description")
        if not desc or desc == "Error generating description":
            missing += 1
            continue
        feats.append(
            {
                "id": _feature_id(n),
                "score": abs(float(n.get("attribution", 0.0))),
                "desc": desc,
                "_neuron": n,
            }
        )
    if missing:
        log.warning("%d neuron(s) had no generated_description — run generate_description.py first.", missing)
    # Most influential first: Phase 1 seeds from feats[:K], Phase 2 assigns feats[K:].
    feats.sort(key=lambda x: x["score"], reverse=True)
    return feats


def build_output_context(graph: dict, features: list[dict], top_features: int = 12) -> str:
    """Describe the model's target/output so the LLM can judge prompt-relevance.

    ADAG exports no logit nodes with probabilities, so we use the traced `target`
    plus the tokens most consistently promoted across the top features'
    `output_contributions`.
    """
    lines: list[str] = []
    target = graph.get("target")
    if target:
        lines.append(f"Model's traced target output: {target!r}")

    promoted: dict[str, float] = defaultdict(float)
    id_to_neuron = {f["id"]: f["_neuron"] for f in features}
    for f in features[:top_features]:
        for item in id_to_neuron[f["id"]].get("output_contributions") or []:
            try:
                tok, score = item[0], float(item[1])
            except (TypeError, ValueError, IndexError):
                continue
            if score > 0:
                promoted[tok] += score
    if promoted:
        top = sorted(promoted, key=lambda t: promoted[t], reverse=True)[:8]
        lines.append("Output tokens the top features most promote: " + ", ".join(repr(t) for t in top))

    return "\n".join(lines)


def format_feature_list(batch: list[dict]) -> str:
    return "\n".join(f"ID: {f['id']} | Desc: {f['desc']}" for f in batch)


# ---------------------------------------------------------------------------
# Phase 1 — Discover groups from top-K seed features
# ---------------------------------------------------------------------------

async def run_phase1(
    client: AsyncOpenAI,
    seed_features: list[dict],
    prompt_text: str,
    output_context: str,
) -> Phase1Output | None:
    """Discover groups from the top-K seed features. Returns parsed output (or None)."""
    log.info("Phase 1: Discovering groups from top %d features…", len(seed_features))

    phase1_prompt = f"""You are an expert AI interpretability researcher analyzing internal representations of a large language model.
Context: The model was given the following prompt: {prompt_text}


{output_context}


Below are the {len(seed_features)} most influential features that activated during this prompt.
Cluster them into meaningful semantic groups ("supernodes").


{GROUPING_PHILOSOPHY}


Additional guidance for this phase:
- A single feature may form its own group only if it reflects a stable, reusable semantic pattern, not a one-off surface detail.
- Prefer names that make the graph easy to read over taxonomically tidy labels.
- Prefer two narrow groups over one vague bucket — Phase 3 later can merge, but cannot recover lost distinctions easily. When a concept is clearly relevant to the prompt or output, err toward creating a group rather than Ungrouped.
- HARD RULE — do NOT create groups for grammatical or structural patterns under any circumstances. Assign those features directly to Ungrouped. This includes: prepositions and locational connectors (say 'of', say 'in', say 'after', say locational preposition), copulas and predicate framing (say 'is', predicate framing, copula, say noun after copula), sentence-completion or next-token patterns (say completion, say next noun), subword or token-prefix fragments, heading markers, where-clause framing, structural/relational patterns derived from words in the prompt itself (containment verbs, prepositional structures, syntactic connectors), typographic or capitalization patterns (title case, capitalized tokens, proper noun formatting), and word-onset or prefix fragments — these describe token shape, not meaning. The test: does this group name a semantic concept, or does it describe a syntactic role or sentence structure? Concept = valid group. Sentence structure = Ungrouped.
- Do not create groups named for prompt format or input structure (e.g. "fact prompt", "fill-in-the-blank") — these describe the wrapper, not the reasoning content. Assign to Ungrouped.


SPECIFICITY GUIDANCE:
{_BIAS}


Features:
{format_feature_list(seed_features)}
"""

    response = await client.beta.chat.completions.parse(
        model=GROUPING_MODEL,
        messages=[{"role": "user", "content": phase1_prompt}],
        response_format=Phase1Output,
        max_completion_tokens=32768,
    )
    p1 = response.choices[0].message.parsed
    if p1 is None:
        log.error("Phase 1 parsing returned None — check OpenAI response.")
        return None
    log.info("Established %d initial supernodes.", len(p1.groups))
    return p1


def apply_phase1_output(
    p1: Phase1Output,
    active_groups: dict[str, str],
    final_assignments: dict[str, str],
) -> None:
    """Stitch a Phase1Output into the running active_groups + final_assignments."""
    for g in p1.groups:
        active_groups[g.group_name] = g.rationale
    for a in p1.assignments:
        final_assignments[a.feature_id] = a.group_name


# ---------------------------------------------------------------------------
# Phase 2 — Concurrent batch assignment of remaining features
# ---------------------------------------------------------------------------

async def process_batch(
    client: AsyncOpenAI,
    batch: list[dict],
    groups_context: str,
    prompt_text: str,
    output_context: str,
    semaphore: asyncio.Semaphore,
) -> Phase2Output:
    """Assign a single batch of features to existing (or new) groups."""
    prompt = f"""You are an expert AI interpretability researcher analyzing internal representations of a large language model.
Context: The model was given the prompt: {prompt_text}


{output_context}


Current groups and rationales:
{groups_context}


{GROUPING_PHILOSOPHY}


Task: Assign each feature below to the best matching existing group.
These are lower-influence features — they rarely promote meaningfully new semantic concepts beyond what Phase 1 already captured. Default to an existing group or "Ungrouped".
Do not force a match: if no group fits clearly, "Ungrouped" is correct.
Only create a new group if the concept is genuinely absent from the existing groups, clearly relevant to the prompt, and specific enough that multiple features would share it — this should be rare.
{_BIAS}
Features:
{format_feature_list(batch)}
"""
    async with semaphore:
        response = await client.beta.chat.completions.parse(
            model=GROUPING_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format=Phase2Output,
            max_completion_tokens=4096,
            reasoning_effort="low",
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            log.warning("Phase 2 batch returned None — skipping batch.")
            return Phase2Output(assignments=[], new_groups=[])
        return parsed


async def run_phase2_batches(
    client: AsyncOpenAI,
    remaining_features: list[dict],
    active_groups: dict[str, str],
    prompt_text: str,
    output_context: str,
) -> list[Phase2Output]:
    """Assign `remaining_features` to the phase-1 groups in concurrent batches."""
    if not remaining_features:
        return []

    n_batches = (len(remaining_features) + GROUPING_BATCH_SIZE - 1) // GROUPING_BATCH_SIZE
    log.info("Phase 2: Assigning %d features in %d batches…", len(remaining_features), n_batches)
    groups_context = json.dumps(active_groups, indent=2)

    semaphore = asyncio.Semaphore(GROUPING_MAX_CONCURRENCY)
    coros = [
        process_batch(
            client,
            remaining_features[i : i + GROUPING_BATCH_SIZE],
            groups_context,
            prompt_text,
            output_context,
            semaphore,
        )
        for i in range(0, len(remaining_features), GROUPING_BATCH_SIZE)
    ]
    return await asyncio.gather(*coros)


def apply_phase2_outputs(
    p2_outputs: list[Phase2Output],
    active_groups: dict[str, str],
    final_assignments: dict[str, str],
) -> None:
    """Stitch a list of Phase2 batch outputs into running state, in order."""
    for p2 in p2_outputs:
        if p2 is None:
            continue
        for a in p2.assignments:
            final_assignments[a.feature_id] = a.group_name
        for g in p2.new_groups:
            if g.group_name not in active_groups:
                active_groups[g.group_name] = g.rationale
                log.info("New group created mid-stream: %s", g.group_name)


# ---------------------------------------------------------------------------
# Phase 3 — Reconciliation
# ---------------------------------------------------------------------------

def build_group_summary(final_assignments: dict[str, str], all_features: list[dict]) -> str:
    """Build a summary of all groups with their member features + descriptions."""
    id_to_desc: dict[str, str] = {f["id"]: f["desc"] for f in all_features}

    group_members: dict[str, list[str]] = {}
    for fid, gname in final_assignments.items():
        if gname == "Ungrouped":
            continue
        group_members.setdefault(gname, []).append(fid)

    lines: list[str] = []
    for gname, members in sorted(group_members.items()):
        lines.append(f"\n## {gname} ({len(members)} members)")
        for fid in members[:15]:  # cap per group to stay within context
            lines.append(f"  - {fid}: {id_to_desc.get(fid, 'no description')}")
        if len(members) > 15:
            lines.append(f"  ... and {len(members) - 15} more")

    ungrouped_ids = [fid for fid, g in final_assignments.items() if g == "Ungrouped"]
    lines.append(f"\n## Ungrouped ({len(ungrouped_ids)} features)")
    for fid in ungrouped_ids[:20]:
        lines.append(f"  - {fid}: {id_to_desc.get(fid, 'no description')}")
    if len(ungrouped_ids) > 20:
        lines.append(f"  ... and {len(ungrouped_ids) - 20} more")

    return "\n".join(lines)


async def run_phase3(
    client: AsyncOpenAI,
    final_assignments: dict[str, str],
    all_features: list[dict],
    prompt_text: str,
    output_context: str,
) -> Phase3Output | None:
    """Run phase 3 reconciliation. Returns the parsed Phase3Output, or None."""
    log.info("Phase 3: Reconciling groups…")

    group_summary = build_group_summary(final_assignments, all_features)
    # Strip control characters that can corrupt the JSON payload.
    group_summary = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", group_summary)
    num_groups = len({g for g in final_assignments.values() if g != "Ungrouped"})

    phase3_prompt = f"""You are an expert AI interpretability researcher reviewing the output of an automated feature grouping pipeline.


Context: The model was given the prompt: {prompt_text}


{output_context}


The pipeline produced {num_groups} groups from {len(final_assignments)} features. Your job is to clean up the result — rename unclear groups, reassign misplaced features, and drop irrelevant groups as defined below.


{GROUPING_PHILOSOPHY}


Your job is limited to five things only:


1. GRAMMAR KILL: Any group whose name describes a syntactic role, sentence structure, token pattern, word-prefix fragment— move its members to Ungrouped and dissolve it. Examples: "containment verb", "prefix 'ill'", "say location after of", "fill-in-the-blank", "[concept] prefix", "[X] relation", "[X] clause", "location clause". The test: does this name a concept or describe sentence structure / token shape? Structure/shape = dissolve. For borderline say-X groups, judge by X: if X names a concept relevant to the prompt or output reasoning chain, keep the group — the "say" promoting does not make it irrelevant. Exception: if a word is clearly semantic and central to the prompt's reasoning chain, judge by its role in context rather than its word class alone.


2. ALTERNATE SENSE: A word has an alternate sense when it shares a surface form with the relevant concept but means something different given this prompt — this includes any domain (financial, architectural, political, etc.) that the prompt does not require. Example: "notes (music)" or "musical notes" when the prompt asks about note-taking. If alternate-sense groups are present, merge them together into a single fallback group named "[concept] (general)" — do not touch the correct-sense group. Do not split the correct-sense group to create a (general) variant; only create "[concept] (general)" by merging existing alternate-sense groups. If no alternate-sense groups exist, take no action. This applies to genuine alternate senses only — do not use this to merge a specific named group into a broader same-sense group. '[concept] (general)' is strictly for features activating on a genuinely different dictionary definition (e.g. 'bank' as a financial institution vs. a riverbank) — not the same concept in different contexts or positions.


Lastly, suppression-flavored groups (“suppress X”, “demote X”, “anti-X”, “avoid X”, “inhibit X”) must never be merged into the concept group “X” — they represent opposite causal roles. If multiple suppression variants for the same concept X exist, consolidate them into one group named “suppress X”. Always use “suppress X” as the canonical name.


3. RENAME: Are any group names unclear, longer than 5 words, or use jargon? Rename for clarity. A rename must not lose specificity, promote a structural name, or flip a concept group to a say-X group or vice versa. Do not drop intermediate reasoning steps from a group name.


4. REASSIGN: Are any individual features obviously in the wrong group given their description and the prompt? Move them. Only reassign with high confidence.


5. RELEVANCE DROP: If a group's concept has no clear connection to the prompt's reasoning chain or predicted output — it is not a named entity in the prompt or is not specific and interesting in general, not an intermediate reasoning step, and not a framing pattern for the output — drop it (members to Ungrouped). Use the SPECIFICITY GUIDANCE to judge relevance. Exception: keep groups that name a competing value in the same category as the answer (e.g., a wrong language when the prompt asks about a language), and lean towards keeping neighbors or related topics in the same domain (e.g., neighboring states, nearby countries) — these may be informative competing signals, not noise.


SPECIFIC → BROAD PROTECTION: Before any merge or rename, check — is one group semantically more precise than the other (a named entity, specific concept, or something referenced in the prompt or output)? If yes, protect the specific group. "say color" must not collapse into "say appearance"; "say school" must not collapse into "say place name". If the specific group is irrelevant to the reasoning chain, send it to Ungrouped — never collapse into a vaguer group.


Only make changes you are CONFIDENT about. If the grouping looks good, return empty lists for all actions.


SPECIFICITY GUIDANCE:
{_BIAS}


Current grouping:
{group_summary}
"""
    try:
        response3 = await client.beta.chat.completions.parse(
            model=GROUPING_MODEL,
            messages=[{"role": "user", "content": phase3_prompt}],
            response_format=Phase3Output,
            max_completion_tokens=8192,
        )
        p3 = response3.choices[0].message.parsed
    except Exception as e:  # noqa: BLE001
        log.warning("Phase 3 API call failed (%s) — skipping reconciliation.", e)
        return None

    if p3 is None:
        log.warning("Phase 3 parsing returned None — skipping reconciliation.")
    return p3


def apply_phase3(
    phase3: Phase3Output,
    final_assignments: dict[str, str],
    active_groups: dict[str, str],
) -> None:
    """Apply Phase 3 reconciliation actions to the assignments in place."""
    # 1. Renames
    for rename in phase3.renames:
        old, new = rename.old_name, rename.new_name
        if old in active_groups:
            active_groups[new] = active_groups.pop(old)
        for fid in list(final_assignments):
            if final_assignments[fid] == old:
                final_assignments[fid] = new
        log.info("Renamed: '%s' → '%s'", old, new)

    # 2. Merges
    for merge in phase3.merges:
        for old_name in merge.groups_to_merge:
            for fid in list(final_assignments):
                if final_assignments[fid] == old_name:
                    final_assignments[fid] = merge.merged_name
            active_groups.pop(old_name, None)
        active_groups[merge.merged_name] = f"Merged from: {', '.join(merge.groups_to_merge)}"
        log.info("Merged: %s → '%s'", merge.groups_to_merge, merge.merged_name)

    # 3. Individual reassignments
    for ra in phase3.reassignments:
        if ra.feature_id in final_assignments:
            final_assignments[ra.feature_id] = ra.to_group
            log.info("Reassigned: %s from '%s' → '%s'", ra.feature_id, ra.from_group, ra.to_group)

    # 4. Dropped groups
    for gname in phase3.dropped_groups:
        for fid in list(final_assignments):
            if final_assignments[fid] == gname:
                final_assignments[fid] = "Ungrouped"
        active_groups.pop(gname, None)
        log.info("Dropped group: '%s' (members → Ungrouped)", gname)


def apply_phase3_actions(
    p3: Phase3Output | None,
    active_groups: dict[str, str],
    final_assignments: dict[str, str],
) -> None:
    """Apply a Phase3Output to running state, with logging."""
    if p3 is None:
        return
    total = len(p3.renames) + len(p3.merges) + len(p3.reassignments) + len(p3.dropped_groups)
    if total == 0:
        log.info("Phase 3: No changes needed — grouping looks clean.")
        return
    log.info("Phase 3: Applying %d actions…", total)
    apply_phase3(p3, final_assignments, active_groups)


# ---------------------------------------------------------------------------
# Write-back + driver
# ---------------------------------------------------------------------------

def write_supernodes_into_graph(
    graph: dict,
    features: list[dict],
    final_assignments: dict[str, str],
) -> tuple[int, int]:
    """Write groups back into the graph in place. Returns (n_groups, n_ungrouped)."""
    supernodes: dict[str, list[str]] = {}
    ungrouped: list[str] = []
    for f in features:
        gname = final_assignments.get(f["id"], "Ungrouped")
        f["_neuron"]["group"] = gname
        if gname == "Ungrouped":
            ungrouped.append(f["id"])
        else:
            supernodes.setdefault(gname, []).append(f["id"])

    # Sort groups by total influence (largest first) for a stable, readable order.
    score = {f["id"]: f["score"] for f in features}
    graph["supernodes"] = {
        g: sorted(ids, key=lambda i: score.get(i, 0.0), reverse=True)
        for g, ids in sorted(supernodes.items(), key=lambda kv: sum(score.get(i, 0.0) for i in kv[1]), reverse=True)
    }
    graph["ungrouped"] = ungrouped
    return len(supernodes), len(ungrouped)


async def process_graph(
    path: Path, client: AsyncOpenAI, top_k_seed: int = GROUPING_TOP_K_SEED
) -> None:
    """Group one graph_*.json into supernodes and write the result back in place.

    `top_k_seed` is how many of the most-influential features seed Phase 1; the rest
    are assigned in Phase 2.
    """
    graph = json.loads(path.read_text(encoding="utf-8"))
    prompt_text = graph.get("prompt") or "Unknown prompt"
    features = load_features(graph)
    log.info("=== %s — %d described features — '%s' ===", path.name, len(features), prompt_text)
    if not features:
        log.warning("No described features in %s — skipping.", path.name)
        return

    output_context = build_output_context(graph, features)

    active_groups: dict[str, str] = {}
    final_assignments: dict[str, str] = {}

    # Phase 1 — discover groups from the top-K seed features.
    p1 = await run_phase1(client, features[:top_k_seed], prompt_text, output_context)
    if p1 is None:
        log.error("Phase 1 failed for %s — skipping.", path.name)
        return
    apply_phase1_output(p1, active_groups, final_assignments)

    # Phase 2 — assign the remaining features.
    p2 = await run_phase2_batches(
        client, features[top_k_seed:], active_groups, prompt_text, output_context
    )
    apply_phase2_outputs(p2, active_groups, final_assignments)

    # Phase 3 — reconcile.
    p3 = await run_phase3(client, final_assignments, features, prompt_text, output_context)
    apply_phase3_actions(p3, active_groups, final_assignments)

    n_groups, n_ungrouped = write_supernodes_into_graph(graph, features, final_assignments)
    path.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(
        "Wrote %d supernodes (%d features, %d ungrouped) back into %s",
        n_groups, len(features) - n_ungrouped, n_ungrouped, path,
    )


async def main_async(graphs: list[Path], top_k_seed: int = GROUPING_TOP_K_SEED) -> None:
    client = AsyncOpenAI()
    for p in graphs:
        await process_graph(p, client, top_k_seed)


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        log.error("OPENAI_API_KEY not set.")
        sys.exit(1)

    ap = argparse.ArgumentParser(description="Group ADAG MLP neurons into supernodes (in place).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--graph", type=Path, help="A single graph_*.json to group.")
    g.add_argument("--graphs-dir", type=Path, help="A folder of graph_*.json to group.")
    ap.add_argument(
        "--top-k-seed",
        type=int,
        default=GROUPING_TOP_K_SEED,
        help="How many of the most-influential features seed Phase 1 groups; the rest "
        "are assigned in Phase 2 (default: %(default)s; also via GROUPING_TOP_K_SEED env var).",
    )
    args = ap.parse_args()

    if args.top_k_seed <= 0:
        log.error("--top-k-seed must be a positive integer (got %d).", args.top_k_seed)
        sys.exit(1)

    if args.graph:
        graphs = [args.graph]
    else:
        graphs = sorted(args.graphs_dir.glob("graph_*.json"))
    if not graphs:
        log.error("No graph_*.json found.")
        sys.exit(1)

    asyncio.run(main_async(graphs, args.top_k_seed))


if __name__ == "__main__":
    main()

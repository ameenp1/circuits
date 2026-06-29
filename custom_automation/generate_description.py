"""
generate_description.py — LLM descriptions for ADAG MLP-neuron graphs.

Adapts the transcoder-pipeline description step to ADAG's exported neuron JSON
(one graph_*.json per prompt, produced by batch_export_neurons.py). The evidence
block is built in the SAME format the transcoder pipeline uses, so the two sides'
descriptions are comparable:

  - prompt                 -> OVERALL PROMPT CONTEXT
  - corpus exemplars       -> INPUT ACTIVATIONS (mlp_exemplars.json, harvested by
                              harvest_corpus_exemplars.py; up to 10 excerpts, examples[:5]
                              from the TOP band, top-3 triggers in <<<>>>) — the SLT-matched
                              raw-activation text, NOT the per-prompt attribution window
  - output_contributions   -> GLOBAL OUTPUT TOKENS (split into promoted / demoted by sign)

…then calls GPT-5-mini for a `LABEL -- elaboration` description, exactly as the
transcoder pipeline does. The description is written back into each neuron as
`generated_description`, in place. Neurons with no corpus exemplars (dead/uncovered)
get an empty INPUT ACTIVATIONS block — fail loud, no per-prompt fallback.

Usage:
    export OPENAI_API_KEY=sk-...
    python generate_description.py --graphs-dir ../neuronpedia_neuron_graphs/ \
        --exemplars custom_automation/np_data/mlp_exemplars.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

MODEL = "gpt-5-mini"
CONCURRENCY_LIMIT = 50

# Corpus raw-activation exemplars (harvest_corpus_exemplars.py), keyed "L{layer}_N{neuron}_{pol}".
# Loaded in main(). The activating-text evidence comes from here (SLT-matched corpus windows),
# so the MLP description prompt mirrors the transcoder pipeline's input format.
EXEMPLARS: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# System prompt (the transcoder pipeline's default v2 variant, verbatim)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    # Verbatim copy of the transcoder pipeline's v2 system prompt (_V2_CORE + V2 output format,
    # custom_automation/pipeline/generate_description.py) so MLP and SLT descriptions use an
    # IDENTICAL prompt — the comparison must not be confounded by prompt wording.
    "You are a mechanistic interpretability researcher. "
    "You will be given evidence about a single feature neuron. "
    "Your task is to produce a label and brief description for this feature.\n\n"

    "You will receive three types of evidence:\n"
    "1. Overall Prompt Context: the original prompt the model was processing.\n"
    "2. Input Activations: text excerpts where the neuron activated strongly. "
    "The most relevant tokens are delimited by <<<>>>.\n"
    "3. Global Output Tokens: tokens this neuron tends to push toward or away from in the output.\n\n"

    "Use input activations as the primary evidence. "
    "Use prompt context only for disambiguation, not as proof by itself. "
    "Output tokens can be noisy — only factor them in when they show a clear, consistent pattern. If they are consistent, they likely reveal a lot of information. "
    "A tight cluster of specific promoted tokens (e.g. one city, one state) outranks a broader category label — prefer the specific entity.\n\n"

    "STYLE: Write in short, direct fragments — not full sentences. "
    "Get to the point immediately. No filler, no hedging, no grammatical padding. "
    "\n\n"

    "FEATURE TYPES — use this to guide your description style:\n"
    "Features tend to fall into three types. Figure out which one fits, then describe accordingly.\n\n"
    "1. Input features — activate on a specific token or category of tokens.\n"
    "   Describe what they activate on: ‘represents X’ or just name the pattern directly.\n"
    "   If they activate on a range of related things, describe the category.\n\n"
    "2. Output features — consistently promote a specific next token or category.\n"
    "   Label as ‘say X’ when a clear next-token pattern exists. Reference definitions below.\n"
    "   Prepositions may fall under this category, where the important words are subsequent to the thing it is referencing."
    "3. Abstract/middle features — neither cleanly input nor output.\n"
    "   Describe the context pattern: what kind of text, what situation, what role it plays.\n"
    "   These often need the surrounding context of activations, not just the highlighted token.\n\n"

    "’SAY X’ vs ‘X ITSELF’:\n"
    "Features can represent a concept directly, or signal that a concept is about to appear "
    "(activating on structural words right before it — prepositions, articles, punctuation).\n"
    "- Highlighted tokens are content words → SHORT_LABEL is the concept itself.\n"
    "- Highlighted tokens are structural words setting up content → SHORT_LABEL is ‘say [what]’.\n"
    "- Check what follows the trigger across activations: if a specific concept X (e.g. a proper noun, a method name) "
    "CONSISTENTLY appears right after the trigger token, that supports ‘say X’. "
    "When unclear, prefer naming the concept directly 'X' without the 'say' — ‘say X’ is a stronger claim and needs consistent evidence and should not be used lightly: be strict about including 'say' in any feature.\n\n"

    "PROPER NOUNS:\n"
    "If a specific name, place, or entity recurs across the activations — even in a minority of them — "
    "include it in the SHORT_LABEL or elaboration. Don’t collapse to a generic label when a specific one is clearly supported. Beyond highlighted triggers, also consider consistently occurring proper nouns."
    "These are a signal of specificity, not noise. "
    "If highlighted tokens vary widely and the feature looks polysemantic, capture the consistently specific entities that recur across excerpts when clear and possible rather than defaulting to a single broad label.\n\n"

    "AVOID:\n"
    "- Linguistic or technical jargon: copula, lemma, morpheme, orthogonal, syntactic, "
    "prepositional phrase, noun phrase, etc. Prefer layman's vocabulary and casual tone.\n"
    "- Broad labels when something more specific is clearly supported.\n"
    "- Full sentences. Filler (grammar is not required).\n\n"

    "OUTPUT FORMAT: SHORT_LABEL — elaboration\n\n"
    "- SHORT_LABEL: 1-5 words. Natural graph node name — specific over generic.\n"
    "- After ‘ — ‘: 1-2 tight fragments. Add context, what it promotes, or consistent subpatterns. "
    "Skip if the label already says it all.\n"
    "- Total: 10-35 words.\n\n"

    "Return only the formatted line, nothing else."
)

# ---------------------------------------------------------------------------
# Evidence formatting from ADAG neuron fields
# ---------------------------------------------------------------------------

def _neuron_id(n: dict) -> str:
    pol = n.get("polarity", "")
    return f"L{n['layer']}_N{n['neuron']}{('_' + pol) if pol else ''}"


def _format_excerpt(context: str, triggers: list[str]) -> str:
    """Wrap each trigger token in <<<>>> within context (verbatim from the transcoder pipeline)."""
    marked = False
    for t in triggers:
        clean = t.strip()
        if clean and clean in context:
            context = context.replace(clean, f"<<<{clean}>>>", 1)
            marked = True
    if not marked and any(t.strip() for t in triggers):
        context += f" [Activates on: <<<{'|'.join(t.strip() for t in triggers)}>>>]"
    return context


def _corpus_top_activations(neuron: dict, n_windows: int = 5, n_triggers: int = 3) -> list[dict]:
    """SLT-matched activation evidence from the harvested corpus exemplars.

    Mirrors the transcoder pipeline exactly: examples_quantiles[0].examples[:5], top-3 triggers
    by activation, full joined context. Returns [] for neurons with no corpus exemplars
    (dead/uncovered) — fail loud, no per-prompt fallback.
    """
    pol = neuron.get("polarity") or "+"
    base = f"L{neuron['layer']}_N{neuron['neuron']}"
    card = EXEMPLARS.get(f"{base}_{pol}") or EXEMPLARS.get(f"{base}_{'-' if pol == '+' else '+'}")
    if not card:
        return []
    bands = card.get("examples_quantiles") or []
    top = bands[0].get("examples", []) if bands else []   # TOP band == SLT examples_quantiles[0]
    out = []
    for ex in top[:n_windows]:
        tokens = ex.get("tokens") or []
        acts = ex.get("tokens_acts_list") or []
        if acts and len(acts) == len(tokens):
            order = sorted(range(len(acts)), key=lambda i: acts[i], reverse=True)[:n_triggers]
            triggers = [str(tokens[i]) for i in order]
        else:
            triggers = []
        out.append({"triggers": triggers, "context": "".join(str(t) for t in tokens)})
    return out


def _corpus_logits(neuron: dict) -> tuple[list[str] | None, list[str] | None]:
    """(top_logits, bottom_logits) from the card's logit-weights (Fix #2, add_logit_weights.py),
    or (None, None) if absent — the SLT-matched promote/demote lens."""
    pol = neuron.get("polarity") or "+"
    base = f"L{neuron['layer']}_N{neuron['neuron']}"
    card = EXEMPLARS.get(f"{base}_{pol}") or EXEMPLARS.get(f"{base}_{'-' if pol == '+' else '+'}")
    if card and ("top_logits" in card or "bottom_logits" in card):
        return card.get("top_logits", []), card.get("bottom_logits", [])
    return None, None


def _split_contributions(contribs: list) -> tuple[list[str], list[str]]:
    """output_contributions are [token, signed_score]; split into promoted / suppressed."""
    promoted, suppressed = [], []
    for item in contribs or []:
        try:
            tok, score = item[0], float(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        (promoted if score >= 0 else suppressed).append(tok)
    return promoted, suppressed


def build_user_prompt(neuron: dict, prompt_text: str) -> str:
    """Compose the user turn — identical structure/format to the transcoder pipeline's."""
    lines = [f"Neuron {_neuron_id(neuron)}:\n"]

    lines.append("--- OVERALL PROMPT CONTEXT ---")
    lines.append(prompt_text)

    lines.append("\n--- INPUT ACTIVATIONS ---")
    for i, act in enumerate(_corpus_top_activations(neuron)[:10], 1):
        triggers = act.get("triggers") or [act.get("trigger", "")]
        lines.append(f"Excerpt {i}: {_format_excerpt(act.get('context', ''), triggers)}")

    lines.append("\n--- GLOBAL OUTPUT TOKENS ---")
    promoted, suppressed = _corpus_logits(neuron)          # logit-weights (Fix #2), SLT-matched lens
    if promoted is None and suppressed is None:             # fallback: per-prompt output_contributions
        promoted, suppressed = _split_contributions(neuron.get("output_contributions"))
    promoted = (promoted or [])[:5]                        # 5 each, matching the SLT side
    suppressed = (suppressed or [])[:5]
    lines.append(f"Top Promoted Tokens: {', '.join(promoted) if promoted else 'None available'}")
    lines.append(f"Top Demoted Tokens: {', '.join(suppressed) if suppressed else 'None available'}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Async generation
# ---------------------------------------------------------------------------

async def describe_neuron(neuron: dict, prompt_text: str, client: AsyncOpenAI,
                          sem: asyncio.Semaphore, idx: int, total: int) -> None:
    nid = _neuron_id(neuron)
    async with sem:
        for attempt in range(1, 4):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_prompt(neuron, prompt_text)},
                    ],
                    reasoning_effort="low",
                    max_completion_tokens=4096,
                )
                desc = (resp.choices[0].message.content or "").strip()
                neuron["generated_description"] = desc
                log.info("[%d/%d] %s -> %s", idx, total, nid, desc[:70])
                return
            except Exception as exc:  # noqa: BLE001
                log.warning("[%d/%d] %s attempt %d failed: %s", idx, total, nid, attempt, exc)
                if attempt < 3:
                    await asyncio.sleep(2 ** (attempt - 1))
                else:
                    neuron["generated_description"] = "Error generating description"


async def process_graph(path: Path, client: AsyncOpenAI, sem: asyncio.Semaphore) -> None:
    graph = json.loads(path.read_text(encoding="utf-8"))
    prompt_text = graph.get("prompt", "Unknown prompt")
    neurons = graph.get("neurons", [])
    log.info("=== %s — %d neurons — '%s' ===", path.name, len(neurons), prompt_text)

    total = len(neurons)
    tasks = [
        describe_neuron(n, prompt_text, client, sem, i + 1, total)
        for i, n in enumerate(neurons)
    ]
    await asyncio.gather(*tasks)

    path.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote descriptions back into %s", path)


async def main_async(graphs: list[Path]) -> None:
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    for p in graphs:
        await process_graph(p, client, sem)


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM-describe ADAG MLP-neuron graphs.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--graph", type=Path, help="A single graph_*.json to describe.")
    g.add_argument("--graphs-dir", type=Path, help="A folder of graph_*.json to describe.")
    ap.add_argument("--exemplars", type=Path,
                    default=Path("custom_automation/np_data/mlp_exemplars.json"),
                    help="Corpus exemplar store from harvest_corpus_exemplars.py (the activating text).")
    args = ap.parse_args()

    if args.exemplars.exists():
        EXEMPLARS.update(json.loads(args.exemplars.read_text(encoding="utf-8")))
        log.info("Loaded %d corpus exemplar cards from %s", len(EXEMPLARS), args.exemplars)
    else:
        log.warning("No corpus exemplars at %s — INPUT ACTIVATIONS will be empty (fail loud).", args.exemplars)

    if args.graph:
        graphs = [args.graph]
    else:
        graphs = sorted(args.graphs_dir.glob("graph_*.json"))
    if not graphs:
        log.error("No graph_*.json found.")
        sys.exit(1)

    asyncio.run(main_async(graphs))


if __name__ == "__main__":
    main()

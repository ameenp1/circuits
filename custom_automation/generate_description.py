"""
generate_description.py — LLM descriptions for ADAG MLP-neuron graphs.

Adapts the transcoder-pipeline description step to ADAG's exported neuron JSON
(one graph_*.json per prompt, produced by batch_export_neurons.py). For each
neuron it builds an evidence block from ADAG's own fields:

  - prompt              -> OVERALL PROMPT CONTEXT
  - highlighted_text    -> INPUT ACTIVATIONS (the {{...}} markers become <<<...>>>)
  - top_input_tokens    -> the driver tokens
  - output_contributions-> GLOBAL OUTPUT TOKENS (split into promoted / suppressed by sign)

…then calls GPT-5-mini for a `LABEL -- elaboration` description, exactly as the
transcoder pipeline does. The description is written back into each neuron as
`generated_description`, in place, so render_report.py can show text + label
together (working around the broken frontend feature panel).

Usage:
    export OPENAI_API_KEY=sk-...
    # one graph
    python generate_description.py --graph ../capitals_neuron_graphs/graph_0000_austin.json
    # a whole folder (writes back in place)
    python generate_description.py --graphs-dir ../capitals_neuron_graphs/
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

# ---------------------------------------------------------------------------
# System prompt (the transcoder pipeline's default v2 variant, verbatim)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a mechanistic interpretability researcher. You will be given evidence about a "
    "single feature neuron. Your task is to produce a label and brief description for this feature.\n\n"

    "You will receive three types of evidence:\n"
    "1. Overall Prompt Context: the original prompt the model was processing.\n"
    "2. Input Activations: text excerpts where the neuron activated strongly. The most relevant "
    "tokens are delimited by <<<>>>.\n"
    "3. Global Output Tokens: tokens this neuron tends to push toward or away from in the output.\n\n"

    "Use input activations as the primary evidence. Use prompt context only for disambiguation, "
    "not as proof by itself. Output tokens can be noisy — only factor them in when they show a "
    "clear, consistent pattern. A tight cluster of specific promoted tokens (e.g. one city, one "
    "state) outranks a broader category label — prefer the specific entity.\n\n"

    "STYLE: Write in short, direct fragments — not full sentences. Get to the point immediately. "
    "No filler, no hedging, no grammatical padding.\n\n"

    "FEATURE TYPES — use this to guide your description style:\n"
    "1. Input features — activate on a specific token or category of tokens. Describe what they "
    "activate on.\n"
    "2. Output features — consistently promote a specific next token or category. Label as "
    "'say X' when a clear next-token pattern exists.\n"
    "3. Abstract/middle features — neither cleanly input nor output. Describe the context pattern.\n\n"

    "'SAY X' vs 'X ITSELF':\n"
    "- Highlighted tokens are content words -> SHORT_LABEL is the concept itself.\n"
    "- Highlighted tokens are structural words setting up content -> SHORT_LABEL is 'say [what]'.\n"
    "- When unclear, prefer naming the concept directly 'X' without the 'say'.\n\n"

    "PROPER NOUNS: If a specific name, place, or entity recurs across the activations, include it "
    "in the SHORT_LABEL or elaboration. Don't collapse to a generic label when a specific one is "
    "clearly supported.\n\n"

    "AVOID: linguistic/technical jargon; broad labels when something more specific is supported; "
    "full sentences.\n\n"

    "OUTPUT FORMAT: SHORT_LABEL — elaboration\n"
    "- SHORT_LABEL: 1-5 words. Natural graph node name — specific over generic.\n"
    "- After ' — ': 1-2 tight fragments. Add context, what it promotes, or consistent subpatterns.\n"
    "- Total: 10-35 words.\n\n"

    "Return only the formatted line, nothing else."
)

# ---------------------------------------------------------------------------
# Evidence formatting from ADAG neuron fields
# ---------------------------------------------------------------------------

def _neuron_id(n: dict) -> str:
    pol = n.get("polarity", "")
    return f"L{n['layer']}_N{n['neuron']}{('_' + pol) if pol else ''}"


def _adag_highlight_to_markers(text: str) -> str:
    """ADAG marks activating tokens with {{...}}; convert to the <<<...>>> the prompt expects."""
    return text.replace("{{", "<<<").replace("}}", ">>>")


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
    lines = [f"Neuron {_neuron_id(neuron)}:\n"]

    lines.append("--- OVERALL PROMPT CONTEXT ---")
    lines.append(prompt_text)

    lines.append("\n--- INPUT ACTIVATIONS ---")
    highlighted = _adag_highlight_to_markers(neuron.get("highlighted_text", "")) or "(none)"
    lines.append(f"Excerpt 1: {highlighted}")
    drivers = [t for t, _ in (neuron.get("top_input_tokens") or [])[:8]]
    if drivers:
        lines.append(f"Strongest driver tokens (by input attribution): {', '.join(drivers)}")

    lines.append("\n--- GLOBAL OUTPUT TOKENS ---")
    promoted, suppressed = _split_contributions(neuron.get("output_contributions"))
    lines.append(f"Top Promoted Tokens: {', '.join(promoted) if promoted else 'None available'}")
    lines.append(f"Top Suppressed Tokens: {', '.join(suppressed) if suppressed else 'None available'}")

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
    args = ap.parse_args()

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

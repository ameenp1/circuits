"""
serve_with_cards.py — the REAL circuit-tracer frontend, with a working sidebar.

The frontend (Circuit.serve()) renders the graph fine, but its feature sidebar
fetches per-neuron cards from Transluce's Modal store via the local server's
`/api/neuron_exemplars` endpoint — which has no raw Gemma MLP neurons, so it
returns isDead and the sidebar shows nothing.

This script changes ONLY the data source: it loads the ADAG per-prompt neuron
JSON (from batch_export_neurons.py) into a local card store, monkeypatches that
one endpoint to serve those cards, then calls the normal serve. Everything the
user sees — HTML, CSS, JS, the graph, the sidebar layout — is the unmodified
real frontend.

Per-neuron card content (built into the frontend's own schema):
  - examples_quantiles -> the activating text (tokens + per-token attribution)
  - top_logits / bottom_logits -> promoted / suppressed tokens

Caveat: the endpoint is keyed by (layer, neuron) only — no prompt — so a neuron
that appears in several prompts shows ONE representative card (the occurrence
with the largest |attribution|). Good enough to read what a neuron does.

Usage:
    uv run python custom_automation/serve_with_cards.py \
        --circuit results/case_studies/capitals_gemma_circuit.pkl \
        --model-id google/gemma-2-2b \
        --graphs-dir capitals_neuron_graphs/ \
        --port 8041
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Build the local card store from ADAG batch-export JSONs
# ---------------------------------------------------------------------------

def _split_logits(contribs):
    pos, neg = [], []
    for it in contribs or []:
        try:
            tok, score = it[0], float(it[1])
        except (TypeError, ValueError, IndexError):
            continue
        (pos if score >= 0 else neg).append((tok, score))
    pos.sort(key=lambda x: -x[1])
    neg.sort(key=lambda x: x[1])
    return [t for t, _ in pos], [t for t, _ in neg]


def _card_from_neuron(n: dict) -> dict | None:
    """Build one frontend feature-card from a single ADAG neuron (this prompt only)."""
    tokens = n.get("tokens") or []
    acts = [float(a) for a in (n.get("attr_activations") or [])]
    if not tokens:
        return None
    top_logits, bottom_logits = _split_logits(n.get("output_contributions"))
    train_idx = max(range(len(acts)), key=lambda i: acts[i]) if acts else 0
    return {
        "act_min": 0,
        "act_max": max(acts) if acts else 1.0,
        "examples_quantiles": [
            {
                "quantile_name": "Activation on this prompt",
                "examples": [
                    {"tokens": tokens, "tokens_acts_list": acts, "train_token_ind": train_idx}
                ],
            }
        ],
        "top_logits": top_logits[:10],
        "bottom_logits": bottom_logits[:10],
    }


def build_per_prompt_annotations(
    graphs_dir: Path,
) -> tuple[dict[str, dict[tuple[int, int], dict]], list[dict[tuple[int, int], dict]]]:
    """prompt-string -> {(layer, neuron): {"group", "desc"}}, plus the same maps in file order.

    PER-PROMPT, no pooling: the same neuron legitimately belongs to a different supernode
    (and carries a different description) in a different prompt's graph. Pooling these into
    one global map makes every prompt's groups bleed onto every graph — which is exactly the
    bug this avoids. Used to rewrite each exported circuit_<i>.json individually.
    """
    prompt_to_map: dict[str, dict[tuple[int, int], dict]] = {}
    ordered: list[dict[tuple[int, int], dict]] = []
    for fp in sorted(graphs_dir.glob("graph_*.json")):
        graph = json.loads(fp.read_text(encoding="utf-8"))
        m: dict[tuple[int, int], dict] = {}
        for n in graph.get("neurons", []):
            g = (n.get("group") or "").strip()
            d = (n.get("generated_description") or "").strip()
            if d == "Error generating description":
                d = ""
            m[(int(n["layer"]), int(n["neuron"]))] = {"group": g, "desc": d}
        prompt_to_map[graph.get("prompt", "")] = m
        ordered.append(m)
    log.info("Per-prompt annotations: %d prompts from %s", len(ordered), graphs_dir)
    return prompt_to_map, ordered


def rewrite_graph_data_per_prompt(
    data_dir: Path,
    prompt_to_map: dict[str, dict[tuple[int, int], dict]],
    ordered_maps: list[dict[tuple[int, int], dict]],
    num_layers: int,
) -> None:
    """Set each exported circuit_<i>.json's supernodes + node labels from ITS prompt only.

    The graph nodes are already per-prompt (the export is per ci); we only replace the
    pooled qParams.supernodes / node.clerp with the per-prompt assignment so the boxes on
    graph i show graph i's groups, not the union of all 15.
    """
    logit_prefix = str(num_layers + 1)
    for fp in sorted(data_dir.glob("circuit_*.json")):
        data = json.loads(fp.read_text(encoding="utf-8"))
        prompt = data.get("metadata", {}).get("prompt", "")
        pm = prompt_to_map.get(prompt)
        if pm is None:  # fall back to file index (circuit_<i> <-> i-th ADAG graph)
            stem = fp.stem
            if "_" in stem and stem.rsplit("_", 1)[1].isdigit():
                idx = int(stem.rsplit("_", 1)[1])
                if 0 <= idx < len(ordered_maps):
                    pm = ordered_maps[idx]
        if not pm:
            continue

        groups: dict[str, list[str]] = {}
        for node in data.get("nodes", []):
            nid = str(node.get("node_id", ""))
            if nid.startswith("E_"):
                continue
            parts = nid.split("_")
            if len(parts) < 3 or not parts[0].isdigit() or parts[0] == logit_prefix:
                continue
            try:
                key = (int(parts[0]), int(parts[1]))
            except ValueError:
                continue
            info = pm.get(key)
            if not info:
                continue
            if info.get("desc"):
                node["clerp"] = info["desc"]
            g = info.get("group")
            if g and g != "Ungrouped":
                groups.setdefault(g, []).append(nid)

        grouped_ids = {i for ids in groups.values() for i in ids}
        pinned = set(grouped_ids)
        # also pin embedding / logit nodes linked to a grouped node (matches export behavior)
        for link in data.get("links", []):
            s, t = str(link.get("source", "")), str(link.get("target", ""))
            if s in grouped_ids or t in grouped_ids:
                for end in (s, t):
                    if end.startswith("E_") or end.split("_")[0] == logit_prefix:
                        pinned.add(end)

        qp = data.setdefault("qParams", {})
        qp["supernodes"] = [[g, *ids] for g, ids in sorted(groups.items(), key=lambda kv: -len(kv[1]))]
        qp["pinnedIds"] = list(pinned)
        fp.write_text(json.dumps(data), encoding="utf-8")
    log.info("Rewrote per-prompt supernodes/labels into %d graphs.", len(list(data_dir.glob("circuit_*.json"))))


def build_per_prompt_stores(graphs_dir: Path) -> dict[str, dict[tuple[int, int], dict]]:
    """prompt-string -> {(layer, neuron): card}, recomputed per prompt (no pooling)."""
    stores: dict[str, dict[tuple[int, int], dict]] = {}
    files = sorted(graphs_dir.glob("graph_*.json"))
    for fp in files:
        graph = json.loads(fp.read_text(encoding="utf-8"))
        prompt = graph.get("prompt", "")
        d: dict[tuple[int, int], dict] = {}
        for n in graph.get("neurons", []):
            card = _card_from_neuron(n)
            if card is not None:
                d[(int(n["layer"]), int(n["neuron"]))] = card
        stores[prompt] = d
    log.info("Per-prompt stores: %d prompts from %s", len(stores), graphs_dir)
    return stores


# ---------------------------------------------------------------------------
# Patch the one endpoint + track which prompt graph is being viewed
# ---------------------------------------------------------------------------

def install_local_card_endpoint(
    stores: dict[str, dict[tuple[int, int], dict]],
    exemplars: dict[str, dict],
) -> None:
    import os
    import urllib.parse

    import circuits.frontend.server as fe

    Base = fe.CircuitGraphHandler
    ordered_prompts = list(stores.keys())
    state = {"prompt": ordered_prompts[0] if ordered_prompts else None}

    def _store_for_current() -> dict:
        # Single-prompt circuit: no matching can fail — always use the one store.
        if len(stores) == 1:
            return next(iter(stores.values()))
        return stores.get(state["prompt"], {})

    class LocalCardHandler(Base):  # type: ignore[valid-type,misc]
        def do_GET(self) -> None:  # noqa: D401
            # Learn the current prompt from the graph the frontend just loaded.
            path = self.path.split("?")[0]
            if "/graph_data/" in path and path.endswith(".json"):
                fn = path.split("/graph_data/")[-1]
                gpath = os.path.join(getattr(self, "data_dir", ""), fn)
                try:
                    if os.path.exists(gpath):
                        meta = json.load(open(gpath, encoding="utf-8")).get("metadata", {})
                        p = meta.get("prompt", "")
                        # Exact match, else best-effort by trailing slug index (circuit_<i>).
                        if p in stores:
                            state["prompt"] = p
                        else:
                            stem = fn.rsplit(".", 1)[0]
                            if "_" in stem and stem.rsplit("_", 1)[1].isdigit():
                                idx = int(stem.rsplit("_", 1)[1])
                                if 0 <= idx < len(ordered_prompts):
                                    state["prompt"] = ordered_prompts[idx]
                except Exception:  # noqa: BLE001
                    pass
            return super().do_GET()

        def _handle_neuron_exemplars(self) -> None:  # noqa: D401
            params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            try:
                layer = int(params.get("layer", [None])[0])
                neuron = int(params.get("neuron", [None])[0])
            except (TypeError, ValueError):
                return super()._handle_neuron_exemplars()
            # Sidebar activating text = corpus exemplars (SLT-matched), keyed by polarity;
            # fall back to the other polarity if only one was harvested.
            sign = params.get("sign", ["%2B"])[0]
            pol = "-" if sign == "-" else "+"
            other = "+" if pol == "-" else "-"
            card = exemplars.get(f"L{layer}_N{neuron}_{pol}") or exemplars.get(f"L{layer}_N{neuron}_{other}")
            if card is None:
                log.info("L%dN%d_%s: MISS (dead/uncovered) -> Modal/Llama fallback", layer, neuron, pol)
                return super()._handle_neuron_exemplars()
            # Promote/demote = the card's own logit-weights (Fix #2, add_logit_weights.py) if
            # present; else the per-prompt output_contributions for the current prompt.
            card = dict(card)
            pp = _store_for_current().get((layer, neuron)) or {}
            card.setdefault("top_logits", pp.get("top_logits", []))
            card.setdefault("bottom_logits", pp.get("bottom_logits", []))
            log.info("L%dN%d_%s: HIT corpus exemplars (act_max=%s)", layer, neuron, pol, card.get("act_max"))
            body = json.dumps(card).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    fe.CircuitGraphHandler = LocalCardHandler
    log.info("Patched: corpus exemplars (sidebar text) + per-prompt promote/demote logits.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve the real circuit-tracer frontend with local ADAG feature cards.")
    ap.add_argument("--circuit", type=Path, required=True, help="CircuitData pickle.")
    ap.add_argument("--model-id", default="google/gemma-2-2b")
    ap.add_argument("--graphs-dir", type=Path, required=True, help="batch_export_neurons.py output dir.")
    ap.add_argument("--port", type=int, default=8041)
    ap.add_argument("--exemplars", type=Path,
                    default=Path("custom_automation/np_data/mlp_exemplars.json"),
                    help="Corpus exemplar store from harvest_corpus_exemplars.py (the activating text).")
    args = ap.parse_args()

    # Per-prompt stores supply promote/demote logits; the corpus store supplies the
    # SLT-matched activating text shown in the sidebar.
    stores = build_per_prompt_stores(args.graphs_dir)
    exemplars = json.loads(args.exemplars.read_text(encoding="utf-8")) if args.exemplars.exists() else {}
    if exemplars:
        log.info("Loaded %d corpus exemplar cards from %s", len(exemplars), args.exemplars)
    else:
        log.warning("No corpus exemplars at %s — neurons fall through to Llama (fail loud).", args.exemplars)
    prompt_to_map, ordered_maps = build_per_prompt_annotations(args.graphs_dir)

    import tempfile

    from transformers import AutoConfig, AutoTokenizer

    from circuits.analysis.circuit_ops import Circuit
    from circuits.frontend.server import serve as _serve

    install_local_card_endpoint(stores, exemplars)

    log.info("Loading circuit %s", args.circuit)
    c = Circuit.load_from_pickle(str(args.circuit))
    num_layers = AutoConfig.from_pretrained(args.model_id).num_hidden_layers
    c.set_tokenizer(AutoTokenizer.from_pretrained(args.model_id), num_layers=num_layers)

    # Export the graphs ourselves, then rewrite supernodes + labels PER PROMPT (so each
    # graph shows only its own groups), then serve that dir. This replaces c.serve(), whose
    # single global cluster_map made every prompt's supernodes appear on every graph.
    temp_dir = Path(tempfile.mkdtemp(prefix="circuit_tracer_"))
    c.export_to_circuit_tracer(str(temp_dir), slug="circuit")
    rewrite_graph_data_per_prompt(temp_dir, prompt_to_map, ordered_maps, num_layers)

    server = _serve(data_dir=str(temp_dir), port=args.port)
    log.info("Real frontend on port %d (per-prompt cards + supernodes). Open ?slug=circuit_0. Ctrl+C to stop.", args.port)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()

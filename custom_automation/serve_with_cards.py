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

def install_local_card_endpoint(stores: dict[str, dict[tuple[int, int], dict]]) -> None:
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
            card = _store_for_current().get((layer, neuron))
            if card is None:
                log.info("L%dN%d: MISS (not in store for this prompt) -> Modal/Llama fallback", layer, neuron)
                return super()._handle_neuron_exemplars()
            log.info("L%dN%d: HIT local card (top=%s)", layer, neuron, card.get("top_logits", [])[:4])
            body = json.dumps(card).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    fe.CircuitGraphHandler = LocalCardHandler
    log.info("Patched: per-prompt cards, current prompt tracked from /graph_data requests.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve the real circuit-tracer frontend with local ADAG feature cards.")
    ap.add_argument("--circuit", type=Path, required=True, help="CircuitData pickle.")
    ap.add_argument("--model-id", default="google/gemma-2-2b")
    ap.add_argument("--graphs-dir", type=Path, required=True, help="batch_export_neurons.py output dir.")
    ap.add_argument("--port", type=int, default=8041)
    args = ap.parse_args()

    # Build per-prompt stores BEFORE importing/patching the server.
    stores = build_per_prompt_stores(args.graphs_dir)

    from transformers import AutoConfig, AutoTokenizer

    from circuits.analysis.circuit_ops import Circuit

    install_local_card_endpoint(stores)

    log.info("Loading circuit %s", args.circuit)
    c = Circuit.load_from_pickle(str(args.circuit))
    num_layers = AutoConfig.from_pretrained(args.model_id).num_hidden_layers
    c.set_tokenizer(AutoTokenizer.from_pretrained(args.model_id), num_layers=num_layers)

    server = c.serve(port=args.port)
    log.info("Real frontend on port %d (sidebar served from local ADAG cards). Ctrl+C to stop.", args.port)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()

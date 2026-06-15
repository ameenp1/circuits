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


def build_card_store(graphs_dir: Path) -> dict[tuple[int, int], dict]:
    """(layer, neuron) -> frontend feature-card dict. Keeps the highest-|attr| occurrence."""
    store: dict[tuple[int, int], dict] = {}
    best_attr: dict[tuple[int, int], float] = {}
    files = sorted(graphs_dir.glob("graph_*.json"))
    for fp in files:
        graph = json.loads(fp.read_text(encoding="utf-8"))
        for n in graph.get("neurons", []):
            key = (int(n["layer"]), int(n["neuron"]))
            attr = abs(float(n.get("attribution", 0.0)))
            if key in best_attr and attr <= best_attr[key]:
                continue
            tokens = n.get("tokens") or []
            acts = [float(a) for a in (n.get("attr_activations") or [])]
            if not tokens:
                continue
            top_logits, bottom_logits = _split_logits(n.get("output_contributions"))
            train_idx = max(range(len(acts)), key=lambda i: acts[i]) if acts else 0
            store[key] = {
                "act_min": 0,
                "act_max": max(acts) if acts else 1.0,
                "examples_quantiles": [
                    {
                        "quantile_name": "Activating example",
                        "examples": [
                            {
                                "tokens": tokens,
                                "tokens_acts_list": acts,
                                "train_token_ind": train_idx,
                            }
                        ],
                    }
                ],
                "top_logits": top_logits[:10],
                "bottom_logits": bottom_logits[:10],
            }
            best_attr[key] = attr
    log.info("Card store: %d neurons from %d graph file(s) in %s", len(store), len(files), graphs_dir)
    return store


# ---------------------------------------------------------------------------
# Patch the one endpoint, then serve the real frontend unchanged
# ---------------------------------------------------------------------------

def install_local_card_endpoint(store: dict[tuple[int, int], dict]) -> None:
    import urllib.parse

    import circuits.frontend.server as fe

    Base = fe.CircuitGraphHandler

    class LocalCardHandler(Base):  # type: ignore[valid-type,misc]
        def _handle_neuron_exemplars(self) -> None:  # noqa: D401
            params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            try:
                layer = int(params.get("layer", [None])[0])
                neuron = int(params.get("neuron", [None])[0])
            except (TypeError, ValueError):
                return super()._handle_neuron_exemplars()
            card = store.get((layer, neuron))
            if card is None:
                # Not in the exported top-N; fall back to the default (Modal) path.
                return super()._handle_neuron_exemplars()
            body = json.dumps(card).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    fe.CircuitGraphHandler = LocalCardHandler
    log.info("Patched /api/neuron_exemplars to serve local ADAG cards.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve the real circuit-tracer frontend with local ADAG feature cards.")
    ap.add_argument("--circuit", type=Path, required=True, help="CircuitData pickle.")
    ap.add_argument("--model-id", default="google/gemma-2-2b")
    ap.add_argument("--graphs-dir", type=Path, required=True, help="batch_export_neurons.py output dir.")
    ap.add_argument("--port", type=int, default=8041)
    args = ap.parse_args()

    # Build the card store BEFORE importing/patching the server.
    store = build_card_store(args.graphs_dir)

    from transformers import AutoConfig, AutoTokenizer

    from circuits.analysis.circuit_ops import Circuit

    install_local_card_endpoint(store)

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

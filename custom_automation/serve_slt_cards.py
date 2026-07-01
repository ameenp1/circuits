"""
serve_slt_cards.py — view SLT (transcoder) graphs with the REAL feature exemplars + logits.

The plain circuit-tracer frontend, for gemma transcoder features, falls back to Transluce's
Modal store (Llama-3 data) for the click-a-node sidebar. This serves the genuine per-feature
evidence from Neuronpedia's artifacts (feature_descriptions_v2.json) instead:

  - text snippets   -> examples_quantiles, from each feature's `top_activations` (context +
                       triggers marked <<< >>> via activation shading)
  - positive/negative logits -> top_logits / bottom_logits, from `promotes` / `demotes`

Everything else (graph, node labels/clerps, supernodes) is already in the downloaded
test_graphs/<slug>.json, so this only patches the one sidebar endpoint. Per-slug: the store
tracks which graph the frontend loaded (like serve_with_cards.py).

Usage:
    uv run python custom_automation/serve_slt_cards.py \
        --graphs-dir /path/to/test_graphs \
        --descriptions-dir /path/to/slt_descriptions \
        --port 8043
Then open  .../index.html?slug=dallas-capital
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _card_from_feature(feat: dict, top_n: int = 20) -> dict:
    """Frontend feature-card from a Neuronpedia feature_descriptions entry."""
    examples = []
    for act in (feat.get("top_activations") or [])[:top_n]:
        ctx = act.get("context") or ""
        trigs = [t.strip() for t in (act.get("triggers") or []) if t and t.strip()]
        # We only have the joined context string (no per-token acts). Split preserving spaces
        # so the join reproduces the text, and shade tokens that contain a trigger.
        toks = re.findall(r"\s*\S+", ctx) or [ctx]
        acts, peak = [], 0
        for i, tk in enumerate(toks):
            hit = any(tr in tk for tr in trigs)
            acts.append(1.0 if hit else 0.15)
            if hit and peak == 0:
                peak = i
        examples.append({"tokens": toks, "tokens_acts_list": acts, "train_token_ind": peak})
    return {
        "act_min": 0,
        "act_max": 1.0,
        "examples_quantiles": [{"quantile_name": "Max activating", "examples": examples}],
        "top_logits": (feat.get("promotes") or [])[:5],
        "bottom_logits": (feat.get("demotes") or [])[:5],
    }


def build_stores(descriptions_dir: Path) -> dict[str, dict[tuple[int, int], dict]]:
    """slug -> {(layer, feature): card}, from slt_descriptions/<slug>.json (feature_descriptions_v2)."""
    stores: dict[str, dict[tuple[int, int], dict]] = {}
    for fp in sorted(descriptions_dir.glob("*.json")):
        slug = fp.stem
        feats = json.loads(fp.read_text(encoding="utf-8"))
        d: dict[tuple[int, int], dict] = {}
        for feat in feats:
            fid = str(feat.get("id", ""))
            parts = fid.split("_")            # "layer_feature_ctx"
            if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
                continue
            key = (int(parts[0]), int(parts[1]))
            d.setdefault(key, _card_from_feature(feat))  # dedup ctx variants (same latent)
        stores[slug] = d
    log.info("SLT stores: %d slugs from %s", len(stores), descriptions_dir)
    return stores


def install_slt_endpoint(stores: dict[str, dict[tuple[int, int], dict]]) -> None:
    import os
    import urllib.parse

    import circuits.frontend.server as fe

    Base = fe.CircuitGraphHandler
    slugs = list(stores.keys())
    state = {"slug": slugs[0] if slugs else None}

    class SLTHandler(Base):  # type: ignore[valid-type,misc]
        def do_GET(self) -> None:  # noqa: D401
            path = self.path.split("?")[0]
            if "/graph_data/" in path and path.endswith(".json"):
                stem = path.split("/graph_data/")[-1].rsplit(".", 1)[0]
                if stem in stores:
                    state["slug"] = stem
            return super().do_GET()

        def _handle_neuron_exemplars(self) -> None:  # noqa: D401
            params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            try:
                layer = int(params.get("layer", [None])[0])
                neuron = int(params.get("neuron", [None])[0])
            except (TypeError, ValueError):
                return super()._handle_neuron_exemplars()
            card = stores.get(state["slug"], {}).get((layer, neuron))
            if card is None:
                log.info("L%dF%d [%s]: MISS -> Modal/Llama fallback", layer, neuron, state["slug"])
                return super()._handle_neuron_exemplars()
            log.info("L%dF%d [%s]: HIT (top=%s)", layer, neuron, state["slug"], card["top_logits"][:3])
            body = json.dumps(card).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    fe.CircuitGraphHandler = SLTHandler
    log.info("Patched: SLT feature exemplars + logits, current slug tracked from /graph_data.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve SLT transcoder graphs with real feature exemplars.")
    ap.add_argument("--graphs-dir", type=Path, required=True, help="SLT test_graphs dir (<slug>.json).")
    ap.add_argument("--descriptions-dir", type=Path, required=True,
                    help="Dir of <slug>.json feature_descriptions_v2 (from the HF artifacts).")
    ap.add_argument("--port", type=int, default=8043)
    args = ap.parse_args()

    stores = build_stores(args.descriptions_dir)
    install_slt_endpoint(stores)

    from circuits.frontend.server import serve as _serve

    server = _serve(data_dir=str(args.graphs_dir), port=args.port)
    log.info("SLT frontend on port %d — open ?slug=<name>. Ctrl+C to stop.", args.port)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()

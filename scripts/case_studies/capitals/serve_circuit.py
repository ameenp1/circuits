"""Serve circuit in frontend with summary labels.

Usage:
    python scripts/case_studies/capitals/serve_circuit.py circuit.pkl \
        --descriptions-json explanations.json --port 8032
"""

import argparse
import asyncio
import json
import logging
import time

from circuits.analysis.circuit_ops import Circuit
from circuits.analysis.label import summarize_attr_contrib_descriptions
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("circuit", type=str)
    parser.add_argument("--descriptions-json", type=str, default="")
    parser.add_argument("--port", type=int, default=8032)
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    args = parser.parse_args()

    logger.info("Loading circuit from %s", args.circuit)
    c = Circuit.load_from_pickle(args.circuit)
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(args.model_id)
    num_layers = cfg.num_hidden_layers
    t = AutoTokenizer.from_pretrained(args.model_id)
    c.set_tokenizer(t, num_layers=num_layers)

    if args.descriptions_json:
        logger.info("Loading descriptions from %s", args.descriptions_json)
        with open(args.descriptions_json) as f:
            d = json.load(f)

        attr_descs: dict[str, str] = {}
        contrib_descs: dict[str, str] = {}
        for name in d.get("attr", {}):
            best_attr, best_attr_s = "", -1.0
            for sign in ("pos", "neg"):
                for e in d["attr"][name].get(sign, []):
                    if e.get("score") and e["score"] > best_attr_s:
                        best_attr_s = e["score"]
                        best_attr = e["explanation"]
            best_contrib, best_contrib_s = "", -1.0
            for sign in ("pos", "neg", "combined"):
                for e in d.get("contrib", {}).get(name, {}).get(sign, []):
                    if e.get("score") and e["score"] > best_contrib_s:
                        best_contrib_s = e["score"]
                        best_contrib = e["explanation"]
            if best_attr:
                attr_descs[name] = best_attr
            if best_contrib:
                contrib_descs[name] = best_contrib

        # Generate summary labels
        descs_input = {
            n: {"attr": attr_descs.get(n, ""), "contrib": contrib_descs.get(n, "")}
            for n in set(attr_descs) | set(contrib_descs)
        }
        logger.info("Generating summary labels for %d clusters...", len(descs_input))
        labels = asyncio.run(summarize_attr_contrib_descriptions(descs_input))
        for name, lbl in sorted(labels.items()):
            logger.info("  %s -> %s", name, lbl)

        c._cluster_descriptions = labels
        c._cluster_attr_descriptions = attr_descs
        c._cluster_contrib_descriptions = contrib_descs

    server = c.serve(port=args.port)
    logger.info("Serving on port %d. Ctrl+C to stop.", args.port)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()

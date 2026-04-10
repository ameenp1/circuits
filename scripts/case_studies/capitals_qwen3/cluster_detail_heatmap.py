"""Plot attr_map and contrib_map for a single cluster on a single example as highlighted tokens.

Each token is shown inline with background color proportional to its score and the
numerical score in superscript.

Usage:
    python scripts/case_studies/capitals_qwen3/cluster_detail_heatmap.py \
        --cluster C44 --label 0
    python scripts/case_studies/capitals_qwen3/cluster_detail_heatmap.py \
        --cluster C44 --label 0 --circuit /path/to/circuit.pkl --model-id Qwen/Qwen3-32B
"""

import argparse
import logging
from html import escape
from pathlib import Path

import numpy as np
from circuits.analysis.circuit_ops import Circuit
from circuits.utils.constants import RESULTS_DIR
from transformers import AutoConfig, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CIRCUIT_PICKLE = RESULTS_DIR / "case_studies/capitals_qwen3_circuit.pkl"
OUTPUT_DIR = RESULTS_DIR / "case_studies/capitals_qwen3"


def _color(val: float, vmax: float, alpha: float = 0.7) -> str:
    """Map value to blue-white-red background color with controllable alpha."""
    if vmax == 0:
        return "rgba(255,255,255,0)"
    t = float(np.clip(val / vmax, -1, 1))
    a = abs(t) * alpha
    if t >= 0:
        return f"rgba(33,102,172,{a:.2f})"  # RdBu blue (positive)
    else:
        return f"rgba(178,24,43,{a:.2f})"  # RdBu red (negative)


def _short_token(tok: str) -> str:
    """Shorten special tokens for display."""
    tok = tok.strip()
    tok = tok.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    if not tok:
        return "' '"
    return tok


def _render_token_row(tokens: list[str], scores: list[float], vmax: float) -> str:
    """Render a row of tokens as inline SVG <tspan> elements with background rects.

    Returns an HTML snippet (foreignObject) since SVG text doesn't support
    inline background colors easily.
    """
    parts: list[str] = []
    for tok, score in zip(tokens, scores):
        bg = _color(score, vmax)
        safe_tok = escape(tok)
        parts.append(f'<span class="tok" style="background:{bg};">{safe_tok}</span>')
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuit", type=str, default=str(CIRCUIT_PICKLE))
    parser.add_argument("--cluster", type=str, default="C44", help="Cluster name e.g. C44")
    parser.add_argument("--label", type=str, default="0", help="CI index or label string")
    parser.add_argument("--n-clusters", type=int, default=64)
    parser.add_argument("--combine", type=str, default="harmonic")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    logger.info("Loading circuit from %s", args.circuit)
    circuit = Circuit.load_from_pickle(args.circuit)
    cfg = AutoConfig.from_pretrained(args.model_id)
    num_layers = cfg.num_hidden_layers
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    circuit.set_tokenizer(tokenizer, num_layers=num_layers)

    # Resolve label
    try:
        ci_idx = int(args.label)
        label_str = circuit.labels[ci_idx]
    except ValueError:
        label_str = args.label
        ci_idx = next(i for i, l in enumerate(circuit.labels) if args.label in l)

    logger.info("Clustering (k=%d, combine=%s)...", args.n_clusters, args.combine)
    circuit.cluster_multiview(
        n_clusters=args.n_clusters, combine=args.combine, get_desc=False, verbose=False
    )

    cluster_map = circuit._cluster_map
    target_cluster = args.cluster

    # Map (layer, neuron) -> cluster name
    neuron_to_cluster: dict[tuple[int, int], str] = {}
    for nid, cl in cluster_map.items():
        neuron_to_cluster[(int(nid.layer), int(nid.neuron))] = cl

    # Find the label string used in df_node
    df_label = f"{label_str}___{ci_idx}"
    if df_label not in circuit.df_node["label"].values:
        df_label = label_str

    # Aggregate attr_map and contrib_map for this cluster + example
    attr_map_sum: np.ndarray | None = None
    contrib_map_sum: np.ndarray | None = None
    # Collect neurons in this cluster for this example
    cluster_neurons: list[tuple[int, int, str, float]] = (
        []
    )  # (layer, neuron, polarity, attribution)

    df_ex = circuit.df_node[circuit.df_node["label"] == df_label]
    logger.info("Found %d neurons for label %s", len(df_ex), df_label)

    for _, row in df_ex.iterrows():
        layer, neuron = int(row["layer"]), int(row["neuron"])
        if layer < 0 or layer >= num_layers:
            continue
        cl = neuron_to_cluster.get((layer, neuron))
        if cl != target_cluster:
            continue

        polarity = "+" if float(row.get("activation", 0)) >= 0 else "-"
        attribution = float(row.get("attribution", 0))
        # Accumulate attribution per unique (layer, neuron, polarity)
        key = (layer, neuron, polarity)
        found = False
        for idx_n, (l, n, p, a) in enumerate(cluster_neurons):
            if (l, n, p) == key:
                cluster_neurons[idx_n] = (l, n, p, a + attribution)
                found = True
                break
        if not found:
            cluster_neurons.append((layer, neuron, polarity, attribution))

        if row["attr_map"] is not None:
            am = np.asarray(row["attr_map"])
            if attr_map_sum is None:
                attr_map_sum = am.copy()
            else:
                ml = max(len(attr_map_sum), len(am))
                attr_map_sum = np.pad(attr_map_sum, (0, ml - len(attr_map_sum))) + np.pad(
                    am, (0, ml - len(am))
                )

        if row["contrib_map"] is not None:
            cm = np.asarray(row["contrib_map"])
            if contrib_map_sum is None:
                contrib_map_sum = cm.copy()
            else:
                ml = max(len(contrib_map_sum), len(cm))
                contrib_map_sum = np.pad(contrib_map_sum, (0, ml - len(contrib_map_sum))) + np.pad(
                    cm, (0, ml - len(cm))
                )

    if attr_map_sum is None:
        attr_map_sum = np.zeros(1)
    if contrib_map_sum is None:
        contrib_map_sum = np.zeros(1)

    # Get input tokens
    ci = circuit.cis[ci_idx]
    pad_offset = 0
    if circuit.attention_masks is not None and ci_idx < len(circuit.attention_masks):
        mask = circuit.attention_masks[ci_idx]
        pad_offset = next(
            (j for j, m in enumerate(mask) if (m.item() if hasattr(m, "item") else m) == 1), 0
        )
        ci = ci[pad_offset:]
    input_tokens = [_short_token(tokenizer.decode([t])) for t in ci]

    # Get output logit tokens
    target_logit_ids = circuit.target_logits[ci_idx] if ci_idx < len(circuit.target_logits) else []
    output_tokens = [_short_token(tokenizer.decode([t])) for t in target_logit_ids]

    # attr_map is stripped of BOS during tracing — align to suffix of input tokens
    attr_offset = len(input_tokens) - len(attr_map_sum)
    attr_scores = [
        (
            float(attr_map_sum[j - attr_offset])
            if j >= attr_offset and (j - attr_offset) < len(attr_map_sum)
            else 0.0
        )
        for j in range(len(input_tokens))
    ]
    contrib_scores = [
        float(contrib_map_sum[j]) if j < len(contrib_map_sum) else 0.0
        for j in range(len(output_tokens))
    ]

    attr_vmax = max(abs(s) for s in attr_scores) if attr_scores else 1.0
    contrib_vmax = max(abs(s) for s in contrib_scores) if contrib_scores else 1.0

    logger.info("Attr vmax=%.4f, Contrib vmax=%.4f", attr_vmax, contrib_vmax)

    attr_html = _render_token_row(input_tokens, attr_scores, attr_vmax)
    contrib_html = _render_token_row(output_tokens, contrib_scores, contrib_vmax)

    cluster_label = target_cluster
    summary_labels = getattr(circuit, "_cluster_summary_labels", {})
    if target_cluster in summary_labels:
        cluster_label = f"{target_cluster}: {summary_labels[target_cluster]}"

    # Fetch per-neuron descriptions from the neuron database
    neuron_desc: dict[tuple[int, int, str], str] = {}
    try:
        from circuits.utils.descriptions import db, get_descriptions_for_neurons
        from neurondb.filters import Neuron as DBNeuron
        from neurondb.filters import NeuronPolarity

        if db is not None:
            neuron_objs = []
            neuron_keys = []
            for layer, neuron, polarity, _ in cluster_neurons:
                pol = NeuronPolarity.POS if polarity == "+" else NeuronPolarity.NEG
                neuron_objs.append(DBNeuron(layer=layer, neuron=neuron, polarity=pol))
                neuron_keys.append((layer, neuron, polarity))
            if neuron_objs:
                descs = get_descriptions_for_neurons(neuron_objs, db)
                for key, desc in zip(neuron_keys, descs):
                    if desc:
                        neuron_desc[key] = desc
            logger.info(
                "Fetched %d/%d neuron descriptions from DB", len(neuron_desc), len(cluster_neurons)
            )
        else:
            logger.warning("Neuron DB not available, skipping descriptions")
    except ImportError:
        logger.warning("neurondb not installed, skipping descriptions")

    # Build neurons list sorted by attribution magnitude
    cluster_neurons.sort(key=lambda x: abs(x[3]), reverse=True)
    neurons_html_parts: list[str] = []
    for layer, neuron, polarity, attribution in cluster_neurons:
        desc = neuron_desc.get((layer, neuron, polarity), "")
        desc_str = f" &mdash; {escape(desc)}" if desc else ""
        neurons_html_parts.append(
            f'<div class="neuron">L{layer}/N{neuron}/{polarity}'
            f' <span class="attr-val">({attribution:+.3f})</span>'
            f"{desc_str}</div>"
        )
    neurons_html = "\n".join(neurons_html_parts) if neurons_html_parts else "<em>none</em>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{escape(cluster_label)}</title>
<style>
body {{ font-family: system-ui, sans-serif; font-size: 13px; margin: 20px; background: #fff; }}
h2 {{ font-size: 15px; margin-bottom: 12px; }}
.section-label {{ font-size: 11px; color: #888; margin-bottom: 4px; }}
.tokens {{ line-height: 2.0; margin-bottom: 16px; }}
.tok {{ padding: 2px 4px; margin: 0 1px; border-radius: 3px; font-family: monospace; font-size: 12px; }}
.neuron {{ font-family: monospace; font-size: 11px; }}
.attr-val {{ color: #888; font-size: 10px; }}
.neurons-list {{ line-height: 1.8; margin-bottom: 16px; }}
</style></head><body>
<h2>{escape(cluster_label)} &mdash; &ldquo;{escape(label_str.strip())}&rdquo;</h2>
<div class="section-label">Input attribution:</div>
<div class="tokens">{attr_html}</div>
<div class="section-label">Output contribution:</div>
<div class="tokens">{contrib_html}</div>
<div class="section-label">Neurons:</div>
<div class="neurons-list">{neurons_html}</div>
</body></html>"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = (
            OUTPUT_DIR / f"cluster_{target_cluster}_{label_str.strip().replace(' ', '_')}.html"
        )
    out_path.write_text(html)
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()

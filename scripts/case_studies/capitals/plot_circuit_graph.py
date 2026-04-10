"""Plot a circuit graph for a single example (CI) as a publication-ready PDF/PNG.

Shows input tokens at the top, clustered hidden neurons in the middle (grouped
by cluster), and output logit tokens at the bottom. Edges show attribution flow
with width proportional to weight.

Usage:
    python scripts/case_studies/capitals/plot_circuit_graph.py circuit.pkl --label 0
    python scripts/case_studies/capitals/plot_circuit_graph.py circuit.pkl --label "capitals___3"
    python scripts/case_studies/capitals/plot_circuit_graph.py circuit.pkl --label 0 --output graph.pdf
"""

import argparse
import logging
from pathlib import Path

import graphviz
from circuits.analysis.circuit_ops import Circuit
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs")

# Colors
CLUSTER_COLORS = [
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ac",
    "#86bcb6",
    "#d4a6c8",
    "#8cd17d",
    "#ffbe7d",
    "#b6992d",
    "#499894",
]
POS_EDGE_COLOR = "#2166ac"
NEG_EDGE_COLOR = "#b2182b"


def resolve_label(circuit: Circuit, label_arg: str) -> tuple[str, int]:
    """Resolve label argument to (label_string, ci_index)."""
    # Try as integer index first
    try:
        ci_idx = int(label_arg)
        if ci_idx < len(circuit.labels):
            return circuit.labels[ci_idx], ci_idx
    except ValueError:
        pass
    # Try as label string
    for i, lbl in enumerate(circuit.labels):
        if lbl == label_arg:
            return lbl, i
    # Try partial match
    for i, lbl in enumerate(circuit.labels):
        if label_arg in lbl:
            return lbl, i
    raise ValueError(f"Label '{label_arg}' not found. Available: {circuit.labels[:10]}...")


_SPECIAL_TOKEN_MAP: dict[str, str] = {
    "<|begin_of_text|>": "[BOT]",
    "<|end_of_text|>": "[EOT]",
    "<|start_header_id|>": "[SHI]",
    "<|end_header_id|>": "[EHI]",
    "<|eot_id|>": "[EOT]",
    "<|finetune_right_pad_id|>": "[PAD]",
    "<|im_start|>": "[IMS]",
    "<|im_end|>": "[IME]",
    "<think>": "[THK]",
    "</think>": "[/THK]",
}


def _pp_token(tok: str) -> str:
    """Pretty-print a token for display, escaping whitespace and special chars."""
    tok = tok.strip()
    # Shorten known special tokens
    if tok in _SPECIAL_TOKEN_MAP:
        return _SPECIAL_TOKEN_MAP[tok]
    # Escape whitespace (double-escape so graphviz renders literally)
    tok = (
        tok.replace("\\", "\\\\")
        .replace("\n", "\\\\n")
        .replace("\r", "\\\\r")
        .replace("\t", "\\\\t")
    )
    tok = tok.strip()
    if not tok:
        tok = "' '"
    return tok


def build_graph(
    circuit: Circuit,
    label: str,
    ci_idx: int,
    top_n_edges: int = 50,
    guarantee_edges: bool = True,
    include_inactive_tokens: bool = False,
) -> graphviz.Digraph:
    """Build a cluster-level graphviz Digraph for a single example.

    Nodes: input tokens, cluster supernodes, output logit tokens.
    Edges: aggregated (summed weights) between clusters and between IO↔clusters.
    """
    from collections import defaultdict

    num_layers = circuit.num_layers or int(circuit.df_node["layer"].max())
    tokenizer = circuit.tokenizer

    # Filter data for this label — df labels may have "___N" suffix
    df_label = f"{label}___{ci_idx}"
    if df_label in circuit.df_node["label"].values:
        match_label = df_label
    elif label in circuit.df_node["label"].values:
        match_label = label
    else:
        matches = [l for l in circuit.df_node["label"].unique() if label in l]
        match_label = matches[0] if matches else label

    df_n = circuit.df_node[circuit.df_node["label"] == match_label].copy()
    df_e = (
        circuit.df_edge[circuit.df_edge["label"] == match_label].copy()
        if len(circuit.df_edge) > 0
        else circuit.df_edge.copy()
    )

    # Get input tokens (strip padding, track offset for edge alignment)
    ci = circuit.cis[ci_idx]
    pad_offset = 0
    if circuit.attention_masks is not None and ci_idx < len(circuit.attention_masks):
        mask = circuit.attention_masks[ci_idx]
        pad_offset = next(
            (j for j, m in enumerate(mask) if (m.item() if hasattr(m, "item") else m) == 1), 0
        )
        ci = ci[pad_offset:]
    prompt_tokens = [tokenizer.decode([t]) for t in ci] if tokenizer else [str(t) for t in ci]

    # Get target logit tokens and probabilities
    target_logit_ids = circuit.target_logits[ci_idx] if ci_idx < len(circuit.target_logits) else []
    target_logit_tokens = (
        [tokenizer.decode([t]) for t in target_logit_ids]
        if tokenizer
        else [str(t) for t in target_logit_ids]
    )
    target_logit_probs: list[float] = []
    if (
        hasattr(circuit, "circuit_data")
        and hasattr(circuit.circuit_data, "target_logit_probs")
        and circuit.circuit_data.target_logit_probs is not None
        and ci_idx < len(circuit.circuit_data.target_logit_probs)
    ):
        target_logit_probs = [float(p) for p in circuit.circuit_data.target_logit_probs[ci_idx]]
    else:
        target_logit_probs = [0.0] * len(target_logit_ids)

    # Map (layer, neuron) -> cluster name
    cluster_map = circuit._cluster_map
    neuron_to_cluster: dict[tuple[int, int], str] = {}
    for nid, cl in cluster_map.items():
        neuron_to_cluster[(int(nid.layer), int(nid.neuron))] = cl

    cluster_descs = getattr(circuit, "_cluster_descriptions", {})

    # Collect active clusters and count neurons per cluster
    active_clusters: set[str] = set()
    for _, row in df_n.iterrows():
        layer, neuron = int(row["layer"]), int(row["neuron"])
        if 0 <= layer < num_layers:
            cl = neuron_to_cluster.get((layer, neuron), "-1")
            if cl not in ("-1", "__unclustered__"):
                active_clusters.add(cl)

    # Count neurons per cluster in this example
    cluster_neuron_count: dict[str, int] = defaultdict(int)
    for _, row in df_n.iterrows():
        layer, neuron = int(row["layer"]), int(row["neuron"])
        if 0 <= layer < num_layers:
            cl = neuron_to_cluster.get((layer, neuron), "-1")
            if cl in active_clusters:
                cluster_neuron_count[cl] += 1

    # Sum signed attribution per cluster (subtract intra-cluster edges later)
    cluster_attr_raw: dict[str, float] = defaultdict(float)
    for _, row in df_n.iterrows():
        layer, neuron = int(row["layer"]), int(row["neuron"])
        if 0 <= layer < num_layers:
            cl = neuron_to_cluster.get((layer, neuron), "-1")
            if cl in active_clusters:
                cluster_attr_raw[cl] += float(row["attribution"])

    # --- Resolve each edge endpoint to a supernode ID ---
    def _supernode_id(layer: int, token: int, neuron: int) -> str | None:
        if layer == -1:
            idx = token - pad_offset
            if idx < 0 or idx >= len(prompt_tokens):
                return None
            return f"inp_{idx}"
        elif layer == num_layers:
            if neuron in target_logit_ids:
                return f"out_{target_logit_ids.index(neuron)}"
            return None
        else:
            cl = neuron_to_cluster.get((layer, neuron), "-1")
            if cl in active_clusters:
                return f"cl_{cl}"
            return None

    # Aggregate edges: sum attribution between supernode pairs
    agg_edges: dict[tuple[str, str], float] = defaultdict(float)
    intra_cluster_attr: dict[str, float] = defaultdict(float)
    for _, row in df_e.iterrows():
        sl, tl = (int(x) for x in str(row["layer"]).split("->"))
        st, tt = (int(x) for x in str(row["token"]).split("->"))
        sn, tn = (int(x) for x in str(row["neuron"]).split("->"))
        src_id = _supernode_id(sl, st, sn)
        tgt_id = _supernode_id(tl, tt, tn)
        if src_id is None or tgt_id is None:
            continue
        if src_id == tgt_id:
            # Track intra-cluster attribution for correction
            if src_id.startswith("cl_"):
                intra_cluster_attr[src_id[3:]] += float(row["attribution"])
            continue
        agg_edges[(src_id, tgt_id)] += float(row["attribution"])

    # Corrected cluster attribution: raw node attr - intra-cluster edge attr
    cluster_attr_signed: dict[str, float] = {}
    cluster_attr: dict[str, float] = {}
    for cl in active_clusters:
        signed = cluster_attr_raw[cl] - intra_cluster_attr.get(cl, 0.0)
        cluster_attr_signed[cl] = signed
        cluster_attr[cl] = abs(signed)

    # Sort by absolute aggregated attribution, keep top N
    sorted_edges = sorted(agg_edges.items(), key=lambda x: abs(x[1]), reverse=True)
    if top_n_edges > 0:
        top_set = set((s, t) for (s, t), _ in sorted_edges[:top_n_edges])

        if guarantee_edges:
            # Ensure each cluster has its strongest incoming and outgoing edge
            best_incoming: dict[str, tuple[tuple[str, str], float]] = {}
            best_outgoing: dict[str, tuple[tuple[str, str], float]] = {}
            for (src, tgt), w in sorted_edges:
                if tgt.startswith("cl_"):
                    cl = tgt[3:]
                    if cl not in best_incoming or abs(w) > abs(best_incoming[cl][1]):
                        best_incoming[cl] = ((src, tgt), w)
                if src.startswith("cl_"):
                    cl = src[3:]
                    if cl not in best_outgoing or abs(w) > abs(best_outgoing[cl][1]):
                        best_outgoing[cl] = ((src, tgt), w)

            # Add guaranteed edges
            guaranteed = set()
            for edge_w in list(best_incoming.values()) + list(best_outgoing.values()):
                guaranteed.add(edge_w[0])
            # Merge: top N + guaranteed
            top_set |= guaranteed
        sorted_edges = [(k, v) for k, v in sorted_edges if k in top_set]

    # Determine connected IO nodes
    connected_io: set[str] = set()
    for (src, tgt), _ in sorted_edges:
        if src.startswith("inp_") or src.startswith("out_"):
            connected_io.add(src)
        if tgt.startswith("inp_") or tgt.startswith("out_"):
            connected_io.add(tgt)

    # Build cluster color map (only active clusters with edges)
    connected_clusters = set()
    for (src, tgt), _ in sorted_edges:
        if src.startswith("cl_"):
            connected_clusters.add(src[3:])
        if tgt.startswith("cl_"):
            connected_clusters.add(tgt[3:])
    cluster_names = sorted(connected_clusters)
    cluster_color = {
        name: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, name in enumerate(cluster_names)
    }

    # --- Build graph (BT = bottom-to-top: inputs at bottom, outputs at top) ---
    dot = graphviz.Digraph(
        "circuit",
        graph_attr={
            "rankdir": "BT",
            "bgcolor": "white",
            "fontname": "Palatino",
            "fontsize": "5",
            "pad": "0.01",
            "margin": "0.01",
            "nodesep": "0.02",
            "ranksep": "0.15",
            "dpi": "300",
            "remincross": "true",
        },
        node_attr={
            "fontname": "Palatino",
            "fontsize": "5",
            "style": "filled",
            "shape": "box",
            "height": "0.1",
            "width": "0",
            "margin": "0.01,0.005",
        },
        edge_attr={
            "fontname": "Palatino",
            "fontsize": "4",
            "arrowsize": "0.2",
        },
    )

    # Input token nodes
    with dot.subgraph() as s:
        s.attr(rank="source")
        for pos, tok in enumerate(prompt_tokens):
            node_id = f"inp_{pos}"
            if node_id not in connected_io and not include_inactive_tokens:
                continue
            display = _pp_token(tok)
            if node_id in connected_io:
                fc, bc = "#333333", "#999999"
            else:
                fc, bc = "#bbbbbb", "#dddddd"
            s.node(
                node_id,
                label=display,
                fillcolor="#f0f0f0",
                color=bc,
                fontcolor=fc,
                style="filled,rounded",
                fontname="Courier",
            )
    # Chain visible tokens to maintain ordering
    visible_positions = [
        pos
        for pos in range(len(prompt_tokens))
        if f"inp_{pos}" in connected_io or include_inactive_tokens
    ]
    for i in range(len(visible_positions) - 1):
        dot.edge(
            f"inp_{visible_positions[i]}",
            f"inp_{visible_positions[i + 1]}",
            style="invis",
            weight="10",
        )

    # Output logit nodes (top)
    with dot.subgraph() as s:
        s.attr(rank="sink")
        for li, tok in enumerate(target_logit_tokens):
            node_id = f"out_{li}"
            display = _pp_token(tok)
            prob = target_logit_probs[li] if li < len(target_logit_probs) else 0.0
            label_html = (
                f"<{display}<BR/>" f'<FONT POINT-SIZE="3" COLOR="#888888">p={prob:.3f}</FONT>>'
            )
            s.node(
                node_id,
                label=label_html,
                fillcolor="#e8e8e8",
                color="#666666",
                fontcolor="#333333",
                style="filled,rounded",
                fontname="Courier",
            )
    for li in range(len(target_logit_tokens) - 1):
        dot.edge(f"out_{li}", f"out_{li + 1}", style="invis", weight="10")

    # Cluster supernode nodes
    def _wrap(text: str, width: int = 40) -> str:
        """Wrap text at word boundaries for graphviz labels."""
        words = text.split()
        lines: list[str] = []
        current = ""
        for w in words:
            if current and len(current) + 1 + len(w) > width:
                lines.append(current)
                current = w
            else:
                current = f"{current} {w}" if current else w
        if current:
            lines.append(current)
        return "\\n".join(lines)

    cluster_attr_descs = getattr(circuit, "_cluster_attr_descriptions", {})
    cluster_contrib_descs = getattr(circuit, "_cluster_contrib_descriptions", {})
    summary_labels = getattr(circuit, "_cluster_summary_labels", {})
    max_cluster_attr = max(cluster_attr.values()) if cluster_attr else 1.0
    for cl_name in cluster_names:
        color = cluster_color[cl_name]

        # Cluster metadata line
        n_neurons = cluster_neuron_count.get(cl_name, 0)
        meta_line = f"{cl_name}: {n_neurons} neurons"

        # Use summary label if available, otherwise full attr/contrib
        if cl_name in summary_labels:
            cl_label = (
                f"<{summary_labels[cl_name]}<BR/>"
                f'<FONT POINT-SIZE="3" COLOR="#888888">{meta_line}</FONT>>'
            )
        else:
            attr_desc = cluster_attr_descs.get(cl_name, "")
            contrib_desc = cluster_contrib_descs.get(cl_name, "")
            if not attr_desc and not contrib_desc:
                combined = cluster_descs.get(cl_name, cl_name)
                attr_desc = combined

            label_lines = []
            if attr_desc:
                label_lines.append(_wrap(f"attr: {attr_desc}"))
            if contrib_desc:
                label_lines.append(_wrap(f"contrib: {contrib_desc}"))
            text = "\\n\\n".join(label_lines) if label_lines else cl_name
            cl_label = f"{text}\\n{meta_line}"

        # Color by sign of attribution (blue=positive, red=negative)
        signed = cluster_attr_signed.get(cl_name, 0)
        attr_frac = cluster_attr.get(cl_name, 0) / max_cluster_attr
        if signed >= 0:
            fill = POS_EDGE_COLOR
        else:
            fill = NEG_EDGE_COLOR
        # Opacity scales with attribution magnitude
        alpha = max(0.4, min(1.0, 0.4 + 0.6 * attr_frac))
        hex_alpha = f"{int(alpha * 255):02x}"

        pw = str(max(0.5, 0.5 + 1.0 * attr_frac))
        dot.node(
            f"cl_{cl_name}",
            label=cl_label,
            fillcolor="white",
            color=fill,
            fontcolor=fill,
            shape="box",
            style="filled,rounded",
            margin="0.02,0.015",
            penwidth=pw,
        )

    # Edges
    if sorted_edges:
        max_w = max(abs(w) for _, w in sorted_edges)
        for (src_id, tgt_id), weight in sorted_edges:
            penwidth = str(max(0.5, 4.0 * abs(weight) / max_w))
            color = POS_EDGE_COLOR if weight > 0 else NEG_EDGE_COLOR
            alpha = max(0.4, abs(weight) / max_w)
            hex_alpha = f"{int(alpha * 255):02x}"
            dot.edge(src_id, tgt_id, penwidth=penwidth, color=color + hex_alpha)

    return dot


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot circuit graph for a single example.")
    parser.add_argument("circuit", type=Path, help="Path to circuit pickle")
    parser.add_argument("--label", type=str, required=True, help="CI index or label string")
    parser.add_argument("--output", type=str, default="", help="Output path (pdf/png/svg)")
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "png", "svg"])
    parser.add_argument("--top-n-edges", type=int, default=50, help="Max aggregated edges (0=all)")
    parser.add_argument(
        "--cluster-state",
        type=str,
        default="",
        help="Cluster state JSON from save_cluster_state",
    )
    parser.add_argument(
        "--descriptions-json",
        type=str,
        default="",
        help="Explanations JSON to load cluster descriptions from",
    )
    parser.add_argument(
        "--no-guarantee-edges",
        action="store_true",
        help="Disable guaranteed best incoming/outgoing edge per cluster",
    )
    parser.add_argument(
        "--include-inactive-tokens",
        action="store_true",
        help="Show input tokens that have no edges (inactive tokens)",
    )
    parser.add_argument(
        "--dot-only",
        action="store_true",
        help="Save .dot source file only (no rendering, no graphviz binary needed)",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Generate short summary labels via Anthropic API (requires ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model ID for tokenizer and layer count",
    )
    args = parser.parse_args()

    logger.info("Loading circuit from %s", args.circuit)
    circuit = Circuit.load_from_pickle(str(args.circuit))

    logger.info("Loading tokenizer...")
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(args.model_id)
    num_layers = cfg.num_hidden_layers
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    circuit.set_tokenizer(tokenizer, num_layers=num_layers)

    # Load cluster state if provided
    if args.cluster_state:
        circuit.load_cluster_state(args.cluster_state)

    # Load cluster descriptions from explanations JSON if provided
    if args.descriptions_json:
        import json

        with open(args.descriptions_json) as f:
            desc_data = json.load(f)
        # Build separate attr and contrib descriptions per cluster
        attr_descs: dict[str, str] = {}
        contrib_descs: dict[str, str] = {}
        combined_descs: dict[str, str] = {}
        for cluster_name in desc_data.get("attr", {}):
            attr_expls = desc_data["attr"][cluster_name]
            contrib_expls = desc_data.get("contrib", {}).get(cluster_name, {})

            # Best attr explanation
            best_attr = ""
            best_attr_score = -1.0
            for sign in ("pos", "neg"):
                for e in attr_expls.get(sign, []):
                    s = e.get("score")
                    if s is not None and s > best_attr_score:
                        best_attr_score = s
                        best_attr = e["explanation"]

            # Best contrib explanation
            best_contrib = ""
            best_contrib_score = -1.0
            for sign in ("pos", "neg", "combined"):
                for e in contrib_expls.get(sign, []):
                    s = e.get("score")
                    if s is not None and s > best_contrib_score:
                        best_contrib_score = s
                        best_contrib = e["explanation"]

            if best_attr:
                attr_descs[cluster_name] = best_attr
            if best_contrib:
                contrib_descs[cluster_name] = best_contrib
            combined_descs[cluster_name] = " | ".join(filter(None, [best_attr, best_contrib]))

        circuit._cluster_attr_descriptions = attr_descs
        circuit._cluster_contrib_descriptions = contrib_descs
        circuit._cluster_descriptions = combined_descs
        logger.info("Loaded descriptions for %d clusters", len(combined_descs))

        # Generate short summary labels via API
        if args.summarize:
            import asyncio

            from circuits.analysis.label import summarize_attr_contrib_descriptions

            cluster_desc_input = {
                name: {"attr": attr_descs.get(name, ""), "contrib": contrib_descs.get(name, "")}
                for name in combined_descs
            }
            logger.info("Generating summary labels for %d clusters...", len(cluster_desc_input))
            summary_labels = asyncio.run(summarize_attr_contrib_descriptions(cluster_desc_input))
            circuit._cluster_summary_labels = summary_labels
            for name, lbl in sorted(summary_labels.items()):
                logger.info("  %s -> %s", name, lbl)

    label, ci_idx = resolve_label(circuit, args.label)
    logger.info("Plotting label=%s (ci_idx=%d)", label, ci_idx)

    dot = build_graph(
        circuit,
        label,
        ci_idx,
        top_n_edges=args.top_n_edges,
        guarantee_edges=not args.no_guarantee_edges,
        include_inactive_tokens=args.include_inactive_tokens,
    )

    # Output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = Path(args.output)
    else:
        safe_label = label.replace("/", "_").replace(" ", "_")[:40]
        out_path = OUTPUT_DIR / f"circuit_graph_{safe_label}"

    if args.dot_only:
        dot_path = str(out_path) + ".dot"
        dot.save(dot_path)
        logger.info("Saved DOT source to %s", dot_path)
        print(dot_path)
    else:
        # graphviz render writes <path>.<format>
        rendered = dot.render(
            str(out_path),
            format=args.format,
            cleanup=True,
        )
        logger.info("Saved to %s", rendered)
        print(rendered)


if __name__ == "__main__":
    main()

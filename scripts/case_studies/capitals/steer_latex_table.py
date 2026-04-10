"""Steer each cluster in a circuit (multiplier=0 ablation) and print top-5 output probs as LaTeX.

Usage:
    python scripts/case_studies/capitals/steer_latex_table.py \
        results/case_studies/capitals_circuit.pkl \
        --cluster-state results/case_studies/capitals/cluster_state_k64_harmonic.json \
        --labels 0 5 10
"""

import argparse
import logging

import torch
from circuits.analysis.circuit_ops import Circuit
from circuits.analysis.cluster import prepare_circuit_data
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Set2 colors (8 colors) as RGB hex
SET2_COLORS = [
    "102,194,165",  # teal
    "252,141,98",  # salmon
    "141,160,203",  # periwinkle
    "231,138,195",  # pink
    "166,216,84",  # lime
    "255,217,47",  # yellow
    "229,196,148",  # tan
    "179,179,179",  # gray
]


def _escape_latex(s: str) -> str:
    """Escape special LaTeX characters."""
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def _color_cell(tok: str, prob: float, color_idx: int) -> str:
    """Format a token cell with Set2 background color (20% tint)."""
    safe_tok = _escape_latex(tok.strip())
    base = SET2_COLORS[color_idx % len(SET2_COLORS)]
    r, g, b = [int(x) for x in base.split(",")]
    r_t = int(r * 0.2 + 255 * 0.8)
    g_t = int(g * 0.2 + 255 * 0.8)
    b_t = int(b * 0.2 + 255 * 0.8)
    prob_str = f"{prob:.1%}".replace("%", r"\%")
    return rf"\cellcolor[RGB]{{{r_t},{g_t},{b_t}}}\texttt{{{safe_tok}}} ({prob_str})"


def _sort_key(x: str) -> int:
    """Sort cluster IDs numerically, handling both '0' and 'C0' formats."""
    stripped = x.lstrip("C")
    try:
        return int(stripped)
    except ValueError:
        return 999999


def build_table(
    cto_label: "pd.DataFrame",
    c: Circuit,
    label_name: str,
    k: int,
    token_to_color: dict[str, int],
    multiplier: float = 0.0,
) -> str:
    """Build a LaTeX table for one label's ablation results."""
    summary_labels = getattr(c, "_cluster_summary_labels", {})

    header_cols = " & ".join([f"Token {i+1}" for i in range(k)])
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{l{'l' * k}}}",
        r"\toprule",
        rf"Cluster & {header_cols} \\",
        r"\midrule",
    ]

    for _, row in cto_label.iterrows():
        cl = row["cluster"]
        cl_id = cl.lstrip("C") if cl.startswith("C") else cl
        canon = f"C{cl_id}"
        summary = summary_labels.get(canon, "")
        cl_name = f"{canon}: {summary}" if summary else canon
        top_toks = row["top_tokens"][:k]
        top_probs = row["top_tokens_probs"][:k]
        tok_cells = " & ".join(
            [
                _color_cell(tok, prob, token_to_color[tok.strip()])
                for tok, prob in zip(top_toks, top_probs)
            ]
        )
        safe_name = _escape_latex(cl_name)
        lines.append(rf"{safe_name} & {tok_cells} \\")

    safe_label = _escape_latex(label_name.strip())
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{Steering results for \textbf{{{safe_label}}} (multiplier$={multiplier}$).}}",
            rf"\label{{tab:ablation-{label_name.strip().lower().replace(' ', '-')}}}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("circuit", type=str, help="Path to circuit pickle")
    parser.add_argument("--cluster-state", type=str, required=True, help="Cluster state JSON")
    parser.add_argument(
        "--labels", type=str, nargs="+", default=["0"], help="CI indices or label strings"
    )
    parser.add_argument("--multiplier", type=float, default=0.0, help="Steering multiplier")
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    logger.info("Loading circuit from %s", args.circuit)
    c = Circuit.load_from_pickle(args.circuit)
    cfg = AutoConfig.from_pretrained(args.model_id)
    t = AutoTokenizer.from_pretrained(args.model_id)
    c.set_tokenizer(t, num_layers=cfg.num_hidden_layers)

    import pandas as pd

    # Load cluster state
    c.load_cluster_state(args.cluster_state)

    # Filter to target labels only (massive speedup for large circuits)
    target_ci_indices = []
    target_labels = []
    for label_arg in args.labels:
        try:
            idx = int(label_arg)
            lbl = c.labels[idx]
        except ValueError:
            lbl = label_arg
            idx = next(i for i, l in enumerate(c.labels) if lbl.strip() in str(l))
        target_ci_indices.append(idx)
        target_labels.append(lbl)

    # Filter df_node to target labels, relabel ___N to sequential indices
    old_to_new = {}
    filtered_dfs = []
    for new_idx, (ci_idx, lbl) in enumerate(zip(target_ci_indices, target_labels)):
        old_label = f"{lbl}___{ci_idx}"
        new_label = f"{lbl}___{new_idx}"
        old_to_new[old_label] = new_label
        df_sub = c.df_node[c.df_node["label"] == old_label].copy()
        df_sub["label"] = new_label
        filtered_dfs.append(df_sub)

    n_before = len(c.df_node)
    c.df_node = pd.concat(filtered_dfs, ignore_index=True) if filtered_dfs else c.df_node.head(0)
    c.df_edge = pd.DataFrame(columns=c.df_edge.columns)
    c.labels = target_labels
    c.cis = [c.cis[i] for i in target_ci_indices]
    c.attention_masks = (
        [c.attention_masks[i] for i in target_ci_indices] if c.attention_masks else []
    )
    c.target_logits = [c.target_logits[i] for i in target_ci_indices] if c.target_logits else []
    logger.info(
        "Filtered to %d labels: %d -> %d nodes", len(target_labels), n_before, len(c.df_node)
    )

    # Ensure input_variable column exists
    if "input_variable" not in c.df_node.columns:
        logger.info("Preparing circuit data (adding input_variable)...")
        c.df_node, c.df_edge = prepare_circuit_data(c.df_node, c.df_edge)

    logger.info("Loading model %s...", args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    logger.info("Steering at multiplier=%.1f...", args.multiplier)
    c.steer(model, multiplier=args.multiplier, verbose=True, store_results=True)

    cto_all = c.cluster_to_output

    # Skip "none", "all" meta-clusters, and unclustered output neurons (NeuronId entries)
    cto_all = cto_all[~cto_all["cluster"].isin(["none", "all"])]
    cto_all = cto_all[~cto_all["cluster"].str.contains("NeuronId")]

    # Build global token-to-color map across all labels
    token_to_color: dict[str, int] = {}
    color_counter = 0
    for _, row in cto_all.iterrows():
        for tok in row["top_tokens"][: args.top_k]:
            key = tok.strip()
            if key not in token_to_color:
                token_to_color[key] = color_counter
                color_counter += 1

    # Generate a table per label (use target_labels which are already resolved)
    for lbl in target_labels:
        cto_label = cto_all[cto_all["label"].str.contains(lbl.strip(), regex=False)]
        cto_label = cto_label.sort_values("cluster", key=lambda s: s.apply(_sort_key))

        table = build_table(cto_label, c, lbl, args.top_k, token_to_color, args.multiplier)
        print("\n" + table + "\n")


if __name__ == "__main__":
    main()

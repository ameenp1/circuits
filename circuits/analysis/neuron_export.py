"""
Export top-N MLP neurons *with their associated text* from a traced circuit, in a clean
shape for external description / feature-grouping pipelines.

For each neuron this surfaces, per prompt where it fired:
  - the prompt tokens and the neuron's per-token **input attribution** (`attr_map`,
    normalized) — the text that drives it,
  - a ``{{highlighted}}`` exemplar string (ADAG's own formatting),
  - the top input tokens (token, score),
  - the **output contributions** (`contrib_map`) over the traced output logits — the tokens
    it promotes / suppresses,
plus the raw `attr_map` / `contrib_map` vectors for use as grouping features.

This reuses ADAG's own representation end-to-end:
    Circuit                       (normalizes attr_map / contrib_map)
      -> prepare_circuit_data(sum_over_tokens=True)   (NeuronId keys, maps summed over tokens)
      -> build_neuron_activation_records              (per-neuron ActivationRecords)
      -> build_attr_exemplars                         (highlighted exemplar text)

Entry points:
  - export_top_neurons_from_circuit  : aggregate across all prompts -> one ranked list
  - export_per_prompt_from_circuit   : per prompt (per graph) -> one ranked list each
  - export_top_neurons               : trace prompts then aggregate-export (convenience)
"""

import numpy as np
from circuits.analysis.circuit_ops import Circuit
from circuits.analysis.cluster import prepare_circuit_data
from circuits.descriptions.exemplars import build_attr_exemplars
from circuits.descriptions.label import build_neuron_activation_records
from circuits.descriptions.types import ActivationRecordWithContrib
from circuits.tracing.clja import ADAGConfig
from circuits.tracing.trace import CircuitData, convert_inputs_to_circuits


def _decode_output_contributions(
    record: ActivationRecordWithContrib, tokenizer, top_k: int
) -> list[tuple[str, float]]:
    """Decode the neuron's output-logit contributions into (token_str, value) pairs."""
    logits = record.output_logits
    contrib = record.contrib_map
    if not logits or contrib is None:
        return []
    pairs = [(tokenizer.decode([tid]), float(c)) for tid, c in zip(logits, contrib)]
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    return pairs[:top_k]


def _top_input_tokens(
    record: ActivationRecordWithContrib, top_k: int
) -> list[tuple[str, float]]:
    """Top input tokens by |attribution| (the text that drives the neuron)."""
    pairs = [
        (tok, float(act)) for tok, act in zip(record.tokens, record.activations) if act != 0.0
    ]
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    return pairs[:top_k]


def _record_to_dict(
    record: ActivationRecordWithContrib,
    polarity: str,
    tokenizer,
    top_tokens: int,
) -> dict:
    """Convert one neuron-example record into a JSON-serializable text bundle."""
    act_sign = "pos" if polarity == "+" else "neg"
    try:
        formatted, _dicts = build_attr_exemplars([record], act_sign=act_sign, max_exemplars=1)
        highlighted = formatted[0] if formatted else "".join(record.tokens)
    except Exception:
        highlighted = "".join(record.tokens)

    return {
        "tokens": record.tokens,
        "attr_activations": [float(a) for a in record.activations],
        "highlighted_text": highlighted,
        "top_input_tokens": _top_input_tokens(record, top_tokens),
        "output_contributions": _decode_output_contributions(record, tokenizer, top_tokens),
        "raw_contrib_map": (
            [float(c) for c in record.contrib_map] if record.contrib_map is not None else None
        ),
    }


def _parse_edges_for_label(df_edge, label: str) -> list[dict]:
    """
    Extract the attribution-graph edges for one prompt (label) from the raw df_edge.

    df_edge stores each edge with "src->tgt" strings in the layer/token/neuron columns plus
    a normalized `attribution` and the raw jacobian `weight`. A source neuron with layer -1
    is the token embedding. Returns one dict per edge.
    """
    if df_edge is None or len(df_edge) == 0:
        return []
    sub = df_edge[df_edge["label"] == label]
    edges: list[dict] = []
    for _, row in sub.iterrows():
        try:
            src_layer, tgt_layer = str(row["layer"]).split("->")
            src_token, tgt_token = str(row["token"]).split("->")
            src_neuron, tgt_neuron = str(row["neuron"]).split("->")
        except ValueError:
            continue
        edges.append(
            {
                "src": {
                    "layer": int(src_layer),
                    "token": int(src_token),
                    "neuron": int(src_neuron),
                },
                "tgt": {
                    "layer": int(tgt_layer),
                    "token": int(tgt_token),
                    "neuron": int(tgt_neuron),
                },
                "attribution": float(row["attribution"]),
                "weight": float(row["weight"]),
            }
        )
    return edges


def _raw_nodes_for_label(df_node, label: str) -> list[dict]:
    """
    All surviving graph nodes for one prompt (per token position), from the raw df_node.

    These are the graph's nodes (lightweight: no attr/contrib maps) that the `edges` connect.
    `layer == -1` is the token embedding; the top (final) layer is the logit node. Each node
    keeps its token position so it lines up with edge endpoints.
    """
    if df_node is None or len(df_node) == 0:
        return []
    sub = df_node[df_node["label"] == label]
    nodes: list[dict] = []
    for _, row in sub.iterrows():
        nodes.append(
            {
                "layer": int(row["layer"]),
                "token": int(row["token"]),
                "neuron": int(row["neuron"]),
                "attribution": float(row["attribution"]),
                "activation": float(row["activation"]),
            }
        )
    return nodes


def _prepare_circuit(circuit: Circuit, tokenizer, num_layers: int | None):
    """Shared prep: add NeuronId columns, sum maps over tokens, build neuron records.

    Returns (df_node, neuron_data, neuron_ci_mapping):
      - df_node: one row per (neuron, prompt-label) with `input_variable`, `attribution`.
      - neuron_data: dict[NeuronId, list[record]] — records across all prompts.
      - neuron_ci_mapping: dict[NeuronId, dict[ci_idx, record]] — record per prompt.
    """
    num_layers = num_layers if num_layers is not None else circuit.num_layers
    df_node, _df_edge = prepare_circuit_data(
        circuit.df_node.copy(),
        circuit.df_edge.copy(),
        sum_over_tokens=True,
        _suppress_warning=True,
    )
    neuron_data, neuron_ci_mapping = build_neuron_activation_records(
        df_node, circuit.cis, tokenizer, circuit.target_logits, num_layers
    )
    return df_node, neuron_data, neuron_ci_mapping


def export_top_neurons_from_circuit(
    circuit: Circuit,
    tokenizer,
    top_n: int = 30,
    top_tokens: int = 8,
    num_layers: int | None = None,
) -> list[dict]:
    """
    Aggregate across all prompts: rank MLP neurons by summed |attribution| and return the
    top `top_n`, each with the text that drives it and the tokens it promotes.

    Returns a list of dicts (one per (layer, neuron, polarity)) ready for json.dump.
    """
    df_node, neuron_data, _ci_mapping = _prepare_circuit(circuit, tokenizer, num_layers)

    scores: dict = {}
    for nid, sub in df_node.groupby("input_variable"):
        scores[nid] = float(np.abs(np.asarray(sub["attribution"], dtype=np.float64)).sum())

    ranked = sorted(
        (nid for nid in neuron_data if nid in scores),
        key=lambda nid: scores[nid],
        reverse=True,
    )[:top_n]

    results: list[dict] = []
    for rank, nid in enumerate(ranked, 1):
        records = neuron_data[nid]
        results.append(
            {
                "rank": rank,
                "layer": int(nid.layer),
                "neuron": int(nid.neuron),
                "polarity": nid.polarity,
                "attribution": scores[nid],
                "num_examples": len(records),
                "examples": [
                    _record_to_dict(rec, nid.polarity, tokenizer, top_tokens) for rec in records
                ],
            }
        )
    return results


def export_per_prompt_from_circuit(
    circuit: Circuit,
    tokenizer,
    top_n: int = 30,
    top_tokens: int = 8,
    num_layers: int | None = None,
    prompts: list[str] | None = None,
    targets: list[str] | None = None,
    include_edges: bool = True,
    include_all_nodes: bool = True,
) -> list[dict]:
    """
    Per-prompt (per attribution graph): for each traced prompt, return its own top-`top_n`
    neurons ranked by |attribution| on that prompt (each with the text bundle), the prompt's
    full attribution-graph `nodes` and `edges` (the pruned graph), so each entry is a complete
    self-contained graph.

    Returns one dict per prompt:
        {ci_idx, label, prompt, target, neurons: [...], nodes: [...], edges: [...]}
    ordered by ci_idx (matching the order prompts were traced in). Pass `include_edges=False`
    / `include_all_nodes=False` to omit those.

    Granularity: `neurons` (the curated features + text) are summed over token positions
    (keyed by layer+neuron); `nodes`/`edges` (the raw graph) keep token positions (carry
    layer, token, neuron). Join a node/edge endpoint to a feature by (layer, neuron).
    """
    df_node, _neuron_data, ci_mapping = _prepare_circuit(circuit, tokenizer, num_layers)

    per_prompt: list[dict] = []
    for label, sub in df_node.groupby("label"):
        try:
            ci_idx = int(str(label).split("___")[1])
        except (IndexError, ValueError):
            continue
        sub = sub.assign(_absattr=sub["attribution"].abs()).sort_values(
            "_absattr", ascending=False
        )

        neurons: list[dict] = []
        for _, row in sub.iterrows():
            nid = row["input_variable"]
            record = ci_mapping.get(nid, {}).get(ci_idx)
            if record is None:  # embedding / unembedding nodes are skipped
                continue
            entry = {
                "rank": len(neurons) + 1,
                "layer": int(nid.layer),
                "neuron": int(nid.neuron),
                "polarity": nid.polarity,
                "attribution": float(row["attribution"]),
            }
            entry.update(_record_to_dict(record, nid.polarity, tokenizer, top_tokens))
            neurons.append(entry)
            if len(neurons) >= top_n:
                break

        entry = {
            "ci_idx": ci_idx,
            "label": str(label),
            "prompt": prompts[ci_idx] if prompts and ci_idx < len(prompts) else None,
            "target": targets[ci_idx] if targets and ci_idx < len(targets) else None,
            "neurons": neurons,
        }
        if include_all_nodes:
            entry["nodes"] = _raw_nodes_for_label(circuit.df_node, label)
        if include_edges:
            entry["edges"] = _parse_edges_for_label(circuit.df_edge, label)
        per_prompt.append(entry)

    per_prompt.sort(key=lambda d: d["ci_idx"])
    return per_prompt


def export_top_neurons(
    model,
    tokenizer,
    prompts: list[str],
    labels: list[str],
    seed_responses: list[str] | None = None,
    targets: list[list[str]] | None = None,
    top_n: int = 30,
    top_tokens: int = 8,
    percentage_threshold: float = 0.01,
    k: int = 5,
    batch_size: int = 4,
    device: str = "cuda:0",
    per_prompt: bool = False,
) -> list[dict]:
    """
    Trace `prompts` on `model`, then export top-N neurons with text.

    `targets` is per-prompt: a list (one entry per prompt) of lists of acceptable answer
    strings to trace toward, e.g. ``[[" Austin"]]``; `k` is taken from the answer-list length.
    When `targets` is None, the model's own top-`k` prediction is traced instead.

    Set `per_prompt=True` for one ranked list per prompt; otherwise one aggregate list.
    """
    if targets is not None:
        k = len(targets[0])
    config = ADAGConfig(
        device=device,
        use_relp_grad=True,
        percentage_threshold=percentage_threshold,
        apply_blacklist=False,
    )
    data: CircuitData = convert_inputs_to_circuits(
        model,
        tokenizer,
        prompts,
        config=config,
        seed_responses=seed_responses or ["Answer:"] * len(prompts),
        labels=labels,
        true_answers=targets,
        batch_size=batch_size,
        k=k,
    )
    circuit = Circuit(data, tokenizer=tokenizer, num_layers=model.config.num_hidden_layers)
    if per_prompt:
        return export_per_prompt_from_circuit(
            circuit,
            tokenizer,
            top_n=top_n,
            top_tokens=top_tokens,
            prompts=prompts,
            targets=[t[0] if t else None for t in targets] if targets else None,
        )
    return export_top_neurons_from_circuit(circuit, tokenizer, top_n=top_n, top_tokens=top_tokens)

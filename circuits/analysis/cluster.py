"""
Utilities for clustering neurons in a circuit.
"""

import json
import logging
import os
from typing import Any, Literal, Mapping, NamedTuple, cast

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from numpy.typing import NDArray

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

NODE_COLUMNS = [
    "layer",
    "token",
    "neuron",
    "label",
    "attr_map",
    "contrib_map",
    "attribution",
    "activation",
]
EDGE_COLUMNS = [
    "layer",
    "token",
    "neuron",
    "label",
    "attribution",
    "weight",
]

EMBEDDING_MODES = Literal["attr", "contrib", "attr + contrib", "attr x contrib", "random"]

UNCLUSTERED_CLUSTER_ID = "__unclustered__"


class NeuronId(NamedTuple):
    """Unique identifier for a neuron: (layer, token, neuron, polarity)."""

    layer: int
    token: int
    neuron: int
    polarity: Literal["+", "-"]

    def to_string(self) -> str:
        """Return 'layer,token,neuron' string (polarity excluded)."""
        return f"{self.layer},{self.token},{self.neuron}"

    @classmethod
    def from_string(cls, s: str, polarity: Literal["+", "-"] = "+") -> "NeuronId":
        """Parse 'layer,token,neuron' string into NeuronId."""
        parts = s.split(",")
        if len(parts) != 3:
            raise ValueError(f"Expected 'layer,token,neuron' format, got: {s}")
        return cls(int(parts[0]), int(parts[1]), int(parts[2]), polarity)


class EdgeId(NamedTuple):
    """Edge between two neurons: (source, target)."""

    source: NeuronId
    target: NeuronId


def df_sum_over_tokens(
    df_node: pd.DataFrame,
    df_edge: pd.DataFrame,
    include_polarity: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sum attr_map/contrib_map over token positions, setting token=-1."""
    groupby_cols = ["layer", "neuron", "label"]
    if include_polarity:
        groupby_cols.insert(2, "polarity")  # ["layer", "neuron", "polarity", "label"]

    df_node = (
        df_node.groupby(groupby_cols)
        .agg(
            {
                "attr_map": "sum",
                "contrib_map": "sum",
                "activation": "mean",
                "attribution": "sum",
            }
        )
        .reset_index()
    )
    if len(df_edge) > 0:
        df_edge = (
            df_edge.groupby(groupby_cols)
            .agg({"attribution": "sum", "weight": "mean"})
            .reset_index()
        )

    # clean up token column
    df_node.loc[:, "token"] = -1
    if len(df_edge) > 0:
        df_edge.loc[:, "token"] = "-1->-1"

    # return
    return df_node, df_edge


def prepare_circuit_data(
    df_node: pd.DataFrame,
    df_edge: pd.DataFrame,
    sum_over_tokens: bool = False,
    verbose: bool = False,
    _suppress_warning: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add NeuronId/EdgeId index columns and optionally sum over tokens."""
    # check columns match
    assert len(df_node) > 0, "df_node must have at least one row"
    if set(df_node.columns) != set(NODE_COLUMNS):
        raise ValueError(
            f"df_node columns do not match: {set(df_node.columns)} != {set(NODE_COLUMNS)}"
        )
    if len(df_edge) > 0 and set(df_edge.columns) != set(EDGE_COLUMNS):
        raise ValueError(
            f"df_edge columns do not match: {set(df_edge.columns)} != {set(EDGE_COLUMNS)}"
        )

    # convert maps/attributions to numpy arrays so downstream math is consistent
    def _to_numpy(value: Any) -> np.ndarray[Any, np.dtype[Any]]:
        """
        Ensure map/attribution entries are numpy arrays.
        """
        if isinstance(value, np.ndarray):
            return value  # type: ignore
        # Treat scalars as length-1 arrays to keep shapes predictable
        return np.asarray(value if isinstance(value, (list, tuple)) else [value], dtype=np.float32)

    df_node["polarity"] = df_node["activation"].apply(lambda x: "+" if x >= 0 else "-")
    df_node["attr_map"] = cast(list[NDArray[np.float32]], df_node["attr_map"].apply(_to_numpy))
    df_node["contrib_map"] = cast(
        list[NDArray[np.float32]], df_node["contrib_map"].apply(_to_numpy)
    )

    # compute polarity for edges
    layer_token_neuron_to_polarity = cast(
        Mapping[tuple[int, int, int], Literal["+", "-"]],
        df_node.groupby(["layer", "token", "neuron"])["polarity"].first().to_dict(),
    )
    if len(df_edge) > 0:
        df_edge["polarity"] = df_edge.apply(
            lambda x: layer_token_neuron_to_polarity.get(
                (x["layer"].split("->")[0], x["token"].split("->")[0], x["neuron"].split("->")[0]),
                "+",
            )
            + "->"
            + layer_token_neuron_to_polarity.get(
                (x["layer"].split("->")[1], x["token"].split("->")[1], x["neuron"].split("->")[1]),
                "+",
            ),
            axis=1,
        )

    # sum over tokens
    if sum_over_tokens:
        df_node, df_edge = df_sum_over_tokens(df_node, df_edge)

    # convert various neuron/edge identifier columns to single types
    df_node["input_variable"] = df_node.apply(
        lambda x: NeuronId(
            layer=x["layer"], token=x["token"], neuron=x["neuron"], polarity=x["polarity"]
        ),
        axis=1,
    )

    # for edges, layer is source->target, same for token and neuron
    if len(df_edge) > 0:
        df_edge["input_variable"] = df_edge.apply(
            lambda x: EdgeId(
                source=NeuronId(
                    layer=x["layer"].split("->")[0],
                    token=x["token"].split("->")[0],
                    neuron=x["neuron"].split("->")[0],
                    polarity=x["polarity"].split("->")[0],
                ),
                target=NeuronId(
                    layer=x["layer"].split("->")[1],
                    token=x["token"].split("->")[1],
                    neuron=x["neuron"].split("->")[1],
                    polarity=x["polarity"].split("->")[1],
                ),
            ),
            axis=1,
        )

    # drop unnecessary columns
    df_node = df_node.drop(columns=["layer", "token", "neuron", "polarity"])
    if len(df_edge) > 0:
        df_edge = df_edge.drop(columns=["layer", "token", "neuron", "polarity"])

    if verbose:
        logger.info(
            "prepare_circuit_data: prepared %d node rows and %d edge rows",
            len(df_node),
            len(df_edge),
        )

    # return
    return df_node, df_edge


def unit_norm(x: NDArray[np.float32], eps: float = 1e-5) -> NDArray[np.float32]:
    """L2-normalize vectors to unit length."""
    # Handle 0-dimensional arrays (scalars) - can happen when k=1
    if x.ndim == 0:
        return x / (np.abs(x) + eps)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def compute_embeddings(
    df_node: pd.DataFrame,
    mode: EMBEDDING_MODES,
    do_unit_norm: bool = False,
) -> NDArray[np.float32]:
    """Compute embedding vectors from attr_map/contrib_map based on mode."""
    attr_map = cast(list[NDArray[np.float32]], df_node["attr_map"].to_list())
    contrib_map = cast(list[NDArray[np.float32]], df_node["contrib_map"].to_list())

    # Ensure all arrays are at least 1D (can be 0D scalars when k=1)
    attr_map = [np.atleast_1d(x) for x in attr_map]
    contrib_map = [np.atleast_1d(x) for x in contrib_map]

    # normalize attribution maps per-row; guard against all-zero rows
    attr_row_sums = cast(
        list[NDArray[np.float32]], [x.sum(axis=-1, keepdims=True) for x in attr_map]
    )
    attr_map = [x / np.where(y == 0, 1.0, y) for x, y in zip(attr_map, attr_row_sums)]

    # unit normalize
    if do_unit_norm:
        attr_map = [unit_norm(x) for x in attr_map]
        contrib_map = [unit_norm(x) for x in contrib_map]

    # compute embeddings
    if mode == "attr":
        return attr_map
    elif mode == "contrib":
        return contrib_map
    elif mode == "attr + contrib":
        return [np.concatenate([x, y], axis=-1) for x, y in zip(attr_map, contrib_map)]
    elif mode == "attr x contrib":
        return [np.outer(x, y) for x, y in zip(attr_map, contrib_map)]
    elif mode == "random":
        return [
            np.random.randn(x.shape[0] + y.shape[0]).astype(np.float32)
            for x, y in zip(attr_map, contrib_map)
        ]


def export_neuron_graph(data: dict, out_html: str):
    """Export visualization data to standalone HTML file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    visualization_path = os.path.join(current_dir, "visualization.html")
    with open(visualization_path, "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace("__GRAPH_JSON__", json.dumps(data))
    html = html.replace("__DEFAULT_METRIC__", "Average")

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {out_html} with {len(data['nodes'])} nodes and {len(data['links'])} edges.")

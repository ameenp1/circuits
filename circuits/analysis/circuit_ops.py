import asyncio
import json
import logging
import random
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from circuits.analysis import label
from circuits.analysis.cluster import (
    EDGE_COLUMNS,
    EMBEDDING_MODES,
    NODE_COLUMNS,
    UNCLUSTERED_CLUSTER_ID,
    NeuronId,
    compute_embeddings,
    df_sum_over_tokens,
    prepare_circuit_data,
)
from circuits.analysis.multiview_cluster import compute_per_ci_similarities, spectral_cluster
from circuits.analysis.steer import export_cluster_data_to_json, get_cluster_steering_effects
from circuits.tracing.trace import CircuitData
from circuits.utils.descriptions import get_descriptions
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger(__name__)


class Circuit:
    def __init__(
        self,
        data: CircuitData,
        tokenizer: Any | None = None,
        num_layers: int | None = None,
    ):
        self.circuit_data = data
        self.tokenizer = tokenizer
        self.model_id = data.model_id or ""
        self.df_node = data.df_node
        self.df_edge = data.df_edge
        self.cis = data.cis
        self.attention_masks = data.attention_masks
        self.labels = data.labels
        self.k = data.k
        self.target_logits = data.target_logits

        # Infer num_layers from df_node if not provided
        if num_layers is not None:
            self.num_layers = num_layers
        elif self.df_node is not None and len(self.df_node) > 0:
            self.num_layers = int(self.df_node["layer"].max())
        else:
            self.num_layers = None

        self._normalize_attr_contrib_maps()

        self.neuron_label_cache: dict = {}
        self._df_node_summed_cache = None
        self._cluster_map: dict[NeuronId, str] = {}
        self._cluster_descriptions: dict[str, str] = {}
        self._cluster_id_to_name: dict[int, str] = {}

        # steering results
        self.diversity_stats = pd.DataFrame()
        self.cluster_to_cluster = pd.DataFrame()
        self.cluster_to_output = pd.DataFrame()

        # feature scoring results
        self.hypothesis_to_scores: dict[str, pd.DataFrame] = {}

    @property
    def _manual_clusters_map(self) -> dict[NeuronId, str]:
        """Backward-compatible alias for _cluster_map."""
        return self._cluster_map

    @_manual_clusters_map.setter
    def _manual_clusters_map(self, value: dict[NeuronId, str]) -> None:
        self._cluster_map = value

    def _normalize_attr_contrib_maps(self) -> None:
        """Normalize raw attr_map and contrib_map stored in CircuitData.

        - attr_map: (grad * embedding) / activation → relative input-token contributions (sum ≈ 1)
        - contrib_map: (grad * activation) / logit_value → relative output-logit contributions
        """
        if "attr_map" not in self.df_node.columns or "activation" not in self.df_node.columns:
            return

        eps = 1e-10

        # attr_map: divide by neuron activation
        if "attr_map" in self.df_node.columns:
            mask = self.df_node["attr_map"].notna() & (self.df_node["activation"].abs() > eps)
            self.df_node.loc[mask, "attr_map"] = self.df_node.loc[mask].apply(
                lambda r: np.asarray(r["attr_map"]) / r["activation"], axis=1
            )

        # contrib_map: divide by target logit values per label
        if "contrib_map" not in self.df_node.columns:
            return
        logit_layer = self.num_layers
        logit_nodes = self.df_node[self.df_node["layer"] == logit_layer]
        logit_activations: dict[str, np.ndarray] = {}
        for lbl in logit_nodes["label"].unique():
            vals = logit_nodes[logit_nodes["label"] == lbl].sort_values("token")["activation"]
            arr = np.asarray(vals.tolist())
            logit_activations[lbl] = np.where(np.abs(arr) > eps, arr, 1.0)

        mask = self.df_node["contrib_map"].notna()
        self.df_node.loc[mask, "contrib_map"] = self.df_node.loc[mask].apply(
            lambda r: (
                np.asarray(r["contrib_map"]) / logit_activations[r["label"]]
                if r["label"] in logit_activations
                and len(r["contrib_map"]) == len(logit_activations[r["label"]])
                else r["contrib_map"]
            ),
            axis=1,
        )

    #########################################################
    # CLUSTERING
    #########################################################

    def _build_cluster_map_from_summed(
        self,
        summed_cluster_map: dict[NeuronId, str],
    ) -> dict[NeuronId, str]:
        """Expand a summed (token=-1) cluster map to cover all raw neurons.

        For each row in df_node, look up the cluster by (layer, neuron, polarity)
        from the summed map, and assign it to the full (layer, token, neuron, polarity)
        NeuronId.
        """
        cluster_map: dict[NeuronId, str] = {}
        last_layer = self.num_layers or 0

        # Build polarity lookup from df_node
        df = self.df_node
        if "polarity" not in df.columns:
            df = df.copy()
            df["polarity"] = df["activation"].apply(lambda x: "+" if x >= 0 else "-")

        for _, row in df.iterrows():
            layer = int(row["layer"])
            token = int(row["token"])
            neuron = int(row["neuron"])
            polarity = row["polarity"]

            nid = NeuronId(layer=layer, token=token, neuron=neuron, polarity=polarity)

            # Look up in summed map (token=-1)
            summed_nid = NeuronId(layer=layer, token=-1, neuron=neuron, polarity=polarity)
            if summed_nid in summed_cluster_map:
                cluster_map[nid] = summed_cluster_map[summed_nid]
            elif layer in (-1, last_layer):
                cluster_map[nid] = str(nid)
            else:
                cluster_map[nid] = UNCLUSTERED_CLUSTER_ID

        return cluster_map

    def cluster(
        self,
        n_clusters: int,
        mode: EMBEDDING_MODES = "attr + contrib",
        do_layernorm: bool = True,
        cluster_linkage: str = "ward",
        cluster_metric: str = "euclidean",
        do_average_over_examples: bool = False,
        sum_over_tokens: bool = True,
        get_desc: bool = True,
        include_attr_contrib: bool = True,
        do_one_cluster_per_neuron: bool = False,
        manual_clusters: Mapping["NeuronId", str] | None = None,
        verbose: bool = False,
    ):
        """Cluster neurons by embeddings. Use manual_clusters for pre-assigned clusters.

        Sets self._cluster_map: dict[NeuronId, str] mapping all raw neurons
        (with actual token positions) to cluster names.
        """
        last_layer = self.num_layers or 0

        if manual_clusters is not None:
            # Manual clusters: just expand the provided mapping to all raw neurons
            self._cluster_map = self._build_cluster_map_from_summed(dict(manual_clusters))
            if verbose:
                n_assigned = len(
                    {v for v in self._cluster_map.values() if v != UNCLUSTERED_CLUSTER_ID}
                    - {str(nid) for nid in self._cluster_map.keys()}
                )
                logger.info("cluster: assigned manual clusters, %d unique", n_assigned)
            if get_desc:
                self.fetch_descriptions()
            return

        # Automatic clustering: prepare data, embed, cluster
        df_indexed, _ = prepare_circuit_data(
            self.df_node[NODE_COLUMNS].copy(),
            self.df_edge[EDGE_COLUMNS].copy() if len(self.df_edge) > 0 else self.df_edge,
            sum_over_tokens=sum_over_tokens,
            verbose=verbose,
            _suppress_warning=True,
        )

        # Compute embeddings
        embeddings = compute_embeddings(df_indexed, mode=mode, do_unit_norm=do_layernorm)
        df_indexed["embedding"] = embeddings

        # Pivot embeddings by label then concatenate (or average)
        labels = list(df_indexed["label"].unique())
        df_pivot = df_indexed.pivot(
            index="input_variable", columns="label", values="embedding"
        ).reset_index()

        # Fill NaN with zero vectors
        for lbl in labels:
            sample = df_pivot[lbl].dropna().iloc[0]
            length = sample.shape[-1] if isinstance(sample, np.ndarray) else 1
            df_pivot[lbl] = df_pivot[lbl].apply(
                lambda x: np.zeros(length, dtype=np.float32) if not isinstance(x, np.ndarray) else x
            )

        if do_average_over_examples:
            df_pivot["embedding"] = df_pivot.apply(
                lambda row: np.mean([row[lbl] for lbl in labels], axis=0), axis=1
            )
        else:
            df_pivot["embedding"] = df_pivot.apply(
                lambda row: np.concatenate([row[lbl] for lbl in labels], axis=0), axis=1
            )

        # Run agglomerative clustering on non-boundary neurons
        not_boundary = df_pivot["input_variable"].apply(lambda x: x.layer not in (-1, last_layer))

        summed_cluster_map: dict[NeuronId, str] = {}

        if n_clusters == 0 or n_clusters >= not_boundary.sum() or do_one_cluster_per_neuron:
            # Trivial: every neuron gets cluster "-1"
            for _, row in df_pivot.iterrows():
                nid = row["input_variable"]
                if nid.layer in (-1, last_layer):
                    summed_cluster_map[nid] = str(nid)
                else:
                    summed_cluster_map[nid] = "-1"
        else:
            train_embeddings = df_pivot.loc[not_boundary, "embedding"].to_list()
            X = np.stack(train_embeddings)

            np.random.seed(42)
            random.seed(42)

            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=cluster_linkage,
                metric=cluster_metric,
            )
            raw_labels = clusterer.fit_predict(X)

            # Store intermediate state for downstream analysis (e.g. sweep scripts)
            self._cluster_embedding_matrix = X
            self._cluster_raw_labels = raw_labels
            self._cluster_neuron_ids = df_pivot.loc[not_boundary, "input_variable"].tolist()
            self._cluster_df_prepared = df_indexed

            idx = 0
            for _, row in df_pivot.iterrows():
                nid = row["input_variable"]
                if nid.layer in (-1, last_layer):
                    summed_cluster_map[nid] = str(nid)
                elif not_boundary[row.name]:
                    summed_cluster_map[nid] = str(raw_labels[idx])
                    idx += 1
                else:
                    summed_cluster_map[nid] = UNCLUSTERED_CLUSTER_ID

        # Expand to full cluster map
        self._cluster_map = self._build_cluster_map_from_summed(summed_cluster_map)

        if verbose:
            unique_clusters = {
                v
                for v in self._cluster_map.values()
                if v != UNCLUSTERED_CLUSTER_ID and not v.startswith("NeuronId")
            }
            logger.info("cluster: %d unique clusters assigned", len(unique_clusters))

        if get_desc:
            self.fetch_descriptions()

    def cluster_multiview(
        self,
        n_clusters: int,
        get_desc: bool = True,
        weight_by_attribution: bool = False,
        use_attributions_as_view: bool = False,
        combine: str = "mean",
        use_linear_shift: bool = False,
        unnormalized: bool = False,
        sum_over_tokens: bool = True,
        verbose: bool = False,
    ) -> None:
        """Cluster neurons using multi-view spectral clustering on per-CI similarities.

        Sets self._cluster_map: dict[NeuronId, str] mapping all raw neurons
        (with actual token positions) to cluster names.
        """
        # Prepare indexed DataFrame
        df_node_raw = self.df_node[NODE_COLUMNS].copy()
        df_prepared, _ = prepare_circuit_data(
            df_node_raw, self.df_edge.head(0), _suppress_warning=True
        )

        # Compute multi-view similarities
        mv_result = compute_per_ci_similarities(
            df_prepared,
            sum_over_tokens=sum_over_tokens,
            weight_by_attribution=weight_by_attribution,
            use_attributions_as_view=use_attributions_as_view,
            combine=combine,
            verbose=verbose,
        )
        sim_matrix = mv_result.sim_matrix
        neuron_ids = mv_result.neuron_ids

        # Filter embed/unembed layers
        max_layer = max(nid.layer for nid in neuron_ids)
        keep_mask = np.array([nid.layer not in (-1, max_layer) for nid in neuron_ids])
        sim_filtered = sim_matrix[np.ix_(keep_mask, keep_mask)]
        neuron_ids_filtered = [nid for nid, k in zip(neuron_ids, keep_mask) if k]

        if verbose:
            logger.info(
                "cluster_multiview: %d neurons, sim matrix %s, overlap zero %.1f%%",
                len(neuron_ids_filtered),
                sim_filtered.shape,
                (~mv_result.had_overlap[np.ix_(keep_mask, keep_mask)]).mean() * 100,
            )

        # Run spectral clustering
        cluster_labels = spectral_cluster(
            sim_filtered,
            n_clusters=n_clusters,
            use_linear_shift=use_linear_shift,
            unnormalized=unnormalized,
        )

        # Build summed cluster map (NeuronId with token=-1 -> "C{n}")
        summed_cluster_map: dict[NeuronId, str] = {}
        for nid, cl in zip(neuron_ids_filtered, cluster_labels):
            summed_cluster_map[nid] = f"C{cl}"

        # Expand to full cluster map
        self._cluster_map = self._build_cluster_map_from_summed(summed_cluster_map)

        # Store reverse mapping for downstream
        unique_cluster_names = sorted(set(summed_cluster_map.values()))
        self._cluster_id_to_name = {i: name for i, name in enumerate(unique_cluster_names)}

        # Store multiview results for downstream scripts (e.g. heatmap visualization)
        self._mv_sim_matrix = sim_filtered
        self._mv_neuron_ids = neuron_ids_filtered
        self._mv_cluster_labels = cluster_labels
        self._mv_overlap_counts = mv_result.overlap_counts[np.ix_(keep_mask, keep_mask)]
        self._mv_had_overlap = mv_result.had_overlap[np.ix_(keep_mask, keep_mask)]
        self._mv_df_prepared = df_prepared

        if verbose:
            n_assigned = len(unique_cluster_names)
            logger.info("cluster_multiview: assigned %d clusters", n_assigned)

        if get_desc:
            self.fetch_descriptions()

    def fetch_descriptions(self) -> None:
        """Fetch per-neuron descriptions from the description database."""
        if self.tokenizer is None:
            return

        # Build a minimal DataFrame with input_variable column for get_descriptions
        nids = list(self._cluster_map.keys())
        if not nids:
            return

        df = pd.DataFrame({"input_variable": nids})
        df, self.neuron_label_cache = get_descriptions(
            df,
            self.tokenizer,
            self.num_layers,
            get_desc=True,
            verbose=False,
            neuron_label_cache=self.neuron_label_cache,
        )

    @property
    def df_node_embedded(self) -> pd.DataFrame:
        """Build a minimal DataFrame with input_variable and cluster columns from _cluster_map.

        Provided for backward compatibility with code that reads df_node_embedded.
        """
        if not self._cluster_map:
            raise ValueError("Call cluster() first to generate cluster assignments")

        records = []
        for nid, cluster in self._cluster_map.items():
            record: dict[str, Any] = {
                "input_variable": nid,
                "cluster": cluster,
            }
            # Add description from cache if available
            desc_key = (nid.layer, nid.neuron)
            if desc_key in self.neuron_label_cache:
                record["description"] = self.neuron_label_cache[desc_key]
            else:
                record["description"] = ""
            records.append(record)

        return pd.DataFrame(records)

    #########################################################
    # STEERING
    #########################################################

    def steer(
        self,
        model: Any,
        multiplier: float = 0.0,
        verbose: bool = False,
        record: bool = False,
        store_results: bool = False,
        complement: bool = False,
        custom_neuron_ids: list[tuple[int, int, int]] | None = None,
    ):
        """Steer the circuit by computing steering effects for each cluster."""
        diversity_stats, cluster_to_cluster, cluster_to_output = get_cluster_steering_effects(
            model=model,
            tokenizer=self.tokenizer,
            df_node=self.df_node,
            df_edge=self.df_edge,
            cluster_map=self._cluster_map,
            cis=self.cis,
            attention_masks=self.attention_masks,
            labels=self.labels,
            multiplier=multiplier,
            verbose=verbose,
            record=record,
            complement=complement,
            custom_neuron_ids=custom_neuron_ids,
        )
        if store_results:
            self.diversity_stats = pd.concat(
                [self.diversity_stats, diversity_stats.assign(multiplier=multiplier)]
            )
            self.cluster_to_cluster = pd.concat(
                [
                    self.cluster_to_cluster,
                    cluster_to_cluster.assign(multiplier=multiplier),
                ]
            )
            self.cluster_to_output = pd.concat(
                [
                    self.cluster_to_output,
                    cluster_to_output.assign(multiplier=multiplier),
                ]
            )
        return diversity_stats, cluster_to_cluster, cluster_to_output

    def clear_steering_results(self):
        """Clear the steering results."""
        self.diversity_stats = pd.DataFrame()
        self.cluster_to_cluster = pd.DataFrame()
        self.cluster_to_output = pd.DataFrame()
        self._df_node_summed_cache = None

    #########################################################
    # SCORING
    #########################################################

    def _get_df_node_summed(self, verbose: bool = False) -> pd.DataFrame:
        """Aggregate node-level attributions by input variable and label (cached)."""
        from circuits.analysis import score_features

        return score_features.get_df_node_summed(self, verbose=verbose)

    def score_features(
        self,
        example_labels: Sequence[bool],
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Compute ROC-AUC scores for aggregated neuron features."""
        from circuits.analysis import score_features

        return score_features.score_features(self, example_labels, verbose=verbose)

    def score_features_multiclass(
        self,
        example_labels: Sequence[Any],
        hypothesis_name: str | None = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Per-class ROC-AUC scores for multiclass labels."""
        if not hasattr(self, "hypothesis_to_scores"):
            self.hypothesis_to_scores: dict[str, pd.DataFrame] = {}
        if hypothesis_name is not None and hypothesis_name in self.hypothesis_to_scores:
            return self.hypothesis_to_scores[hypothesis_name]
        from circuits.analysis import score_features

        scores = score_features.score_features_multiclass(self, example_labels, verbose=verbose)
        if hypothesis_name is not None:
            self.hypothesis_to_scores[hypothesis_name] = scores
        return scores

    def cluster_with_hypotheses(
        self,
        hypotheses: Mapping[str, Sequence[Any]],
        above_threshold: float = 0.8,
        below_threshold: float = 0.2,
        in_class_attribution_threshold: float | None = None,
        unique_only: bool = False,
        subset_labels: Sequence[str] | None = None,
        cluster_kwargs: Mapping[str, Any] | None = None,
        verbose: bool = False,
    ) -> dict[str, int]:
        """Assign neurons to hypothesis-defined clusters."""
        from circuits.analysis import score_features

        return asyncio.run(
            score_features.cluster_with_hypotheses(
                self,
                hypotheses=hypotheses,
                above_threshold=above_threshold,
                below_threshold=below_threshold,
                in_class_attribution_threshold=in_class_attribution_threshold,
                unique_only=unique_only,
                subset_labels=subset_labels,
                cluster_kwargs=cluster_kwargs,
                max_layer=self.num_layers,
                verbose=verbose,
            )
        )

    def export_hypothesis_score_jsons(
        self,
        hypotheses: Mapping[str, Sequence[Any]],
        output_dir: Path,
        auc_threshold_high: float = 0.8,
        auc_threshold_low: float = 0.2,
        in_class_attribution_threshold: float | None = None,
    ) -> None:
        from circuits.analysis import score_features

        score_features.export_hypothesis_score_jsons(
            self,
            hypotheses,
            output_dir,
            auc_threshold_high,
            auc_threshold_low,
            in_class_attribution_threshold,
        )

    #########################################################
    # LABELLING
    #########################################################

    def set_up_train_test_split(
        self,
        test_ratio: float = 0.2,
        seed: int = 42,
    ):
        """Set up the train-test split for example labels."""
        random.seed(seed)
        unique_labels = list(self.df_node.label.unique())
        random.shuffle(unique_labels)
        num_test = int(len(unique_labels) * test_ratio)
        self.test_labels = unique_labels[:num_test]
        self.train_labels = unique_labels[num_test:]

    def _prepare_clustered_df_for_labelling(
        self,
        weight_attr_by_attribution: bool = False,
    ) -> tuple[pd.DataFrame, dict[int, list[tuple[int, int]]]]:
        """Aggregate neurons by cluster for labelling. Returns (df, cluster_to_neurons)."""
        if not self._cluster_map:
            raise ValueError("Must call cluster() before labelling clusters")

        # Add polarity to df_node if not present
        df = self.df_node.copy()
        if "polarity" not in df.columns:
            df["polarity"] = df["activation"].apply(lambda x: "+" if x >= 0 else "-")

        # Create NeuronId for each row and map to cluster using _cluster_map
        def get_cluster(row: pd.Series) -> str:
            neuron_id = NeuronId(
                layer=row["layer"],
                token=row["token"],
                neuron=row["neuron"],
                polarity=row["polarity"],
            )
            return self._cluster_map.get(neuron_id, "-1")

        df["cluster"] = df.apply(get_cluster, axis=1)

        # Filter out unclustered neurons (cluster == "-1") and boundary layers
        last_layer = self.num_layers if self.num_layers else df["layer"].max()
        df = df[(df["cluster"] != "-1") & (df["layer"] != -1) & (df["layer"] != last_layer)]

        if len(df) == 0:
            raise ValueError("No clustered neurons found. Make sure to call cluster() first.")

        # Optionally weight attr_map by attribution so that neurons with higher
        # attribution weight more heavily when summing across tokens and clusters.
        if weight_attr_by_attribution:
            df["attr_map"] = df.apply(
                lambda row: np.asarray(row["attr_map"]) * row["attribution"], axis=1
            )

        # Step 1: Sum over tokens per neuron (within each label)
        # First build a mapping from (layer, neuron) -> cluster
        neuron_to_cluster: dict[tuple[int, int], str] = {}
        for _, row in df.iterrows():
            key = (int(row["layer"]), int(row["neuron"]))
            if key not in neuron_to_cluster:
                neuron_to_cluster[key] = row["cluster"]

        # Use df_sum_over_tokens to sum over tokens (merging polarities)
        df_summed, _ = df_sum_over_tokens(df, pd.DataFrame(), include_polarity=True)

        # Add cluster column back
        df_summed["cluster"] = df_summed.apply(
            lambda row: neuron_to_cluster.get((int(row["layer"]), int(row["neuron"])), "-1"),
            axis=1,
        )

        # Step 2: Group by cluster and label, summing attr_map/contrib_map across neurons
        # Since attr_map and contrib_map are normalized (relative contributions), summing
        # gives the total attribution from all neurons in the cluster
        def sum_arrays(series: pd.Series) -> np.ndarray:
            arrays = [np.asarray(x) for x in series if x is not None]
            if not arrays:
                return np.array([0.0])
            # Pad arrays to same length if needed
            max_len = max(len(a) for a in arrays)
            padded = [np.pad(a, (0, max_len - len(a))) for a in arrays]
            return np.sum(padded, axis=0)

        df_clustered = (
            df_summed.groupby(["cluster", "label"])
            .agg(
                {
                    "attr_map": sum_arrays,
                    "contrib_map": sum_arrays,
                    "activation": "sum",  # Sum activations too for total cluster contribution
                    "attribution": "sum",
                }
            )
            .reset_index()
        )

        # Convert to format expected by label_clusters:
        # Use cluster ID as "layer", 0 as token and neuron
        # For string cluster names, assign stable sequential IDs and store the mapping
        unique_clusters = sorted(df_clustered["cluster"].unique())
        cluster_name_to_id: dict[str, int] = {}
        for cluster_str in unique_clusters:
            if cluster_str.lstrip("-").isdigit():
                cluster_name_to_id[cluster_str] = int(cluster_str)
            else:
                # Assign sequential IDs starting from 0 for named clusters
                cluster_name_to_id[cluster_str] = len(cluster_name_to_id)

        # Store reverse mapping on self for downstream use
        self._cluster_id_to_name = {v: k for k, v in cluster_name_to_id.items()}

        df_clustered["layer"] = df_clustered["cluster"].apply(lambda x: cluster_name_to_id[x])
        df_clustered["token"] = 0
        df_clustered["neuron"] = 0

        # Build cluster_to_neurons mapping: cluster_id (int) -> list of NeuronId
        # Use df_summed which has unique (layer, neuron, polarity, cluster) combinations
        cluster_to_neurons: dict[int, list[NeuronId]] = {}
        for _, row in df_summed.iterrows():
            cluster_str = row["cluster"]
            cluster_int = cluster_name_to_id[cluster_str]
            neuron_id = NeuronId(
                layer=int(row["layer"]),
                token=0,
                neuron=int(row["neuron"]),
                polarity=row.get("polarity", "+"),
            )
            if cluster_int not in cluster_to_neurons:
                cluster_to_neurons[cluster_int] = []
            if neuron_id not in cluster_to_neurons[cluster_int]:
                cluster_to_neurons[cluster_int].append(neuron_id)

        # Create input_variable (NeuronId) column for compatibility with label_clusters
        # Use constant "+" polarity since we've merged +/- in aggregation
        df_clustered["input_variable"] = df_clustered.apply(
            lambda row: NeuronId(row["layer"], row["token"], row["neuron"], "+"),
            axis=1,
        )

        # Keep only required columns
        result = df_clustered[
            [
                "layer",
                "token",
                "neuron",
                "label",
                "attr_map",
                "contrib_map",
                "activation",
                "attribution",
                "input_variable",
            ]
        ].copy()

        return result, cluster_to_neurons

    def label_clusters_simulator_v2(
        self,
        score_explanations: bool = True,
        max_clusters: int | None = None,
        num_expl_samples: int = 5,
        use_clustered: bool = True,
        min_highlights: int = 1,
        threshold_mode: str = "quantile",
        enforce_top_exemplars: int = 0,
        attr_explainer_name: str = "Transluce/llama_8b_explainer",
        attr_simulator_name: str = "Transluce/llama_8b_simulator",
        contrib_model_name: str = "claude-haiku-4-5-20251001",
        verbose: bool = False,
        weight_attr_by_attribution: bool = False,
        return_neuron_data: bool = False,
        target_cluster_ids: list[int] | None = None,
        max_exemplars_per_batch: int = 25,
        skip_attr: bool = False,
        skip_contrib: bool = False,
        attr_backend: str = "vllm",
        attr_api_model_name: str = "claude-haiku-4-5-20251001",
    ):
        """Generate explanations using descriptions (no observatory dependency).

        Uses VLLM for attr explanation generation, Anthropic API for contrib (minibatch),
        finetuned simulator for attr scoring, and API for contrib scoring.

        Args:
            threshold_mode: "quantile" (walk quantiles from extreme) or "topk"
                (walk quantiles from extreme). Default "topk" for cleaner highlights.
            enforce_top_exemplars: If > 0, always include this many top records (by max
                activation) in every exemplar subset, filling remaining slots randomly.
            skip_attr: Skip attr explanation generation and scoring (phases 1 & 3).
            skip_contrib: Skip contrib explanation generation and scoring (phases 2 & 4).
            attr_backend: "vllm" (default, GPU models) or "api" (Anthropic API).
            attr_api_model_name: Anthropic model name for API attr backend.

        Returns same structure as label_clusters_simulator:
            (attr_results, contrib_results, attr_exemplars, contrib_exemplars, cluster_to_neurons)
            or 6-tuple with neuron_data if return_neuron_data=True.
        """
        from circuits.descriptions import label as label_v2

        cluster_to_neurons: dict[int, list[NeuronId]] = {}
        if use_clustered:
            df_for_labelling, cluster_to_neurons = self._prepare_clustered_df_for_labelling(
                weight_attr_by_attribution=weight_attr_by_attribution,
            )
            num_layers = None
        else:
            df_for_labelling = self.df_node
            num_layers = self.num_layers

        result = label_v2.label_clusters(
            df_node=df_for_labelling,
            cis=self.cis,
            tokenizer=self.tokenizer,
            target_logits=self.target_logits,
            attr_explainer_name=attr_explainer_name,
            attr_simulator_name=attr_simulator_name,
            contrib_model_name=contrib_model_name,
            score_explanations=score_explanations,
            num_layers=num_layers,
            max_neurons=max_clusters,
            num_expl_samples=num_expl_samples,
            min_highlights=min_highlights,
            threshold_mode=threshold_mode,
            enforce_top_exemplars=enforce_top_exemplars,
            verbose=verbose,
            return_neuron_data=return_neuron_data,
            target_neuron_ids=(
                [NeuronId(layer=cid, token=0, neuron=0, polarity="+") for cid in target_cluster_ids]
                if target_cluster_ids is not None
                else None
            ),
            max_exemplars_per_batch=max_exemplars_per_batch,
            skip_attr=skip_attr,
            skip_contrib=skip_contrib,
            attr_backend=attr_backend,
            attr_api_model_name=attr_api_model_name,
        )
        if return_neuron_data:
            attr_results, contrib_results, attr_exemplars, contrib_exemplars, neuron_data = result
        else:
            attr_results, contrib_results, attr_exemplars, contrib_exemplars = result
            neuron_data = None

        # Store best explanation per cluster for frontend use
        self._store_cluster_descriptions(attr_results, contrib_results)

        if return_neuron_data:
            return (
                attr_results,
                contrib_results,
                attr_exemplars,
                contrib_exemplars,
                cluster_to_neurons,
                neuron_data,
            )
        else:
            return (
                attr_results,
                contrib_results,
                attr_exemplars,
                contrib_exemplars,
                cluster_to_neurons,
            )

    def _store_cluster_descriptions(
        self,
        attr_results: dict,
        contrib_results: dict,
    ) -> None:
        """Extract best attr and contrib explanation per cluster and store for frontend."""
        self._cluster_descriptions = {}
        self._cluster_attr_descriptions: dict[str, str] = {}
        self._cluster_contrib_descriptions: dict[str, str] = {}

        def _best_explanation(signed: dict) -> str | None:
            best: str | None = None
            best_score: float = -1.0
            for sign in ("pos", "neg", "combined"):
                for expl in signed.get(sign, []):
                    # Handle both ScoredExplanation objects and raw strings
                    if isinstance(expl, str):
                        if expl and best is None:
                            best = expl
                        continue
                    score = expl.score if expl.score is not None else 0.0
                    if score > best_score:
                        best_score = score
                        best = expl.explanation
            return best

        for nid, signed in attr_results.items():
            cluster_int = nid.layer
            cluster_name = self._cluster_id_to_name.get(cluster_int, str(cluster_int))
            best = _best_explanation(signed)
            if best:
                self._cluster_attr_descriptions[cluster_name] = best

        for nid, signed in contrib_results.items():
            cluster_int = nid.layer
            cluster_name = self._cluster_id_to_name.get(cluster_int, str(cluster_int))
            best = _best_explanation(signed)
            if best:
                self._cluster_contrib_descriptions[cluster_name] = best

        # Combined label for frontend: "attr | contrib"
        all_clusters = set(self._cluster_attr_descriptions) | set(
            self._cluster_contrib_descriptions
        )
        for cl in all_clusters:
            attr = self._cluster_attr_descriptions.get(cl)
            contrib = self._cluster_contrib_descriptions.get(cl)
            if attr and contrib:
                self._cluster_descriptions[cl] = f"{attr} | {contrib}"
            elif attr:
                self._cluster_descriptions[cl] = attr
            elif contrib:
                self._cluster_descriptions[cl] = contrib

    def _get_cluster_top_examples(self, top_k: int = 10) -> dict[str, list[str]]:
        """Get top-k dataset examples (by summed attribution) for each cluster.

        Returns decoded prompt text for each example, sorted by total attribution descending.

        Returns:
            {cluster_name: ["prompt_text (→ label)", ...]}
        """
        from collections import defaultdict

        if not self._cluster_map:
            return {}

        neuron_to_cluster: dict[tuple[int, int], str] = {}
        for nid, cl in self._cluster_map.items():
            neuron_to_cluster[(int(nid.layer), int(nid.neuron))] = cl

        num_layers = self.num_layers or int(self.df_node["layer"].max())

        # Build label -> ci_idx mapping
        label_to_idx: dict[str, int] = {}
        for i, lbl in enumerate(self.labels):
            label_to_idx[lbl] = i

        # Accumulate attribution per (cluster, ci_idx)
        cluster_ci_attr: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        for _, row in self.df_node.iterrows():
            layer, neuron = int(row["layer"]), int(row["neuron"])
            if layer < 0 or layer >= num_layers:
                continue
            cl = neuron_to_cluster.get((layer, neuron))
            if cl is None:
                continue
            label = str(row["label"])
            if "___" in label:
                ci_idx = int(label.rsplit("___", 1)[1])
            elif label in label_to_idx:
                ci_idx = label_to_idx[label]
            else:
                continue
            cluster_ci_attr[cl][ci_idx] += abs(float(row["attribution"]))

        # Sort and take top-k per cluster, decode prompts
        result: dict[str, list[str]] = {}
        for cl, ci_attrs in cluster_ci_attr.items():
            sorted_cis = sorted(ci_attrs.items(), key=lambda x: x[1], reverse=True)[:top_k]
            examples = []
            for ci_idx, _ in sorted_cis:
                if ci_idx < len(self.cis):
                    ci = self.cis[ci_idx]
                    # Strip padding
                    if self.attention_masks is not None and ci_idx < len(self.attention_masks):
                        mask = self.attention_masks[ci_idx]
                        pad_start = next(
                            (
                                j
                                for j, m in enumerate(mask)
                                if (m.item() if hasattr(m, "item") else m) == 1
                            ),
                            0,
                        )
                        ci = ci[pad_start:]
                    prompt_text = self.tokenizer.decode(ci) if self.tokenizer else str(ci)
                    label = self.labels[ci_idx].strip() if ci_idx < len(self.labels) else "?"
                    examples.append(f"{prompt_text} → {label}")
            result[cl] = examples
        return result

    def summarize_clusters(
        self,
        mode: str = "rich",
        summary_model: str = "claude-opus-4-6",
        max_neurons_per_cluster: int = 20,
        neurons_only: bool = False,
        attr_only: bool = False,
        top_k_examples: int = 10,
    ) -> dict[str, str]:
        """Generate summary labels for clusters.

        Args:
            mode: "rich" (attr + contrib + individual neuron descriptions per cluster,
                one API call per cluster) or "batch" (original batch approach, attr + contrib
                only, all clusters in one call).
            summary_model: Anthropic model for summarization.
            max_neurons_per_cluster: Max neuron descriptions per cluster (rich mode only).
            neurons_only: If True, only use individual neuron descriptions (skip attr/contrib).
                Only applies to rich mode.
            attr_only: If True, use only attr + contrib descriptions (skip neuron descriptions).
                Only applies to rich mode.
            top_k_examples: Number of top dataset examples to include per cluster (0 to skip).

        Returns:
            {cluster_name: short_label}. Also stored as self._cluster_summary_labels.
        """
        import asyncio

        attr_descs = getattr(self, "_cluster_attr_descriptions", {})
        contrib_descs = getattr(self, "_cluster_contrib_descriptions", {})

        if mode == "rich":
            from circuits.descriptions.label import summarize_clusters_rich

            # Build per-cluster neuron descriptions sorted by avg attribution
            cluster_neuron_descs = self._get_cluster_neuron_descriptions()

            # Build per-cluster top examples
            cluster_exemplars: dict[str, list[str]] = {}
            if top_k_examples > 0:
                cluster_exemplars = self._get_cluster_top_examples(top_k=top_k_examples)

            labels, thinking_traces = asyncio.run(
                summarize_clusters_rich(
                    cluster_attr_descs=attr_descs,
                    cluster_contrib_descs=contrib_descs,
                    cluster_neuron_descs=cluster_neuron_descs,
                    cluster_exemplars=cluster_exemplars,
                    model_id=self.model_id,
                    num_layers=self.num_layers or 32,
                    summary_model=summary_model,
                    max_neurons_per_cluster=max_neurons_per_cluster,
                    neurons_only=neurons_only,
                    attr_only=attr_only,
                )
            )
            self._cluster_summary_thinking = thinking_traces
        elif mode == "batch":
            from circuits.analysis.label import summarize_attr_contrib_descriptions

            descs_input = {
                n: {"attr": attr_descs.get(n, ""), "contrib": contrib_descs.get(n, "")}
                for n in set(attr_descs) | set(contrib_descs)
            }
            batch_exemplars: dict[str, list[str]] = {}
            if top_k_examples > 0:
                batch_exemplars = self._get_cluster_top_examples(top_k=top_k_examples)
            labels = asyncio.run(
                summarize_attr_contrib_descriptions(
                    descs_input, model=summary_model, cluster_exemplars=batch_exemplars
                )
            )
        else:
            raise ValueError(f"Unknown summarization mode: {mode!r}. Use 'rich' or 'batch'.")

        self._cluster_summary_labels = labels
        return labels

    def _get_cluster_neuron_descriptions(self) -> dict[str, list[tuple[str, float]]]:
        """Build per-cluster neuron descriptions sorted by avg attribution.

        Returns:
            {cluster_name: [(description, avg_attribution), ...]} sorted descending by |attribution|.
        """
        from collections import defaultdict

        import numpy as np

        # Compute avg |attribution| per (layer, neuron) across all labels
        num_layers = self.num_layers or int(self.df_node["layer"].max())
        mlp_nodes = self.df_node[
            (self.df_node["layer"] >= 0) & (self.df_node["layer"] < num_layers)
        ]
        avg_attr = (
            mlp_nodes.groupby(["layer", "neuron"])["attribution"]
            .agg(["mean", lambda x: np.mean(np.abs(x))])
            .reset_index()
        )
        avg_attr.columns = ["layer", "neuron", "mean_attr", "mean_abs_attr"]

        # Build description lookup from neuron_label_cache
        neuron_desc: dict[tuple[int, int], str] = {}
        for key, desc in self.neuron_label_cache.items():
            if not isinstance(key, tuple) or not desc or desc == "N.A.":
                continue
            if len(key) == 2:
                neuron_desc.setdefault((int(key[0]), int(key[1])), desc)
            elif len(key) == 3:
                k2 = (int(key[0]), int(key[1]))
                if k2 not in neuron_desc:
                    neuron_desc[k2] = desc

        # Group by cluster, deduplicating by (layer, neuron)
        cluster_neurons: dict[str, dict[tuple[int, int], float]] = defaultdict(dict)
        for nid, cl in self._cluster_map.items():
            layer, neuron = int(nid.layer), int(nid.neuron)
            if layer < 0 or layer >= num_layers:
                continue
            # Sum attribution across token positions for same (layer, neuron)
            match = avg_attr[(avg_attr["layer"] == layer) & (avg_attr["neuron"] == neuron)]
            score = float(match["mean_attr"].iloc[0]) if len(match) > 0 else 0.0
            if (layer, neuron) in cluster_neurons[cl]:
                cluster_neurons[cl][(layer, neuron)] += score
            else:
                cluster_neurons[cl][(layer, neuron)] = score

        result: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for cl, neurons in cluster_neurons.items():
            for (layer, neuron), score in neurons.items():
                desc = neuron_desc.get((layer, neuron), "")
                if not desc:
                    desc = f"L{layer}N{neuron} (no description)"
                result[cl].append((desc, score))

        # Sort each cluster by |attribution| descending
        for cl in result:
            result[cl].sort(key=lambda x: abs(x[1]), reverse=True)

        return dict(result)

    def diagnose_highlight_thresholds(
        self,
        max_clusters: int | None = None,
        min_highlights_values: list[int] | None = None,
        use_clustered: bool = True,
    ) -> dict:
        """Diagnose highlighting thresholds without generating explanations."""
        if use_clustered:
            df_for_labelling, _ = self._prepare_clustered_df_for_labelling()
            num_layers = None
        else:
            df_for_labelling = self.df_node
            num_layers = self.num_layers

        from circuits.analysis import label_simulator

        return label_simulator.diagnose_highlight_thresholds(
            df_node=df_for_labelling,
            cis=self.cis,
            tokenizer=self.tokenizer,
            target_logits=self.target_logits,
            num_layers=num_layers,
            max_neurons=max_clusters,
            min_highlights_values=min_highlights_values,
        )

    def eval_labels(
        self,
        labels: list[str],
        model: str = "gpt-4o",
        polarity: Literal["positive", "negative"] = "positive",
        use_test_only: bool = False,
    ):
        if use_test_only:
            assert self.test_labels is not None, "Test labels not set up"

        cluster_to_output = self.cluster_to_output
        if use_test_only:
            cluster_to_output = cluster_to_output[cluster_to_output.label.isin(self.test_labels)]

        return asyncio.run(
            label.eval_labels_on_steering(
                labels, self.cluster_to_output, model=model, polarity=polarity
            )
        )

    #########################################################
    # VISUALIZATION
    #########################################################

    def clear_label_cache(self):
        """Clear the label cache."""
        self.neuron_label_cache = {}

    def save_to_pickle(self, out_pickle: str | Path):
        """Save the full Circuit object (preserves cluster assignments etc.)."""
        import pickle

        # Clear tokenizer before pickling (not serializable / large)
        tokenizer = self.tokenizer
        self.tokenizer = None
        with open(out_pickle, "wb") as f:
            pickle.dump(self, f)
        self.tokenizer = tokenizer
        logger.info(f"Saved circuit to {out_pickle}")

    def save_circuit_data_to_pickle(self, out_pickle: str | Path):
        """Save only the CircuitData (no cluster assignments)."""
        self.circuit_data.save_to_pickle(str(out_pickle))
        logger.info(f"Saved circuit data to {out_pickle}")

    @classmethod
    def load_from_pickle(cls, in_pickle: str | Path) -> "Circuit":
        """Load circuit from pickle (supports both Circuit and legacy CircuitData pickles)."""
        import pickle

        with open(in_pickle, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, cls):
            logger.info(f"Loaded circuit from {in_pickle}")
            return obj
        # Legacy: CircuitData pickle
        if isinstance(obj, CircuitData):
            circuit = cls(obj)
            logger.info(f"Loaded circuit (from CircuitData) from {in_pickle}")
            return circuit
        raise TypeError(f"Expected Circuit or CircuitData, got {type(obj)}")

    def set_tokenizer(self, tokenizer: Any, num_layers: int | None = None) -> None:
        """Set the tokenizer and num_layers after loading from pickle."""
        self.tokenizer = tokenizer
        if num_layers is not None:
            self.num_layers = num_layers

    def save_cluster_state(self, path: str | Path) -> None:
        """Save cluster assignments and descriptions as lightweight JSON.

        Much smaller than pickling the full Circuit (~KB vs ~GB).
        """
        import json

        state: dict[str, Any] = {
            "cluster_map": {
                f"{nid.layer},{nid.token},{nid.neuron},{nid.polarity}": cl
                for nid, cl in self._cluster_map.items()
            },
            "cluster_id_to_name": {str(k): v for k, v in self._cluster_id_to_name.items()},
            "cluster_descriptions": getattr(self, "_cluster_descriptions", {}),
            "cluster_attr_descriptions": getattr(self, "_cluster_attr_descriptions", {}),
            "cluster_contrib_descriptions": getattr(self, "_cluster_contrib_descriptions", {}),
            "cluster_summary_labels": getattr(self, "_cluster_summary_labels", {}),
            "cluster_summary_thinking": getattr(self, "_cluster_summary_thinking", {}),
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        logger.info(f"Saved cluster state ({len(self._cluster_map)} neurons) to {path}")

    def load_cluster_state(self, path: str | Path) -> None:
        """Load cluster assignments and descriptions from JSON."""
        import json

        with open(path) as f:
            state = json.load(f)

        def _parse_nid(s: str) -> NeuronId:
            parts = s.split(",")
            return NeuronId(int(parts[0]), int(parts[1]), int(parts[2]), parts[3])

        self._cluster_map = {_parse_nid(k): v for k, v in state["cluster_map"].items()}
        self._cluster_id_to_name = {int(k): v for k, v in state["cluster_id_to_name"].items()}
        self._cluster_descriptions = state.get("cluster_descriptions", {})
        self._cluster_attr_descriptions = state.get("cluster_attr_descriptions", {})
        self._cluster_contrib_descriptions = state.get("cluster_contrib_descriptions", {})
        self._cluster_summary_labels = state.get("cluster_summary_labels", {})
        self._cluster_summary_thinking = state.get("cluster_summary_thinking", {})
        logger.info(f"Loaded cluster state ({len(self._cluster_map)} neurons) from {path}")

    def save_cluster_to_output_to_json(self, out_json: str | Path):
        """Save the cluster data to a JSON file."""
        export_cluster_data_to_json(self.cluster_to_output, out_json)

    #########################################################
    # CIRCUIT-TRACER EXPORT
    #########################################################

    def export_to_circuit_tracer(
        self,
        output_dir: str | Path,
        slug: str = "circuit",
    ) -> list[Path]:
        """Export circuit data to circuit-tracer JSON format.

        Args:
            output_dir: Directory to write JSON files to.
            slug: Base slug for filenames.

        Returns:
            List of paths to written JSON files.
        """
        from circuits.frontend.graph_models import Link, Metadata, Model, Node, QParams

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        unique_labels = sorted(self.df_node["label"].unique())
        written_files: list[Path] = []
        graph_metadata_entries: list[dict] = []
        num_layers = self.num_layers or int(self.df_node["layer"].max())

        # Build (layer, neuron) -> cluster lookup from _cluster_map
        neuron_to_cluster: dict[tuple[int, int], str] = {}
        for nid, cluster in self._cluster_map.items():
            key = (int(nid.layer), int(nid.neuron))
            if key not in neuron_to_cluster:
                neuron_to_cluster[key] = cluster

        # Build (layer, neuron) -> description lookup from neuron_label_cache
        # Cache keys are (layer, neuron, polarity) triples; collapse to (layer, neuron)
        # preferring positive polarity descriptions
        cluster_desc: dict[tuple[int, int], str] = {}
        for key, desc in self.neuron_label_cache.items():
            if not isinstance(key, tuple) or not desc or desc == "N.A.":
                continue
            if len(key) == 2:
                cluster_desc.setdefault(key, desc)
            elif len(key) == 3:
                layer_n, neuron_n = key[0], key[1]
                k2 = (layer_n, neuron_n)
                # Prefer positive polarity; overwrite only if we don't have one yet
                if k2 not in cluster_desc:
                    cluster_desc[k2] = desc

        for i, lbl in enumerate(unique_labels):
            df_n = self.df_node[self.df_node["label"] == lbl]
            df_e = (
                self.df_edge[self.df_edge["label"] == lbl]
                if len(self.df_edge) > 0
                else self.df_edge
            )

            # Decode prompt tokens — labels in df have "___N" suffix encoding the ci index
            if "___" in lbl:
                ci_idx = int(lbl.rsplit("___", 1)[1])
            else:
                ci_idx = list(self.labels).index(lbl) if lbl in list(self.labels) else 0
            ci = self.cis[ci_idx]

            # Build logit token -> prob mapping
            logit_probs: dict[int, float] = {}
            if (
                hasattr(self.circuit_data, "target_logit_probs")
                and self.circuit_data.target_logit_probs is not None
                and ci_idx < len(self.circuit_data.target_logit_probs)
            ):
                for tok_id, prob in zip(
                    self.target_logits[ci_idx], self.circuit_data.target_logit_probs[ci_idx]
                ):
                    logit_probs[tok_id] = prob
            # Strip left-padding using attention mask so prompt_tokens aligns with attr_map
            pad_offset = 0
            if self.attention_masks is not None and ci_idx < len(self.attention_masks):
                mask = self.attention_masks[ci_idx]
                # Find first non-pad position
                for j, m in enumerate(mask):
                    val = m.item() if hasattr(m, "item") else m
                    if val == 1:
                        pad_offset = j
                        break
                ci = ci[pad_offset:]

            if self.tokenizer is not None:
                prompt_tokens = [self.tokenizer.decode([t]) for t in ci]
            else:
                prompt_tokens = [str(t.item() if hasattr(t, "item") else t) for t in ci]

            # Trim prompt_tokens to match attr_map length (attr_map may have BOS stripped
            # during tracing). Use the min token position in df_node to detect the offset.
            min_token_pos = int(df_n["token"].min()) - pad_offset if len(df_n) > 0 else 0
            if min_token_pos > 0:
                prompt_tokens = prompt_tokens[min_token_pos:]

            prompt_str = "".join(prompt_tokens)

            # Build nodes
            nodes: list[Node] = []
            node_id_set: set[str] = set()
            # Track (layer, neuron) -> node_ids for cluster grouping
            neuron_key_to_node_ids: dict[tuple[int, int], list[str]] = {}

            max_influence = 0.0
            for _, row in df_n.iterrows():
                layer = int(row["layer"])
                token = int(row["token"]) - pad_offset - min_token_pos
                neuron = int(row["neuron"])
                attr = float(row["attribution"])
                act = float(row["activation"]) if pd.notna(row.get("activation")) else 0.0
                influence = abs(attr)
                max_influence = max(max_influence, influence)

                if layer == -1:
                    # Embedding node
                    if token >= len(ci):
                        continue  # skip if token position is out of range (e.g. BOS stripped)
                    vocab_id = int(ci[token].item() if hasattr(ci[token], "item") else ci[token])
                    node = Node.token_node(pos=token, vocab_idx=vocab_id, influence=influence)
                    if self.tokenizer is not None:
                        node.clerp = self.tokenizer.decode([vocab_id])
                    else:
                        node.clerp = str(vocab_id)
                elif layer == num_layers:
                    # Logit node
                    token_prob = logit_probs.get(neuron, 0.0)
                    node = Node.logit_node(
                        pos=token,
                        vocab_idx=neuron,
                        token=(
                            self.tokenizer.decode([neuron])
                            if self.tokenizer is not None
                            else str(neuron)
                        ),
                        num_layers=num_layers,
                        target_logit=True,
                        token_prob=token_prob,
                    )
                    node.influence = influence
                    node.activation = act
                else:
                    # Hidden neuron
                    node = Node.feature_node(
                        layer=layer,
                        pos=token,
                        feat_idx=neuron,
                        influence=influence,
                        activation=act,
                    )
                    desc = cluster_desc.get((layer, neuron))
                    node.clerp = desc if desc else f"L{layer} N{neuron}"
                    neuron_key_to_node_ids.setdefault((layer, neuron), []).append(node.node_id)

                # Attach attr_map and contrib_map for all node types
                # attr_map is already stripped of padding+BOS during tracing (trace.py extract_map)
                if "attr_map" in row and row["attr_map"] is not None:
                    node.attr_map = np.asarray(row["attr_map"]).tolist()
                if "contrib_map" in row and row["contrib_map"] is not None:
                    node.contrib_map = np.asarray(row["contrib_map"]).tolist()

                if node.node_id not in node_id_set:
                    node_id_set.add(node.node_id)
                    nodes.append(node)

            # Build links
            links: list[dict] = []
            for _, row in df_e.iterrows():
                src_layer_s, tgt_layer_s = str(row["layer"]).split("->")
                src_token_s, tgt_token_s = str(row["token"]).split("->")
                src_neuron_s, tgt_neuron_s = str(row["neuron"]).split("->")
                src_layer = int(src_layer_s)
                tgt_layer = int(tgt_layer_s)
                src_token = int(src_token_s) - pad_offset - min_token_pos
                tgt_token = int(tgt_token_s) - pad_offset - min_token_pos
                src_neuron = int(src_neuron_s)
                tgt_neuron = int(tgt_neuron_s)
                weight = float(row["weight"])

                # Build source node_id
                if src_layer == -1:
                    if src_token >= len(ci):
                        continue
                    vocab_id = int(
                        ci[src_token].item() if hasattr(ci[src_token], "item") else ci[src_token]
                    )
                    src_id = f"E_{vocab_id}_{src_token}"
                elif src_layer == num_layers:
                    src_id = f"{num_layers + 1}_{src_neuron}_{src_token}"
                else:
                    src_id = f"{src_layer}_{src_neuron}_{src_token}"

                # Build target node_id
                if tgt_layer == -1:
                    if tgt_token >= len(ci):
                        continue
                    vocab_id = int(
                        ci[tgt_token].item() if hasattr(ci[tgt_token], "item") else ci[tgt_token]
                    )
                    tgt_id = f"E_{vocab_id}_{tgt_token}"
                elif tgt_layer == num_layers:
                    tgt_id = f"{num_layers + 1}_{tgt_neuron}_{tgt_token}"
                else:
                    tgt_id = f"{tgt_layer}_{tgt_neuron}_{tgt_token}"

                if src_id in node_id_set and tgt_id in node_id_set:
                    links.append(Link(source=src_id, target=tgt_id, weight=weight).model_dump())

            # Decode target logit tokens for contrib_map labels
            target_logit_tokens: list[str] | None = None
            if ci_idx < len(self.target_logits) and self.tokenizer is not None:
                target_logit_tokens = [
                    self.tokenizer.decode([t]) for t in self.target_logits[ci_idx]
                ]

            file_slug = f"{slug}_{i}"
            metadata = Metadata(
                slug=file_slug,
                scan=self.model_id or "unknown",
                transcoder_list=[],
                prompt_tokens=prompt_tokens,
                prompt=prompt_str,
                node_threshold=max_influence * 0.5 if max_influence > 0 else None,
                target_logit_tokens=target_logit_tokens,
            )
            # Build supernodes from cluster assignments
            supernodes: list[list[str]] = []
            pinned_ids: list[str] = []
            if self._cluster_map:
                # Group node_ids by cluster
                cluster_to_node_ids: dict[str, list[str]] = {}
                for nkey, nids in neuron_key_to_node_ids.items():
                    cluster_label = neuron_to_cluster.get(nkey)
                    if cluster_label and cluster_label != "-1":
                        cluster_to_node_ids.setdefault(cluster_label, []).extend(nids)

                for cl, nids in cluster_to_node_ids.items():
                    if cl == UNCLUSTERED_CLUSTER_ID:
                        continue
                    summary_labels = getattr(self, "_cluster_summary_labels", {})
                    if summary_labels and cl in summary_labels:
                        cl_label = f"{cl}: {summary_labels[cl]}"
                    elif self._cluster_descriptions and cl in self._cluster_descriptions:
                        cl_label = self._cluster_descriptions[cl]
                    elif cl.isdigit():
                        cl_label = f"Cluster {cl}"
                    else:
                        cl_label = cl
                    supernodes.append([cl_label] + nids)
                    pinned_ids.extend(nids)

                # Pin embedding/logit nodes that have edges to/from pinned nodes
                pinned_set = set(pinned_ids)
                link_node_ids: set[str] = set()
                for link in links:
                    src, tgt = link["source"], link["target"]
                    if src in pinned_set or tgt in pinned_set:
                        link_node_ids.add(src)
                        link_node_ids.add(tgt)
                for node in nodes:
                    if node.feature_type in ("embedding", "logit"):
                        if node.node_id in link_node_ids:
                            pinned_ids.append(node.node_id)

            q_params = QParams(
                pinnedIds=pinned_ids,
                supernodes=supernodes,
                linkType="both",
                clickedId="",
                sg_pos="",
            )

            model = Model(
                metadata=metadata,
                qParams=q_params,
                nodes=nodes,
                links=links,
            )

            out_path = output_dir / f"{file_slug}.json"
            with open(out_path, "w") as f:
                json.dump(model.model_dump(), f)
            written_files.append(out_path)

            display_label = lbl.rsplit("___", 1)[0] if "___" in lbl else lbl
            graph_metadata_entries.append(
                {
                    "slug": file_slug,
                    "scan": self.model_id or "unknown",
                    "prompt": display_label,
                    "node_threshold": metadata.node_threshold,
                }
            )

        # Write graph-metadata.json
        meta_path = output_dir / "graph-metadata.json"
        with open(meta_path, "w") as f:
            json.dump({"graphs": graph_metadata_entries}, f)
        written_files.append(meta_path)

        logger.info(f"Exported {len(unique_labels)} graphs to {output_dir}")
        return written_files

    def serve(self, port: int = 8032, slug: str = "circuit") -> Any:
        """Export and serve circuit in circuit-tracer frontend.

        Args:
            port: Port to serve on.
            slug: Base slug for filenames.

        Returns:
            Server object with .stop() to shut down.
        """
        import tempfile

        from circuits.frontend.server import serve as _serve

        temp_dir = tempfile.mkdtemp(prefix="circuit_tracer_")
        self.export_to_circuit_tracer(temp_dir, slug)

        server = _serve(data_dir=temp_dir, port=port)
        first_slug = f"{slug}_0"
        print(f"Circuit viewer: http://localhost:{port}/index.html?slug={first_slug}")
        return server

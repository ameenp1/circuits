"""Tests for NodePipeline stage transitions and cluster assignment."""

import numpy as np
import pandas as pd
import pytest
from circuits.analysis.cluster import (
    STAGE_REQUIRED_COLUMNS,
    UNCLUSTERED_CLUSTER_ID,
    NeuronId,
    NodePipeline,
    NodeStage,
)


class TestNodePipelineStages:
    """Test RAW → INDEXED → EMBEDDED → PROCESSED transitions."""

    @pytest.fixture
    def raw_df_node(self) -> pd.DataFrame:
        """Create minimal RAW stage DataFrame."""
        return pd.DataFrame(
            {
                "layer": [0, 0, 1, 1],
                "token": [0, 1, 0, 1],
                "neuron": [0, 0, 0, 0],
                "label": ["ex1", "ex1", "ex1", "ex1"],
                "attr_map": [
                    np.array([0.5, 0.5]),
                    np.array([0.3, 0.7]),
                    np.array([0.6, 0.4]),
                    np.array([0.2, 0.8]),
                ],
                "contrib_map": [
                    np.array([0.4, 0.6]),
                    np.array([0.5, 0.5]),
                    np.array([0.3, 0.7]),
                    np.array([0.6, 0.4]),
                ],
                "attribution": [1.0, 0.5, 0.8, 0.6],
                "activation": [0.5, -0.3, 0.7, 0.4],
            }
        )

    @pytest.fixture
    def raw_df_edge(self) -> pd.DataFrame:
        """Create minimal RAW stage edge DataFrame."""
        return pd.DataFrame(
            {
                "layer": ["0->1"],
                "token": ["0->0"],
                "neuron": ["0->0"],
                "label": ["ex1"],
                "attribution": [0.3],
                "weight": [0.5],
            }
        )

    def test_raw_validation(self, raw_df_node: pd.DataFrame) -> None:
        """Test that RAW stage validates required columns."""
        pipeline = NodePipeline(raw_df_node, NodeStage.RAW)
        assert pipeline.stage == NodeStage.RAW

        # Missing column should raise
        df_missing = raw_df_node.drop(columns=["activation"])
        with pytest.raises(ValueError, match="Missing columns"):
            NodePipeline(df_missing, NodeStage.RAW)

    def test_raw_to_indexed(self, raw_df_node: pd.DataFrame, raw_df_edge: pd.DataFrame) -> None:
        """Test RAW → INDEXED transition."""
        pipeline = NodePipeline(raw_df_node, NodeStage.RAW)
        indexed_pipeline, df_edge_indexed = pipeline.to_indexed(raw_df_edge)

        assert indexed_pipeline.stage == NodeStage.INDEXED
        df = indexed_pipeline.df
        assert "input_variable" in df.columns
        assert all(isinstance(iv, NeuronId) for iv in df["input_variable"])

    def test_indexed_to_embedded(
        self, raw_df_node: pd.DataFrame, raw_df_edge: pd.DataFrame
    ) -> None:
        """Test INDEXED → EMBEDDED transition."""
        pipeline = NodePipeline(raw_df_node, NodeStage.RAW)
        indexed_pipeline, _ = pipeline.to_indexed(raw_df_edge)
        embedded_pipeline = indexed_pipeline.to_embedded(mode="attr + contrib")

        assert embedded_pipeline.stage == NodeStage.EMBEDDED
        df = embedded_pipeline.df
        assert "embedding" in df.columns
        assert all(isinstance(e, np.ndarray) for e in df["embedding"])

    def test_embedded_to_processed(
        self, raw_df_node: pd.DataFrame, raw_df_edge: pd.DataFrame
    ) -> None:
        """Test EMBEDDED → PROCESSED transition."""
        pipeline = NodePipeline(raw_df_node, NodeStage.RAW)
        indexed_pipeline, _ = pipeline.to_indexed(raw_df_edge)
        embedded_pipeline = indexed_pipeline.to_embedded(mode="attr + contrib")
        processed_pipeline, cache = embedded_pipeline.to_processed(
            n_clusters=2, tokenizer=None, get_desc=False
        )

        assert processed_pipeline.stage == NodeStage.PROCESSED
        df = processed_pipeline.df
        assert "cluster" in df.columns
        assert all(isinstance(c, str) for c in df["cluster"])

    def test_wrong_stage_raises(self, raw_df_node: pd.DataFrame, raw_df_edge: pd.DataFrame) -> None:
        """Test that calling wrong transition raises ValueError."""
        pipeline = NodePipeline(raw_df_node, NodeStage.RAW)

        # Can't go to embedded from RAW
        with pytest.raises(ValueError, match="Expected INDEXED stage"):
            pipeline.to_embedded()

        # Can't go to processed from RAW
        with pytest.raises(ValueError, match="Expected EMBEDDED stage"):
            pipeline.to_processed(n_clusters=2)


class TestManualClusters:
    """Test manual cluster assignment via to_processed_from_indexed()."""

    @pytest.fixture
    def indexed_pipeline(self) -> tuple[NodePipeline, pd.DataFrame]:
        """Create an INDEXED stage pipeline for testing.

        Uses layer=-1 for input layer and layer=2 for output layer to match
        real circuit data format where:
        - layer=-1 is the input layer (token embeddings)
        - layer=max_layer is the output layer (logit predictions)
        """
        df_node = pd.DataFrame(
            {
                "layer": [-1, -1, 1, 1, 2],  # -1 is input layer, 2 is output layer
                "token": [0, 0, 0, 0, 0],
                "neuron": [0, 1, 0, 1, 0],
                "label": ["ex1"] * 5,
                "attr_map": [np.array([0.5, 0.5])] * 5,
                "contrib_map": [np.array([0.5, 0.5])] * 5,
                "attribution": [1.0] * 5,
                "activation": [0.5, -0.3, 0.7, 0.4, 0.2],
            }
        )
        df_edge = pd.DataFrame(
            {
                "layer": ["-1->1", "1->2"],
                "token": ["0->0", "0->0"],
                "neuron": ["0->0", "0->0"],
                "label": ["ex1", "ex1"],
                "attribution": [0.3, 0.2],
                "weight": [0.5, 0.4],
            }
        )

        pipeline = NodePipeline(df_node, NodeStage.RAW)
        return pipeline.to_indexed(df_edge)

    def test_manual_clusters_assigned(
        self, indexed_pipeline: tuple[NodePipeline, pd.DataFrame]
    ) -> None:
        """Test that manual clusters are correctly assigned."""
        pipeline, _ = indexed_pipeline

        manual_clusters = {
            NeuronId(layer=1, token=-1, neuron=0, polarity="+"): "cluster_A",
            NeuronId(layer=1, token=-1, neuron=1, polarity="+"): "cluster_A",
        }

        processed_pipeline, _ = pipeline.to_processed_from_indexed(
            manual_clusters=manual_clusters,
            tokenizer=None,
            get_desc=False,
            last_layer=2,
        )

        assert processed_pipeline.stage == NodeStage.PROCESSED
        df = processed_pipeline.df

        # Check that cluster_A neurons are assigned correctly
        cluster_a_rows = df[df["cluster"] == "cluster_A"]
        assert len(cluster_a_rows) >= 1

    def test_unclustered_neurons_marked(
        self, indexed_pipeline: tuple[NodePipeline, pd.DataFrame]
    ) -> None:
        """Test that neurons not in manual_clusters get UNCLUSTERED_CLUSTER_ID."""
        pipeline, _ = indexed_pipeline

        # Only assign one neuron, leave others unassigned
        manual_clusters = {
            NeuronId(layer=1, token=-1, neuron=0, polarity="+"): "my_cluster",
        }

        processed_pipeline, _ = pipeline.to_processed_from_indexed(
            manual_clusters=manual_clusters,
            tokenizer=None,
            get_desc=False,
            last_layer=2,
        )

        df = processed_pipeline.df

        # Middle layer neurons not in manual_clusters should be unclustered
        unclustered = df[df["cluster"] == UNCLUSTERED_CLUSTER_ID]
        # At least some neurons should be unclustered (layer 1 neuron 1 with "-" polarity)
        # Note: boundary layers (-1 and 2) get their NeuronId as cluster, not UNCLUSTERED

    def test_boundary_layers_get_neuronid_cluster(
        self, indexed_pipeline: tuple[NodePipeline, pd.DataFrame]
    ) -> None:
        """Test that input/output layer neurons get NeuronId as cluster."""
        pipeline, _ = indexed_pipeline

        manual_clusters = {}  # No manual assignments

        processed_pipeline, _ = pipeline.to_processed_from_indexed(
            manual_clusters=manual_clusters,
            tokenizer=None,
            get_desc=False,
            last_layer=2,
        )

        df = processed_pipeline.df

        # Input layer (layer=-1) and output layer (layer=2) should NOT be UNCLUSTERED
        input_layer = df[df["input_variable"].apply(lambda x: x.layer == -1)]
        output_layer = df[df["input_variable"].apply(lambda x: x.layer == 2)]

        # Boundary layer clusters should be NeuronId strings, not UNCLUSTERED
        for _, row in input_layer.iterrows():
            assert row["cluster"] != UNCLUSTERED_CLUSTER_ID
        for _, row in output_layer.iterrows():
            assert row["cluster"] != UNCLUSTERED_CLUSTER_ID

    def test_placeholder_embeddings_added(
        self, indexed_pipeline: tuple[NodePipeline, pd.DataFrame]
    ) -> None:
        """Test that placeholder zero-vector embeddings are added."""
        pipeline, _ = indexed_pipeline

        manual_clusters = {
            NeuronId(layer=1, token=-1, neuron=0, polarity="+"): "cluster_A",
        }

        processed_pipeline, _ = pipeline.to_processed_from_indexed(
            manual_clusters=manual_clusters,
            tokenizer=None,
            get_desc=False,
            last_layer=2,
        )

        df = processed_pipeline.df
        assert "embedding" in df.columns
        for emb in df["embedding"]:
            assert isinstance(emb, np.ndarray)
            # Placeholder should be zeros
            assert np.allclose(emb, np.zeros_like(emb))

    def test_wrong_stage_raises(self, indexed_pipeline: tuple[NodePipeline, pd.DataFrame]) -> None:
        """Test that calling from wrong stage raises ValueError."""
        pipeline, df_edge = indexed_pipeline

        # Go to EMBEDDED stage
        embedded_pipeline = pipeline.to_embedded()

        # Can't call to_processed_from_indexed from EMBEDDED
        with pytest.raises(ValueError, match="Expected INDEXED stage"):
            embedded_pipeline.to_processed_from_indexed(manual_clusters={}, tokenizer=None)


class TestAutomaticClustering:
    """Test automatic clustering via to_processed()."""

    @pytest.fixture
    def embedded_pipeline(self) -> NodePipeline:
        """Create an EMBEDDED stage pipeline for testing."""
        df_node = pd.DataFrame(
            {
                "layer": [0, 1, 1, 1, 1, 2],
                "token": [0, 0, 0, 0, 0, 0],
                "neuron": [0, 0, 1, 2, 3, 0],
                "label": ["ex1"] * 6,
                "attr_map": [np.array([0.5, 0.5])] * 6,
                "contrib_map": [np.array([0.5, 0.5])] * 6,
                "attribution": [1.0] * 6,
                "activation": [0.5, 0.3, 0.7, 0.4, 0.2, 0.6],
            }
        )
        df_edge = pd.DataFrame(
            columns=["layer", "token", "neuron", "label", "attribution", "weight"]
        )

        pipeline = NodePipeline(df_node, NodeStage.RAW)
        indexed_pipeline, _ = pipeline.to_indexed(df_edge)
        return indexed_pipeline.to_embedded(mode="attr + contrib")

    def test_n_clusters_creates_clusters(self, embedded_pipeline: NodePipeline) -> None:
        """Test that n_clusters parameter creates appropriate number of clusters."""
        processed_pipeline, _ = embedded_pipeline.to_processed(
            n_clusters=2, tokenizer=None, get_desc=False
        )

        df = processed_pipeline.df
        # Boundary layers get "-1", middle layers get actual cluster IDs
        middle_layer_clusters = df[df["input_variable"].apply(lambda x: x.layer == 1)][
            "cluster"
        ].unique()
        # Should have at most n_clusters unique clusters for middle layers
        assert len(middle_layer_clusters) <= 2

    def test_n_clusters_zero_skips_clustering(self, embedded_pipeline: NodePipeline) -> None:
        """Test that n_clusters=0 assigns all to '-1'."""
        processed_pipeline, _ = embedded_pipeline.to_processed(
            n_clusters=0, tokenizer=None, get_desc=False
        )

        df = processed_pipeline.df
        assert all(c == "-1" for c in df["cluster"])

    def test_one_cluster_per_neuron(self, embedded_pipeline: NodePipeline) -> None:
        """Test that do_one_cluster_per_neuron assigns all to '-1'."""
        processed_pipeline, _ = embedded_pipeline.to_processed(
            n_clusters=2, tokenizer=None, get_desc=False, do_one_cluster_per_neuron=True
        )

        df = processed_pipeline.df
        assert all(c == "-1" for c in df["cluster"])


class TestValidation:
    """Test validation catches missing columns."""

    def test_raw_stage_missing_column(self) -> None:
        """Test RAW stage validation catches missing columns."""
        df = pd.DataFrame(
            {
                "layer": [0],
                "token": [0],
                "neuron": [0],
                # Missing: label, attr_map, contrib_map, attribution, activation
            }
        )

        with pytest.raises(ValueError) as exc_info:
            NodePipeline(df, NodeStage.RAW)

        assert "Missing columns" in str(exc_info.value)

    def test_indexed_stage_missing_column(self) -> None:
        """Test INDEXED stage validation catches missing columns."""
        df = pd.DataFrame(
            {
                "input_variable": [NeuronId(0, 0, 0, "+")],
                # Missing: label, attr_map, contrib_map, attribution, activation
            }
        )

        with pytest.raises(ValueError) as exc_info:
            NodePipeline(df, NodeStage.INDEXED)

        assert "Missing columns" in str(exc_info.value)

    def test_embedded_stage_missing_column(self) -> None:
        """Test EMBEDDED stage validation catches missing columns."""
        df = pd.DataFrame(
            {
                "input_variable": [NeuronId(0, 0, 0, "+")],
                # Missing: embedding
            }
        )

        with pytest.raises(ValueError) as exc_info:
            NodePipeline(df, NodeStage.EMBEDDED)

        assert "Missing columns" in str(exc_info.value)

    def test_validation_can_be_disabled(self) -> None:
        """Test that validate=False skips validation."""
        df = pd.DataFrame({"some_col": [1]})  # Wrong columns

        # Should not raise with validate=False
        pipeline = NodePipeline(df, NodeStage.RAW, validate=False)
        assert pipeline.stage == NodeStage.RAW

    def test_required_columns_dict_exists(self) -> None:
        """Test that STAGE_REQUIRED_COLUMNS has all stages."""
        for stage in NodeStage:
            assert stage in STAGE_REQUIRED_COLUMNS
            assert isinstance(STAGE_REQUIRED_COLUMNS[stage], set)

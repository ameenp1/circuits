"""Multi-view similarity clustering for circuits.

Instead of concatenating per-label attr_map/contrib_map into one high-dimensional embedding,
this approach computes pairwise neuron similarity within each CI (label) independently,
then aggregates across CIs for a more robust similarity signal.
"""

import logging
from collections import Counter
from typing import Any, Literal, NamedTuple, cast

import numpy as np
import pandas as pd
from circuits.analysis.cluster import NeuronId, df_sum_over_tokens, unit_norm
from numpy.typing import NDArray
from sklearn.cluster import SpectralClustering

logger = logging.getLogger(__name__)

try:
    import igraph  # type: ignore
    import leidenalg  # type: ignore

    HAS_LEIDEN = True
except ImportError:
    HAS_LEIDEN = False


class MultiviewSimilarity(NamedTuple):
    """Result of compute_per_ci_similarities."""

    sim_matrix: NDArray[np.float64]
    overlap_counts: NDArray[np.int64]
    had_overlap: NDArray[np.bool_]
    neuron_ids: list[NeuronId]


def compute_per_ci_similarities(
    df_indexed: pd.DataFrame,
    sum_over_tokens: bool = True,
    repr_types: tuple[str, ...] = ("attr_map", "contrib_map"),
    weight_by_attribution: bool = False,
    use_attributions_as_view: bool = False,
    combine: Literal["mean", "harmonic"] = "mean",
    verbose: bool = False,
) -> MultiviewSimilarity:
    """Compute mean pairwise cosine similarity across CIs (labels).

    For each label and representation type, gathers the neurons present in that label,
    L2-normalizes their representation vectors, computes pairwise cosine similarity,
    and accumulates into running sum/count matrices.

    Args:
        df_indexed: INDEXED stage DataFrame with input_variable (NeuronId), label,
            attr_map, contrib_map columns.
        sum_over_tokens: If True, first collapse token positions via df_sum_over_tokens.
        repr_types: Which representation columns to use.
        weight_by_attribution: If True, weight each CI's contribution by the geometric
            mean of the two neurons' |attribution| values in that CI. This upweights
            similarity evidence from CIs where both neurons are highly important.
        use_attributions_as_view: If True, build an additional "importance profile" view
            where each neuron gets a vector of its |attribution| across all CIs (0 for
            CIs where the neuron is absent). Cosine similarity on these profiles is
            added to the aggregation as an extra term, providing a complete (non-sparse)
            similarity signal.
        combine: How to combine per-repr-type similarities.
            "mean": Arithmetic mean across repr types (default).
            "harmonic": Clamp each view to [0, inf), then take harmonic mean. This
                requires both attr and contrib similarity to be high for a high score;
                if either view is near zero or negative, the combined score is near zero.

    Returns:
        MultiviewSimilarity with sim_matrix, overlap_counts, had_overlap, neuron_ids.
    """
    df = df_indexed.copy()

    if sum_over_tokens:
        # Need layer/token/neuron/polarity columns for df_sum_over_tokens
        # Re-extract from input_variable
        df["layer"] = df["input_variable"].apply(lambda x: x.layer)
        df["token"] = df["input_variable"].apply(lambda x: x.token)
        df["neuron"] = df["input_variable"].apply(lambda x: x.neuron)
        df["polarity"] = df["input_variable"].apply(lambda x: x.polarity)
        df_summed, _ = df_sum_over_tokens(df, pd.DataFrame())
        # Reconstruct input_variable after summing
        df_summed["input_variable"] = df_summed.apply(
            lambda x: NeuronId(
                layer=x["layer"], token=x["token"], neuron=x["neuron"], polarity=x["polarity"]
            ),
            axis=1,
        )
        df = df_summed

    # Build neuron index
    all_neuron_ids = sorted(df["input_variable"].unique())
    neuron_to_idx = {nid: i for i, nid in enumerate(all_neuron_ids)}
    n = len(all_neuron_ids)

    # Track per-repr-type similarities separately for non-mean combination modes
    per_repr_sim_sum: dict[str, NDArray[np.float64]] = {}
    per_repr_weight_sum: dict[str, NDArray[np.float64]] = {}
    per_repr_overlap: dict[str, NDArray[np.int64]] = {}
    for rt in repr_types:
        per_repr_sim_sum[rt] = np.zeros((n, n), dtype=np.float64)
        per_repr_weight_sum[rt] = np.zeros((n, n), dtype=np.float64)
        per_repr_overlap[rt] = np.zeros((n, n), dtype=np.int64)

    overlap_counts = np.zeros((n, n), dtype=np.int64)

    labels = df["label"].unique()
    logger.info(
        "Computing similarities across %d labels and %d neurons "
        "(weight_by_attribution=%s, combine=%s)",
        len(labels),
        n,
        weight_by_attribution,
        combine,
    )

    from tqdm import tqdm

    # Pre-group by label for fast iteration (avoids repeated df filtering)
    grouped = {label: group for label, group in df.groupby("label") if len(group) >= 2}

    label_iter = tqdm(grouped.items(), desc="Computing similarities", disable=not verbose)
    for label, df_label in label_iter:
        neuron_ids_in_label = df_label["input_variable"].tolist()
        idx_arr = np.array([neuron_to_idx[nid] for nid in neuron_ids_in_label])

        # Precompute attribution weights for this CI
        if weight_by_attribution:
            attributions = np.abs(df_label["attribution"].values.astype(np.float64))
            attr_weights = np.sqrt(np.outer(attributions, attributions))
        else:
            attr_weights = None

        # Outer index arrays for vectorized accumulation
        ii, jj = np.meshgrid(idx_arr, idx_arr, indexing="ij")

        for repr_type in repr_types:
            vecs_raw = cast(list[NDArray[np.float32]], df_label[repr_type].tolist())
            vecs = [np.atleast_1d(v) for v in vecs_raw]

            # Row-normalize attr_map (matching compute_embeddings behavior)
            if repr_type == "attr_map":
                vecs = [v / np.where(v.sum() == 0, 1.0, v.sum()) for v in vecs]

            # L2-normalize
            vecs_normed = [unit_norm(v) for v in vecs]

            # Stack into matrix and compute pairwise cosine similarities
            mat = np.stack(vecs_normed)  # (m, d)
            cos_sim = mat @ mat.T  # (m, m)

            # Vectorized accumulation via np.add.at
            if attr_weights is not None:
                np.add.at(per_repr_sim_sum[repr_type], (ii, jj), attr_weights * cos_sim)
                np.add.at(per_repr_weight_sum[repr_type], (ii, jj), attr_weights)
            else:
                np.add.at(per_repr_sim_sum[repr_type], (ii, jj), cos_sim)
            np.add.at(per_repr_overlap[repr_type], (ii, jj), 1)
            np.add.at(overlap_counts, (ii, jj), 1)

    # Compute per-repr-type mean similarities
    per_repr_mean: dict[str, NDArray[np.float64]] = {}
    per_repr_had_overlap: dict[str, NDArray[np.bool_]] = {}
    for rt in repr_types:
        if weight_by_attribution:
            mask = per_repr_weight_sum[rt] > 0
            per_repr_mean[rt] = np.where(mask, per_repr_sim_sum[rt] / per_repr_weight_sum[rt], 0.0)
        else:
            mask = per_repr_overlap[rt] > 0
            per_repr_mean[rt] = np.where(mask, per_repr_sim_sum[rt] / per_repr_overlap[rt], 0.0)
        per_repr_had_overlap[rt] = mask

    # Combine across repr types
    had_overlap = overlap_counts > 0
    if combine == "harmonic" and len(repr_types) >= 2:
        # Clamp each view to non-negative, then harmonic mean.
        # HM(a, b) = 2ab / (a + b), with HM = 0 if either is 0.
        clamped = {rt: np.maximum(per_repr_mean[rt], 0.0) for rt in repr_types}
        if len(repr_types) == 2:
            rt_a, rt_b = repr_types
            a, b = clamped[rt_a], clamped[rt_b]
            denom = a + b
            sim_matrix = np.where(denom > 0, 2.0 * a * b / denom, 0.0)
        else:
            # General harmonic mean for N views: N / sum(1/x_i), with 0 if any x_i == 0
            stacked = np.stack([clamped[rt] for rt in repr_types])  # (N, n, n)
            any_zero = np.any(stacked == 0, axis=0)
            # Safe reciprocal (avoid div by zero, will be masked out)
            with np.errstate(divide="ignore"):
                recip_sum = np.sum(1.0 / np.where(stacked > 0, stacked, 1.0), axis=0)
            sim_matrix = np.where(any_zero, 0.0, len(repr_types) / recip_sum)
        # For pairs where only one view has overlap, fall back to that view (clamped)
        for rt in repr_types:
            only_this = per_repr_had_overlap[rt] & ~np.all(
                [per_repr_had_overlap[other] for other in repr_types if other != rt], axis=0
            )
            sim_matrix = np.where(only_this, clamped[rt], sim_matrix)
    else:
        # Arithmetic mean: sum all repr types and divide by total overlap count
        total_sim_sum = sum(per_repr_sim_sum[rt] for rt in repr_types)
        if weight_by_attribution:
            total_weight = sum(per_repr_weight_sum[rt] for rt in repr_types)
            sim_matrix = np.where(total_weight > 0, total_sim_sum / total_weight, 0.0)
        else:
            sim_matrix = np.where(had_overlap, total_sim_sum / overlap_counts, 0.0)

    # Attribution profile view: each neuron gets a vector of |attribution| across all CIs
    if use_attributions_as_view:
        labels_list = sorted(df["label"].unique())
        label_to_col = {label: j for j, label in enumerate(labels_list)}
        n_labels = len(labels_list)

        # Build (n_neurons, n_labels) importance profile matrix
        profile_matrix = np.zeros((n, n_labels), dtype=np.float64)
        for _, row in df.iterrows():
            nid = row["input_variable"]
            idx = neuron_to_idx[nid]
            col = label_to_col[row["label"]]
            profile_matrix[idx, col] += abs(float(row["attribution"]))

        # L2-normalize profiles and compute cosine similarity
        norms = np.linalg.norm(profile_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        profile_normed = profile_matrix / norms
        profile_sim = profile_normed @ profile_normed.T  # (n, n), complete

        # Blend: average the per-CI sim_matrix with the profile similarity.
        # For pairs with no per-CI overlap, the profile sim fills in entirely.
        # For pairs with overlap, take the mean of both signals.
        has_per_ci = had_overlap
        sim_matrix = np.where(
            has_per_ci,
            (sim_matrix + profile_sim) / 2.0,
            profile_sim,
        )
        # Update had_overlap: profile view covers all pairs with nonzero profiles
        profile_nonzero = (np.linalg.norm(profile_matrix, axis=1) > 0).astype(bool)
        had_overlap = had_overlap | np.outer(profile_nonzero, profile_nonzero)

        logger.info(
            "Attribution profile view: %d labels, overlap coverage %.1f%% -> %.1f%%",
            n_labels,
            (overlap_counts > 0).mean() * 100,
            had_overlap.mean() * 100,
        )

    return MultiviewSimilarity(
        sim_matrix=sim_matrix,
        overlap_counts=overlap_counts,
        had_overlap=had_overlap,
        neuron_ids=all_neuron_ids,
    )


def spectral_cluster(
    sim_matrix: NDArray[np.float64],
    n_clusters: int,
    random_state: int = 42,
    use_linear_shift: bool = False,
    unnormalized: bool = False,
) -> NDArray[np.int64]:
    """Spectral clustering on a precomputed similarity matrix.

    Args:
        sim_matrix: Pairwise similarity matrix with values in [-1, 1].
        n_clusters: Number of clusters.
        random_state: Random seed.
        use_linear_shift: If True, use linear shift (S+1)/2 instead of max(0, S).
            The linear shift maps uncorrelated pairs (S~0) to affinity 0.5, which can
            obscure cluster structure. Default max(0, S) is more aggressive.
        unnormalized: If True, use the unnormalized graph Laplacian. Default uses the
            normalized Laplacian, which corrects for degree bias (high-degree neurons
            dominating the objective).
    """
    if use_linear_shift:
        sim_nn = (sim_matrix + 1.0) / 2.0
    else:
        sim_nn = np.maximum(sim_matrix, 0.0)
    np.fill_diagonal(sim_nn, 1.0)

    clusterer = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=random_state,
        assign_labels="kmeans",
        n_init=10,
    )
    if not unnormalized:
        # Compute normalized Laplacian eigenvectors manually, then run k-means.
        # sklearn's SpectralClustering uses the unnormalized Laplacian by default
        # with precomputed affinity, so we handle normalization ourselves.
        from sklearn.cluster import KMeans

        degree = sim_nn.sum(axis=1)
        # Guard against zero-degree nodes
        d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
        # Normalized Laplacian: L_norm = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}
        sim_normalized = sim_nn * np.outer(d_inv_sqrt, d_inv_sqrt)
        np.fill_diagonal(sim_normalized, 1.0)
        # Eigendecompose the normalized affinity (top-k eigenvectors = bottom-k of L_norm)
        eigenvalues, eigenvectors = np.linalg.eigh(sim_normalized)
        # Take the k largest eigenvectors of the normalized affinity
        embedding = eigenvectors[:, -n_clusters:]
        # Row-normalize the embedding
        row_norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        row_norms = np.where(row_norms == 0, 1.0, row_norms)
        embedding = embedding / row_norms
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        return km.fit_predict(embedding).astype(np.int64)

    return clusterer.fit_predict(sim_nn)


def leiden_cluster(
    sim_matrix: NDArray[np.float64],
    knn: int = 10,
    resolution: float = 1.0,
    random_state: int = 42,
) -> NDArray[np.int64]:
    """Leiden clustering on a kNN-sparsified similarity graph.

    Requires igraph and leidenalg packages.
    """
    if not HAS_LEIDEN:
        raise ImportError(
            "leiden_cluster requires igraph and leidenalg. "
            "Install them with: uv pip install igraph leidenalg"
        )

    n = sim_matrix.shape[0]

    # Build kNN sparse adjacency
    from scipy.sparse import lil_matrix  # type: ignore

    adj = lil_matrix((n, n), dtype=np.float64)

    for i in range(n):
        sims = sim_matrix[i].copy()
        sims[i] = -np.inf  # exclude self
        top_k = np.argsort(sims)[-knn:]
        for j in top_k:
            w = max(sims[j], 0.0)  # zero out negative similarities
            if w > 0:
                adj[i, j] = w
                adj[j, i] = w  # symmetrize

    adj_csr = adj.tocsr()

    # Build igraph from sparse adjacency
    sources, targets = adj_csr.nonzero()
    edges = [(int(s), int(t)) for s, t in zip(sources, targets) if s < t]
    weights = [float(adj_csr[s, t]) for s, t in edges]

    g = igraph.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = weights

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=random_state,
    )

    return np.array(partition.membership, dtype=np.int64)


def compute_laplacian_eigenvalues(
    sim_matrix: NDArray[np.float64],
    n_eigenvalues: int = 20,
    use_linear_shift: bool = False,
    unnormalized: bool = False,
) -> NDArray[np.float64]:
    """Compute smallest eigenvalues of the graph Laplacian for eigengap analysis."""
    if use_linear_shift:
        sim_nn = (sim_matrix + 1.0) / 2.0
        sim_nn = np.maximum(sim_nn, 0.0)
    else:
        sim_nn = np.maximum(sim_matrix, 0.0)
    np.fill_diagonal(sim_nn, 0.0)
    degree = sim_nn.sum(axis=1)
    if unnormalized:
        laplacian = np.diag(degree) - sim_nn
    else:
        d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
        laplacian = np.eye(len(degree)) - sim_nn * np.outer(d_inv_sqrt, d_inv_sqrt)
    eigenvalues = np.linalg.eigvalsh(laplacian)
    return eigenvalues[:n_eigenvalues]


def compute_diagnostics(
    sim_matrix: NDArray[np.float64],
    labels: NDArray[np.int64],
    overlap_counts: NDArray[np.int64],
    use_linear_shift: bool = False,
    unnormalized: bool = False,
) -> dict[str, Any]:
    """Compute diagnostic metrics for a clustering result.

    Returns dict with:
        - cluster_sizes: Counter of cluster sizes
        - eigengap: eigenvalues of graph Laplacian (first 20)
        - overlap_stats: statistics on overlap counts
        - silhouette: silhouette score (if > 1 cluster)
    """
    from sklearn.metrics import silhouette_score

    diagnostics: dict[str, Any] = {}

    # Cluster sizes
    diagnostics["cluster_sizes"] = dict(Counter(int(x) for x in labels))
    diagnostics["n_clusters"] = len(set(labels))

    # Eigengap
    eigenvalues = compute_laplacian_eigenvalues(
        sim_matrix, use_linear_shift=use_linear_shift, unnormalized=unnormalized
    )
    diagnostics["eigenvalues"] = eigenvalues.tolist()
    eigengaps = np.diff(eigenvalues)
    diagnostics["eigengaps"] = eigengaps.tolist()

    # Overlap statistics
    upper_tri = overlap_counts[np.triu_indices_from(overlap_counts, k=1)]
    diagnostics["overlap_mean"] = float(upper_tri.mean())
    diagnostics["overlap_median"] = float(np.median(upper_tri))
    diagnostics["overlap_max"] = int(upper_tri.max())
    diagnostics["overlap_pct_zero"] = float((upper_tri == 0).mean() * 100)

    # Silhouette on similarity matrix
    n_unique = len(set(labels))
    if n_unique > 1 and n_unique < len(labels):
        # Convert similarity to distance for silhouette
        dist_matrix = 1.0 - (sim_matrix + 1.0) / 2.0
        np.fill_diagonal(dist_matrix, 0.0)
        diagnostics["silhouette"] = float(
            silhouette_score(dist_matrix, labels, metric="precomputed")
        )
    else:
        diagnostics["silhouette"] = None

    return diagnostics


def compute_intra_inter_sim_ratio(
    sim_matrix: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> dict[str, float]:
    """Compute mean intra-cluster vs inter-cluster similarity and their ratio."""
    n = len(labels)
    intra_sims: list[float] = []
    inter_sims: list[float] = []

    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                intra_sims.append(float(sim_matrix[i, j]))
            else:
                inter_sims.append(float(sim_matrix[i, j]))

    mean_intra = float(np.mean(intra_sims)) if intra_sims else 0.0
    mean_inter = float(np.mean(inter_sims)) if inter_sims else 0.0
    ratio = mean_intra / mean_inter if mean_inter != 0 else float("inf")

    return {
        "mean_intra_sim": mean_intra,
        "mean_inter_sim": mean_inter,
        "intra_inter_ratio": ratio,
    }


def compute_contrib_sign_conflicts(
    df_indexed: pd.DataFrame,
    labels: NDArray[np.int64],
    neuron_ids: list[NeuronId],
) -> dict[str, float]:
    """Compute % of same-cluster neuron pairs with opposing contrib_map signs.

    For each pair of neurons in the same cluster, looks at all CIs where both
    appear and checks their contrib_map entries (ignoring zeros).

    Returns:
        pct_any_conflict: % of intra-cluster pairs where ANY non-zero contrib
            entry has opposite signs across the two neurons (in any shared CI).
        pct_all_conflict: % of intra-cluster pairs where ALL non-zero contrib
            entries have opposite signs (in every shared CI).
    """
    from circuits.analysis.cluster import df_sum_over_tokens

    df = df_indexed.copy()

    # Sum over tokens if needed (contrib_map should be per-neuron-per-CI)
    if "layer" not in df.columns:
        df["layer"] = df["input_variable"].apply(lambda x: x.layer)
        df["token"] = df["input_variable"].apply(lambda x: x.token)
        df["neuron"] = df["input_variable"].apply(lambda x: x.neuron)
        df["polarity"] = df["input_variable"].apply(lambda x: x.polarity)
    df_summed, _ = df_sum_over_tokens(df, pd.DataFrame())
    df_summed["input_variable"] = df_summed.apply(
        lambda x: NeuronId(
            layer=x["layer"], token=x["token"], neuron=x["neuron"], polarity=x["polarity"]
        ),
        axis=1,
    )

    # Build neuron -> cluster mapping using (layer, neuron, polarity) key
    # to handle mismatched token values between neuron_ids and df_summed
    NeuronKey = tuple[int, int, str]

    def _nkey(nid: NeuronId) -> NeuronKey:
        return (nid.layer, nid.neuron, nid.polarity)

    key_to_cluster: dict[NeuronKey, int] = {}
    key_to_nid: dict[NeuronKey, NeuronId] = {}
    for i, nid in enumerate(neuron_ids):
        k = _nkey(nid)
        key_to_cluster[k] = int(labels[i])
        key_to_nid[k] = nid

    # Build neuron -> {label -> contrib_map}
    nid_to_contribs: dict[NeuronKey, dict[str, NDArray]] = {}
    for _, row in df_summed.iterrows():
        nid = row["input_variable"]
        k = _nkey(nid)
        if k not in key_to_cluster:
            continue
        label = row["label"]
        cm = np.array(row["contrib_map"])
        if k not in nid_to_contribs:
            nid_to_contribs[k] = {}
        nid_to_contribs[k][label] = cm

    # Group neurons by cluster
    cluster_to_keys: dict[int, list[NeuronKey]] = {}
    for k, cl in key_to_cluster.items():
        cluster_to_keys.setdefault(cl, []).append(k)

    n_pairs = 0
    n_any_conflict = 0
    n_all_conflict = 0

    for cl, keys in cluster_to_keys.items():
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                key_a, key_b = keys[i], keys[j]
                contribs_a = nid_to_contribs.get(key_a, {})
                contribs_b = nid_to_contribs.get(key_b, {})

                # Shared CIs
                shared_labels = set(contribs_a.keys()) & set(contribs_b.keys())
                if not shared_labels:
                    continue

                n_pairs += 1
                pair_has_any_conflict = False
                pair_all_conflict = True
                had_comparable = False

                for label in shared_labels:
                    cm_a = contribs_a[label]
                    cm_b = contribs_b[label]
                    # Mask to entries where both are non-zero
                    both_nonzero = (cm_a != 0) & (cm_b != 0)
                    if not both_nonzero.any():
                        continue
                    had_comparable = True
                    signs_differ = (cm_a[both_nonzero] * cm_b[both_nonzero]) < 0
                    if signs_differ.any():
                        pair_has_any_conflict = True
                    if not signs_differ.all():
                        pair_all_conflict = False

                if pair_has_any_conflict:
                    n_any_conflict += 1
                if pair_all_conflict and had_comparable:
                    n_all_conflict += 1

    pct_any = (n_any_conflict / n_pairs * 100) if n_pairs > 0 else 0.0
    pct_all = (n_all_conflict / n_pairs * 100) if n_pairs > 0 else 0.0

    return {
        "n_intra_pairs": n_pairs,
        "pct_any_sign_conflict": pct_any,
        "pct_all_sign_conflict": pct_all,
    }

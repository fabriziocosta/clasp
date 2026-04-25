from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from scalp_lite.graph.hubness import edge_weights
from scalp_lite.graph.params import _coerce_int


def _nearest_neighbors(X: np.ndarray, *, n_neighbors: int, metric: str) -> tuple[np.ndarray, np.ndarray]:
    n_obs = X.shape[0]
    n_query = min(n_neighbors + 1, n_obs)
    model = NearestNeighbors(n_neighbors=n_query, metric=metric)
    distances, indices = model.fit(X).kneighbors(X)

    rows = np.arange(n_obs)
    if n_query > 0 and np.all(indices[:, 0] == rows):
        distances = distances[:, 1:]
        indices = indices[:, 1:]
    else:
        keep = indices != rows[:, None]
        distances = np.stack([row_distances[row_keep][:n_neighbors] for row_distances, row_keep in zip(distances, keep)])
        indices = np.stack([row_indices[row_keep][:n_neighbors] for row_indices, row_keep in zip(indices, keep)])
    return indices, distances


def _candidate_graph(
    indices: np.ndarray,
    distances: np.ndarray,
    *,
    n_edges: int,
    hubness_correction: str,
    hubness_k: int,
    edge_weighting: str,
    neighbor_mode: str,
) -> sparse.csr_matrix:
    n_obs = indices.shape[0]
    candidate_count = indices.shape[1]
    if candidate_count == 0:
        return sparse.csr_matrix((n_obs, n_obs), dtype=np.float32)

    rank_scores = np.tile(np.arange(candidate_count, dtype=float), (n_obs, 1))
    if hubness_correction == "csls":
        local_k = min(hubness_k, candidate_count)
        row_scale = distances[:, :local_k].mean(axis=1)
        corrected_distances = 2.0 * distances - row_scale[:, None] - row_scale[indices]
    elif hubness_correction == "none":
        corrected_distances = distances
    else:
        raise ValueError("hubness_correction must be one of: 'none', 'csls'.")

    if neighbor_mode == "distance":
        scores = corrected_distances
    elif neighbor_mode == "rank":
        scores = rank_scores
    else:
        raise ValueError("neighbor_mode must be one of: 'rank', 'distance'.")

    selected = np.argsort(scores, axis=1)[:, :n_edges]
    rows = np.repeat(np.arange(n_obs), n_edges)
    cols = np.take_along_axis(indices, selected, axis=1).reshape(-1)
    edge_distances = np.take_along_axis(corrected_distances, selected, axis=1).reshape(-1)
    edge_scores = np.take_along_axis(scores, selected, axis=1).reshape(-1)
    keep = np.isfinite(edge_scores)
    vals = edge_weights(edge_distances[keep], edge_weighting=edge_weighting)
    return sparse.csr_matrix((vals.astype(np.float32), (rows[keep], cols[keep])), shape=(n_obs, n_obs))


def build_intra_batch_graph(
    X: np.ndarray,
    *,
    n_neighbors: int,
    metric: str = "euclidean",
    hubness_correction: str = "csls",
    hubness_k: int = 10,
    edge_weighting: str = "distance",
    mutual_neighbors: bool = True,
    neighbor_mode: str = "rank",
) -> sparse.csr_matrix:
    """Build a symmetric within-batch kNN connectivity graph."""
    n_neighbors = _coerce_int(n_neighbors, name="n_neighbors", minimum=0)
    hubness_k = _coerce_int(hubness_k, name="hubness_k", minimum=1)
    n_obs = X.shape[0]
    if n_obs == 0:
        return sparse.csr_matrix((0, 0), dtype=np.float32)
    if n_obs == 1 or n_neighbors <= 0:
        return sparse.csr_matrix((n_obs, n_obs), dtype=np.float32)

    n_edges = min(n_neighbors, n_obs - 1)
    candidate_count = min(n_obs - 1, max(n_edges, hubness_k if hubness_correction == "csls" else n_edges))
    indices, distances = _nearest_neighbors(X, n_neighbors=candidate_count, metric=metric)
    graph = _candidate_graph(
        indices,
        distances,
        n_edges=n_edges,
        hubness_correction=hubness_correction,
        hubness_k=hubness_k,
        edge_weighting=edge_weighting,
        neighbor_mode=neighbor_mode,
    )
    graph = graph.minimum(graph.T) if mutual_neighbors else graph.maximum(graph.T)
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph

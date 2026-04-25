from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.metrics import pairwise_distances

from scalp_lite.graph.hubness import csls_distances, edge_weights


def _rank_scores(distances: np.ndarray) -> np.ndarray:
    order = np.argsort(distances, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    row_ids = np.arange(distances.shape[0])[:, None]
    ranks[row_ids, order] = np.arange(distances.shape[1], dtype=float)
    ranks[~np.isfinite(distances)] = np.inf
    return ranks + ranks.T


def _select_neighbors(distances: np.ndarray, *, n_neighbors: int, neighbor_mode: str) -> tuple[np.ndarray, np.ndarray]:
    if neighbor_mode == "distance":
        scores = distances
    elif neighbor_mode == "rank":
        scores = _rank_scores(distances)
    else:
        raise ValueError("neighbor_mode must be one of: 'rank', 'distance'.")

    indices = np.argpartition(scores, kth=n_neighbors - 1, axis=1)[:, :n_neighbors]
    row_order = np.arange(scores.shape[0])[:, None]
    edge_scores = scores[row_order, indices]
    order = np.argsort(edge_scores, axis=1)
    indices = np.take_along_axis(indices, order, axis=1)
    edge_distances = distances[row_order, indices]
    return indices, edge_distances


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
    n_obs = X.shape[0]
    if n_obs == 0:
        return sparse.csr_matrix((0, 0), dtype=np.float32)
    if n_obs == 1 or n_neighbors <= 0:
        return sparse.csr_matrix((n_obs, n_obs), dtype=np.float32)

    n_edges = min(n_neighbors, n_obs - 1)
    distances = pairwise_distances(X, metric=metric)
    np.fill_diagonal(distances, np.inf)
    if hubness_correction == "csls":
        distances = csls_distances(distances, k=min(hubness_k, n_obs - 1))
    elif hubness_correction != "none":
        raise ValueError("hubness_correction must be one of: 'none', 'csls'.")

    indices, edge_distances = _select_neighbors(distances, n_neighbors=n_edges, neighbor_mode=neighbor_mode)
    rows = np.repeat(np.arange(n_obs), n_edges)
    cols = indices.reshape(-1)
    vals = edge_weights(edge_distances.reshape(-1), edge_weighting=edge_weighting)
    graph = sparse.csr_matrix((vals.astype(np.float32), (rows, cols)), shape=(n_obs, n_obs))
    graph = graph.minimum(graph.T) if mutual_neighbors else graph.maximum(graph.T)
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph

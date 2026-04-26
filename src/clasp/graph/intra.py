from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.metrics import pairwise_distances

from clasp.graph.hubness import correct_distances, edge_weights
from clasp.graph.params import _coerce_int


def _select_edges(distances: np.ndarray, *, n_edges: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.argpartition(distances, kth=n_edges - 1, axis=1)[:, :n_edges]
    row_ids = np.arange(distances.shape[0])[:, None]
    selected_distances = distances[row_ids, indices]
    order = np.argsort(selected_distances, axis=1)
    indices = np.take_along_axis(indices, order, axis=1)
    selected_distances = np.take_along_axis(selected_distances, order, axis=1)

    rows = np.repeat(np.arange(distances.shape[0]), n_edges)
    cols = indices.reshape(-1)
    vals = selected_distances.reshape(-1)
    keep = np.isfinite(vals)
    return rows[keep], cols[keep], vals[keep]


def _legacy_rank_correction(neighbor_mode: str, rank_correction: bool) -> bool:
    if neighbor_mode == "rank":
        return True
    if neighbor_mode == "distance":
        return rank_correction
    raise ValueError("neighbor_mode must be one of: 'rank', 'distance'.")


def build_intra_batch_graph(
    X: np.ndarray,
    *,
    n_neighbors: int,
    metric: str = "euclidean",
    hubness_correction: str = "csls",
    hubness_k: int = 10,
    rank_correction: bool = True,
    edge_weighting: str = "distance",
    mutual_neighbors: bool = True,
    neighbor_mode: str = "distance",
) -> sparse.csr_matrix:
    """Build a symmetric within-batch kNN connectivity graph."""
    n_neighbors = _coerce_int(n_neighbors, name="n_neighbors", minimum=0)
    hubness_k = _coerce_int(hubness_k, name="hubness_k", minimum=1)
    rank_correction = _legacy_rank_correction(neighbor_mode, rank_correction)
    n_obs = X.shape[0]
    if n_obs == 0:
        return sparse.csr_matrix((0, 0), dtype=np.float32)
    if n_obs == 1 or n_neighbors <= 0:
        return sparse.csr_matrix((n_obs, n_obs), dtype=np.float32)

    n_edges = min(n_neighbors, n_obs - 1)
    distances = pairwise_distances(X, metric=metric)
    np.fill_diagonal(distances, np.inf)
    distances = correct_distances(
        distances,
        hubness_correction=hubness_correction,
        hubness_k=min(hubness_k, n_obs - 1),
        rank_correction=rank_correction,
    )

    rows, cols, edge_distances = _select_edges(distances, n_edges=n_edges)
    vals = edge_weights(edge_distances, edge_weighting=edge_weighting)
    graph = sparse.csr_matrix((vals.astype(np.float32), (rows, cols)), shape=(n_obs, n_obs))
    graph = graph.minimum(graph.T) if mutual_neighbors else graph.maximum(graph.T)
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph

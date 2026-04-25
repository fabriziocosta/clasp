from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from scalp_lite.graph.hubness import csls_distances, edge_weights


def build_intra_batch_graph(
    X: np.ndarray,
    *,
    n_neighbors: int,
    metric: str = "euclidean",
    hubness_correction: str = "csls",
    hubness_k: int = 10,
    edge_weighting: str = "distance",
) -> sparse.csr_matrix:
    """Build a symmetric within-batch kNN connectivity graph."""
    n_obs = X.shape[0]
    if n_obs == 0:
        return sparse.csr_matrix((0, 0), dtype=np.float32)
    if n_obs == 1 or n_neighbors <= 0:
        return sparse.csr_matrix((n_obs, n_obs), dtype=np.float32)

    k = min(n_neighbors + 1, n_obs)
    if hubness_correction == "csls":
        distances = pairwise_distances(X, metric=metric)
        np.fill_diagonal(distances, np.inf)
        corrected = csls_distances(distances, k=min(hubness_k, n_obs - 1))
        indices = np.argpartition(corrected, kth=k - 2, axis=1)[:, : k - 1]
        row_order = np.arange(n_obs)[:, None]
        edge_distances = corrected[row_order, indices]
        order = np.argsort(edge_distances, axis=1)
        indices = np.take_along_axis(indices, order, axis=1)
        edge_distances = np.take_along_axis(edge_distances, order, axis=1)

        rows = np.repeat(np.arange(n_obs), k - 1)
        cols = indices.reshape(-1)
        vals = edge_weights(edge_distances.reshape(-1), edge_weighting=edge_weighting)
        graph = sparse.csr_matrix((vals, (rows, cols)), shape=(n_obs, n_obs))
        graph = graph.maximum(graph.T)
        graph.setdiag(0)
        graph.eliminate_zeros()
        return graph

    if hubness_correction != "none":
        raise ValueError("hubness_correction must be one of: 'none', 'csls'.")

    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    rows = np.repeat(np.arange(n_obs), k - 1)
    cols = indices[:, 1:].reshape(-1)
    vals = edge_weights(distances[:, 1:].reshape(-1), edge_weighting=edge_weighting)
    graph = sparse.csr_matrix((vals.astype(np.float32), (rows, cols)), shape=(n_obs, n_obs))
    graph = graph.maximum(graph.T)
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph

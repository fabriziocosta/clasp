from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def build_intra_batch_graph(
    X: np.ndarray,
    *,
    n_neighbors: int,
    metric: str = "euclidean",
) -> sparse.csr_matrix:
    """Build a symmetric within-batch kNN connectivity graph."""
    n_obs = X.shape[0]
    if n_obs == 0:
        return sparse.csr_matrix((0, 0), dtype=np.float32)
    if n_obs == 1 or n_neighbors <= 0:
        return sparse.csr_matrix((n_obs, n_obs), dtype=np.float32)

    k = min(n_neighbors + 1, n_obs)
    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    rows = np.repeat(np.arange(n_obs), k - 1)
    cols = indices[:, 1:].reshape(-1)
    vals = 1.0 / (1.0 + distances[:, 1:].reshape(-1))
    graph = sparse.csr_matrix((vals.astype(np.float32), (rows, cols)), shape=(n_obs, n_obs))
    graph = graph.maximum(graph.T)
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph

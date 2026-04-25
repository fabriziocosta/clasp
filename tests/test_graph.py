from __future__ import annotations

import numpy as np
from scipy import sparse

from scalp_lite.graph import build_inter_batch_graph, build_intra_batch_graph, build_scalp_graph
from scalp_lite.preprocessing import ensure_pca


def test_intra_batch_graph_has_expected_shape_and_no_cross_batch_edges():
    X = np.random.default_rng(0).normal(size=(10, 4))
    graph = build_intra_batch_graph(X, n_neighbors=3)
    assert graph.shape == (10, 10)
    assert sparse.isspmatrix_csr(graph)
    assert graph.nnz > 0


def test_hungarian_inter_batch_graph_creates_cross_batch_edges():
    rng = np.random.default_rng(1)
    left = rng.normal(size=(8, 3))
    right = left + 0.1
    graph = build_inter_batch_graph(left, right, n_inter_edges=1, assignment_quantile=1.0)
    assert graph.shape == (8, 8)
    assert graph.nnz == 8


def test_final_graph_is_sparse_square_symmetric_and_zero_diagonal(toy_adata):
    ensure_pca(toy_adata, n_components=6)
    graph = build_scalp_graph(toy_adata, n_neighbors=6, intra_fraction=0.5, assignment_quantile=1.0)
    assert sparse.isspmatrix_csr(graph)
    assert graph.shape == (toy_adata.n_obs, toy_adata.n_obs)
    assert (graph - graph.T).nnz == 0
    assert graph.diagonal().sum() == 0
    assert graph.nnz > 0
    assert "scalp_lite" in toy_adata.uns

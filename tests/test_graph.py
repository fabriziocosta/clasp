from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from anndata import AnnData
from scipy import sparse

from scalp_lite.graph import GraphParams, build_inter_batch_graph, build_intra_batch_graph, build_scalp_graph
from scalp_lite.graph.hubness import csls_distances, edge_weights
from scalp_lite.preprocessing import ensure_pca


def test_intra_batch_graph_has_expected_shape_and_no_cross_batch_edges():
    X = np.random.default_rng(0).normal(size=(10, 4))
    graph = build_intra_batch_graph(X, n_neighbors=3)
    assert graph.shape == (10, 10)
    assert sparse.isspmatrix_csr(graph)
    assert graph.nnz > 0


def test_intra_batch_graph_uses_mutual_neighbors_by_default():
    X = np.array([[0.0], [1.0], [3.0]])

    mutual = build_intra_batch_graph(X, n_neighbors=1, hubness_correction="none")
    non_mutual = build_intra_batch_graph(X, n_neighbors=1, hubness_correction="none", mutual_neighbors=False)

    assert mutual[0, 1] > 0
    assert mutual[1, 0] > 0
    assert mutual[1, 2] == 0
    assert mutual[2, 1] == 0
    assert non_mutual[1, 2] > 0
    assert non_mutual[2, 1] > 0


def test_intra_batch_graph_accepts_rank_neighbor_mode():
    X = np.random.default_rng(4).normal(size=(12, 5))

    graph = build_intra_batch_graph(X, n_neighbors=3, neighbor_mode="rank")

    assert graph.shape == (12, 12)
    assert sparse.isspmatrix_csr(graph)
    assert graph.nnz > 0


def test_graph_params_validate_and_coerce_integral_values():
    params = GraphParams(n_neighbors=5.0, n_inter_edges=2.0, hubness_k=3.0)

    assert params.n_neighbors == 5
    assert isinstance(params.n_neighbors, int)
    assert params.n_inter_edges == 2
    assert isinstance(params.n_inter_edges, int)
    assert params.hubness_k == 3
    assert isinstance(params.hubness_k, int)


def test_graph_params_reject_non_integral_edge_counts():
    with pytest.raises(ValueError, match="n_inter_edges must be an integer"):
        GraphParams(n_inter_edges=2.5)


def test_inter_batch_graph_coerces_integral_float_edge_count():
    rng = np.random.default_rng(5)
    left = rng.normal(size=(5, 3))
    right = rng.normal(size=(6, 3))

    graph = build_inter_batch_graph(left, right, n_inter_edges=2.0, assignment_quantile=1.0)

    assert graph.shape == (5, 6)
    assert graph.nnz > 0


def test_hungarian_inter_batch_graph_creates_cross_batch_edges():
    rng = np.random.default_rng(1)
    left = rng.normal(size=(8, 3))
    right = left + 0.1
    graph = build_inter_batch_graph(left, right, n_inter_edges=1, assignment_quantile=1.0)
    assert graph.shape == (8, 8)
    assert graph.nnz == 8


def test_csls_distances_apply_local_scaling():
    distances = np.array(
        [
            [1.0, 2.0, 10.0],
            [2.0, 3.0, 11.0],
        ]
    )

    corrected = csls_distances(distances, k=1)

    expected = np.array(
        [
            [0.0, 1.0, 9.0],
            [1.0, 2.0, 10.0],
        ]
    )
    np.testing.assert_allclose(corrected, expected)


def test_hubness_correction_is_available_for_inter_batch_graph():
    rng = np.random.default_rng(2)
    left = rng.normal(size=(6, 3))
    right = rng.normal(size=(7, 3))

    graph = build_inter_batch_graph(
        left,
        right,
        n_inter_edges=1,
        assignment_quantile=1.0,
        hubness_correction="csls",
        hubness_k=2,
    )

    assert graph.shape == (6, 7)
    assert graph.nnz == 6
    assert np.isfinite(graph.data).all()
    assert np.all(graph.data > 0)


def test_binary_edge_weighting_sets_retained_edges_to_one():
    distances = np.array([0.2, 1.5, -0.4])

    weights = edge_weights(distances, edge_weighting="binary")

    np.testing.assert_array_equal(weights, np.ones(3, dtype=np.float32))


def test_binary_edge_weighting_is_available_for_graph_builders():
    X = np.random.default_rng(3).normal(size=(8, 3))
    intra = build_intra_batch_graph(X, n_neighbors=3, edge_weighting="binary")
    inter = build_inter_batch_graph(X[:4], X[4:], n_inter_edges=1, assignment_quantile=1.0, edge_weighting="binary")

    assert intra.nnz > 0
    assert inter.nnz > 0
    np.testing.assert_array_equal(intra.data, np.ones(intra.nnz, dtype=np.float32))
    np.testing.assert_array_equal(inter.data, np.ones(inter.nnz, dtype=np.float32))


def test_final_graph_is_sparse_square_symmetric_and_zero_diagonal(toy_adata):
    ensure_pca(toy_adata, n_components=6)
    graph = build_scalp_graph(toy_adata, n_neighbors=6, intra_fraction=0.5, assignment_quantile=1.0)
    assert sparse.isspmatrix_csr(graph)
    assert graph.shape == (toy_adata.n_obs, toy_adata.n_obs)
    assert (graph - graph.T).nnz == 0
    assert graph.diagonal().sum() == 0
    assert graph.nnz > 0
    assert "scalp_lite" in toy_adata.uns


def test_final_graph_preserves_original_observation_order_for_interleaved_batches():
    adata = AnnData(
        np.zeros((4, 2)),
        obs=pd.DataFrame({"batch": ["a", "b", "a", "b"]}),
    )
    adata.obsm["X_pca"] = np.array(
        [
            [0.0, 0.0],
            [10.0, 10.0],
            [0.1, 0.0],
            [10.1, 10.0],
        ]
    )

    graph = build_scalp_graph(
        adata,
        n_neighbors=1,
        intra_fraction=1.0,
        n_inter_edges=0,
    )

    assert graph[0, 2] > 0
    assert graph[2, 0] > 0
    assert graph[1, 3] > 0
    assert graph[3, 1] > 0
    assert graph[0, 1] == 0
    assert graph[2, 3] == 0

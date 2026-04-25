from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData
import matplotlib
from scipy import sparse

from scalp_lite import ScalpEstimator

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def test_estimator_to_data_reads_h5ad(tmp_path, toy_adata):
    path = tmp_path / "toy.h5ad"
    toy_adata.write_h5ad(path)

    estimator = ScalpEstimator()
    adata = estimator.to_data(path)

    assert adata.shape == toy_adata.shape


def test_estimator_input_alias_reads_h5ad(tmp_path, toy_adata):
    path = tmp_path / "toy.h5ad"
    toy_adata.write_h5ad(path)

    estimator = ScalpEstimator()
    adata = estimator.input(path)

    assert adata.shape == toy_adata.shape


def test_estimator_save_writes_h5ad(tmp_path, toy_adata):
    path = tmp_path / "saved.h5ad"
    estimator = ScalpEstimator()

    estimator.save(toy_adata, path)

    assert path.exists()
    assert estimator.to_data(path).shape == toy_adata.shape


def test_estimator_preprocess_selects_genes_subsamples_and_adds_pca(toy_adata):
    estimator = ScalpEstimator(n_components=4, random_state=11)

    adata = estimator.preprocess(toy_adata, n_top_genes=5, max_cells=12, min_gene_counts=0)

    assert adata.n_vars == 5
    assert adata.n_obs == 12
    assert estimator.rep_key in adata.obsm
    assert adata.obsm[estimator.rep_key].shape == (12, 4)
    assert adata.obs["batch"].nunique() == 2


def test_estimator_preprocess_filters_cells_by_min_genes():
    adata = AnnData(
        np.array(
            [
                [1, 0, 0, 0],
                [1, 1, 1, 0],
                [0, 2, 2, 2],
            ],
            dtype=float,
        ),
        obs=pd.DataFrame({"batch": ["a", "a", "b"]}),
    )
    estimator = ScalpEstimator(n_components=2)

    result = estimator.preprocess(
        adata,
        n_top_genes=None,
        min_cell_genes=3,
        min_gene_counts=0,
        normalize=False,
    )

    assert result.n_obs == 2
    assert result.obsm["X_pca"].shape == (2, 1)


def test_estimator_preprocess_respects_target_sum_and_log1p():
    adata = AnnData(
        np.array([[1, 1, 0], [0, 2, 2], [3, 0, 3]], dtype=float),
        obs=pd.DataFrame({"batch": ["a", "a", "b"]}),
    )
    estimator = ScalpEstimator(n_components=2)

    result = estimator.preprocess(
        adata,
        n_top_genes=None,
        min_gene_counts=0,
        normalize=True,
        target_sum=100,
        log1p=False,
    )

    assert np.allclose(np.asarray(result.X).sum(axis=1), 100)


def test_estimator_preprocess_accepts_hvg_batch_key(toy_adata):
    estimator = ScalpEstimator(n_components=4)

    adata = estimator.preprocess(
        toy_adata,
        n_top_genes=5,
        min_gene_counts=0,
        normalize=False,
        hvg_batch_key="batch",
    )

    assert adata.n_vars == 5


def test_estimator_preprocess_accepts_variance_hvg_flavor(toy_adata):
    estimator = ScalpEstimator(n_components=4)

    adata = estimator.preprocess(
        toy_adata,
        n_top_genes=5,
        min_gene_counts=0,
        normalize=False,
        hvg_flavor="variance",
    )

    assert adata.n_vars == 5
    assert "scalp_lite_hvg_score" in adata.var


def test_estimator_preprocess_can_create_artificial_batches():
    adata = AnnData(np.arange(24, dtype=float).reshape(6, 4))
    estimator = ScalpEstimator(n_components=2)

    result = estimator.preprocess(
        adata,
        n_top_genes=None,
        min_gene_counts=0,
        normalize=False,
        create_artificial_batch=True,
        artificial_batch_count=3,
    )

    assert list(result.obs["batch"].astype(str)) == ["split_0", "split_1", "split_2", "split_0", "split_1", "split_2"]


def test_estimator_preprocess_can_infer_label_key():
    adata = AnnData(
        np.arange(16, dtype=float).reshape(4, 4),
        obs=pd.DataFrame({"batch": ["a", "a", "b", "b"], "clusters_coarse": ["x", "x", "y", "y"]}),
    )
    estimator = ScalpEstimator(n_components=2)

    result = estimator.preprocess(adata, n_top_genes=None, min_gene_counts=0, normalize=False)

    assert "label" in result.obs
    assert list(result.obs["label"].astype(str)) == ["x", "x", "y", "y"]


def test_estimator_graph_and_embedding_methods(toy_adata):
    estimator = ScalpEstimator(n_components=6, embedding_method="spectral")
    adata = estimator.preprocess(toy_adata, n_top_genes=None)

    graph = estimator.data_to_graph(adata)
    coords = estimator.graph_to_vector(graph)

    assert sparse.isspmatrix_csr(graph)
    assert graph.shape == (adata.n_obs, adata.n_obs)
    assert coords.shape == (adata.n_obs, 2)


def test_estimator_graph_to_vector_accepts_call_overrides(toy_adata):
    estimator = ScalpEstimator(n_components=6, embedding_method="spectral")
    adata = estimator.preprocess(toy_adata, n_top_genes=None)
    graph = estimator.data_to_graph(adata)

    coords = estimator.graph_to_vector(graph, method="spectral", n_components=3, random_state=5)

    assert coords.shape == (adata.n_obs, 3)


def test_estimator_data_to_graph_accepts_call_overrides(toy_adata):
    estimator = ScalpEstimator(n_components=6, n_neighbors=6, embedding_method="spectral")
    adata = estimator.preprocess(toy_adata, n_top_genes=None)

    graph = estimator.data_to_graph(
        adata,
        n_neighbors=4,
        intra_fraction=0.25,
        n_inter_edges=2,
        assignment_quantile=1.0,
        symmetrize=False,
    )

    params = adata.uns["scalp_lite"]["graph"]["parameters"]
    assert graph.shape == (adata.n_obs, adata.n_obs)
    assert params["n_neighbors"] == 4
    assert params["intra_fraction"] == 0.25
    assert params["n_inter_edges"] == 2
    assert params["assignment_quantile"] == 1.0
    assert params["symmetrize"] is False


def test_estimator_embed_combines_graph_and_vector(toy_adata):
    estimator = ScalpEstimator(n_components=6, embedding_method="spectral")
    adata = estimator.preprocess(toy_adata, n_top_genes=None)

    coords = estimator.embed(adata)

    assert coords.shape == (adata.n_obs, 2)


def test_estimator_embed_accepts_graph_overrides(toy_adata):
    estimator = ScalpEstimator(n_components=6, embedding_method="spectral")
    adata = estimator.preprocess(toy_adata, n_top_genes=None)

    coords = estimator.embed(
        adata,
        n_neighbors=4,
        intra_fraction=0.25,
        assignment_quantile=1.0,
        embedding_method="spectral",
        embedding_components=3,
    )

    assert coords.shape == (adata.n_obs, 3)
    assert adata.uns["scalp_lite"]["graph"]["parameters"]["n_neighbors"] == 4


def test_estimator_plot_wraps_embedding_pair(toy_adata):
    estimator = ScalpEstimator()
    toy_adata.obsm["X_scalp"] = toy_adata.X[:, :2]

    axes = estimator.plot(toy_adata)

    assert len(axes) == 2
    assert axes[0].get_title() == "X_scalp by batch"
    assert axes[1].get_title() == "X_scalp by label"
    plt.close(axes[0].figure)

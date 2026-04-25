from __future__ import annotations

from scipy import sparse

from scalp_lite import ScalpEstimator


def test_estimator_input_reads_h5ad(tmp_path, toy_adata):
    path = tmp_path / "toy.h5ad"
    toy_adata.write_h5ad(path)

    estimator = ScalpEstimator()
    adata = estimator.input(path)

    assert adata.shape == toy_adata.shape


def test_estimator_preprocess_selects_genes_subsamples_and_adds_pca(toy_adata):
    estimator = ScalpEstimator(n_components=4, random_state=11)

    adata = estimator.preprocess(toy_adata, n_top_genes=5, max_cells=12, min_gene_counts=0)

    assert adata.n_vars == 5
    assert adata.n_obs == 12
    assert estimator.rep_key in adata.obsm
    assert adata.obsm[estimator.rep_key].shape == (12, 4)
    assert adata.obs["batch"].nunique() == 2


def test_estimator_graph_and_embedding_methods(toy_adata):
    estimator = ScalpEstimator(n_components=6, embedding_method="spectral")
    adata = estimator.preprocess(toy_adata, n_top_genes=None)

    graph = estimator.data_to_graph(adata)
    coords = estimator.graph_to_vector(graph)

    assert sparse.isspmatrix_csr(graph)
    assert graph.shape == (adata.n_obs, adata.n_obs)
    assert coords.shape == (adata.n_obs, 2)


def test_estimator_embed_combines_graph_and_vector(toy_adata):
    estimator = ScalpEstimator(n_components=6, embedding_method="spectral")
    adata = estimator.preprocess(toy_adata, n_top_genes=None)

    coords = estimator.embed(adata)

    assert coords.shape == (adata.n_obs, 2)

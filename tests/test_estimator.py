from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData
import matplotlib
import pytest
from scipy import sparse

from clasp import (
    available_presets,
    ClaspGraphEmbedder,
    ClaspEstimator,
    EstimatorTuneParams,
    GraphTuneParams,
    LatentBOTuneParams,
    PreprocessTuneParams,
    TuneParams,
)
from imste import IteratedMinimumSpanningTreeEmbedder

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def test_estimator_to_data_reads_h5ad(tmp_path, toy_adata):
    path = tmp_path / "toy.h5ad"
    toy_adata.write_h5ad(path)

    estimator = ClaspEstimator()
    adata = estimator.to_data(path)

    assert adata.shape == toy_adata.shape


def test_estimator_input_alias_reads_h5ad(tmp_path, toy_adata):
    path = tmp_path / "toy.h5ad"
    toy_adata.write_h5ad(path)

    estimator = ClaspEstimator()
    adata = estimator.input(path)

    assert adata.shape == toy_adata.shape


def test_estimator_save_writes_h5ad(tmp_path, toy_adata):
    path = tmp_path / "saved.h5ad"
    estimator = ClaspEstimator()

    estimator.save(toy_adata, path)

    assert path.exists()
    assert estimator.to_data(path).shape == toy_adata.shape


def test_estimator_preprocess_selects_genes_subsamples_and_adds_pca(toy_adata):
    estimator = ClaspEstimator(n_components=4, random_state=11)

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
    estimator = ClaspEstimator(n_components=2)

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
    estimator = ClaspEstimator(n_components=2)

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
    estimator = ClaspEstimator(n_components=4)

    adata = estimator.preprocess(
        toy_adata,
        n_top_genes=5,
        min_gene_counts=0,
        normalize=False,
        hvg_batch_key="batch",
    )

    assert adata.n_vars == 5


def test_estimator_preprocess_accepts_variance_hvg_flavor(toy_adata):
    estimator = ClaspEstimator(n_components=4)

    adata = estimator.preprocess(
        toy_adata,
        n_top_genes=5,
        min_gene_counts=0,
        normalize=False,
        hvg_flavor="variance",
    )

    assert adata.n_vars == 5
    assert "clasp_hvg_score" in adata.var


def test_estimator_preprocess_warns_when_scanpy_hvg_falls_back(toy_adata):
    estimator = ClaspEstimator(n_components=4)

    with pytest.warns(RuntimeWarning, match="falling back to variance-based gene selection"):
        adata = estimator.preprocess(
            toy_adata,
            n_top_genes=5,
            min_gene_counts=0,
            normalize=False,
            hvg_flavor="not_a_real_flavor",
            hvg_fallback="variance",
        )

    assert adata.n_vars == 5


def test_estimator_preprocess_can_error_on_scanpy_hvg_failure(toy_adata):
    estimator = ClaspEstimator(n_components=4)

    with pytest.raises(Exception):
        estimator.preprocess(
            toy_adata,
            n_top_genes=5,
            min_gene_counts=0,
            normalize=False,
            hvg_flavor="not_a_real_flavor",
            hvg_fallback="error",
        )


def test_estimator_preprocess_can_create_artificial_batches():
    adata = AnnData(np.arange(24, dtype=float).reshape(6, 4))
    estimator = ClaspEstimator(n_components=2)

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
    estimator = ClaspEstimator(n_components=2)

    result = estimator.preprocess(adata, n_top_genes=None, min_gene_counts=0, normalize=False)

    assert "label" in result.obs
    assert list(result.obs["label"].astype(str)) == ["x", "x", "y", "y"]


def test_estimator_graph_and_embedding_methods(toy_adata):
    estimator = ClaspEstimator(n_components=6, embedding_method="spectral")
    adata = estimator.preprocess(toy_adata, n_top_genes=None)

    graph = estimator.data_to_graph(adata)
    coords = estimator.graph_to_embeddings(graph)

    assert sparse.isspmatrix_csr(graph)
    assert graph.shape == (adata.n_obs, adata.n_obs)
    assert coords.shape == (adata.n_obs, 2)


def test_estimator_defaults_to_imste_and_embeds_clasp_distance_graph(toy_adata):
    estimator = ClaspEstimator(n_components=6)
    estimator.embedder.set_params(n_msts=2, n_epochs=0)
    adata = estimator.preprocess(toy_adata, n_top_genes=None)
    graph = estimator.data_to_graph(
        adata,
        n_neighbors=2,
        intra_fraction=1.0,
        n_inter_edges=0,
    )

    assert isinstance(estimator.embedder, IteratedMinimumSpanningTreeEmbedder)
    from scipy.sparse.csgraph import connected_components

    assert connected_components(graph, directed=False, return_labels=False) > 1
    coords = estimator.graph_to_embeddings(graph)

    assert coords.shape == (adata.n_obs, 2)
    assert np.isfinite(coords).all()
    assert estimator.embedder.graph_input_ is True
    assert estimator.embedder.embedding_.shape[0] == adata.n_obs


def test_estimator_uses_supplied_embedder_without_overriding_its_configuration(toy_adata):
    configured = IteratedMinimumSpanningTreeEmbedder(
        n_msts=3, n_epochs=0, n_components=3, random_state=31
    )
    estimator = ClaspEstimator(embedder=configured, embedding_components=2, random_state=0)
    adata = estimator.preprocess(toy_adata, n_top_genes=None)
    graph = estimator.data_to_graph(adata, n_neighbors=4)

    coords = estimator.graph_to_embeddings(graph)

    assert estimator.embedder is configured
    assert configured.n_msts == 3
    assert configured.n_components == 3
    assert configured.random_state == 31
    assert coords.shape == (adata.n_obs, 3)
    np.testing.assert_array_equal(coords, configured.embedding_)


def test_estimator_accepts_legacy_graph_embedder_object(toy_adata):
    backend = ClaspGraphEmbedder(method="spectral", n_components=3, random_state=5)
    estimator = ClaspEstimator(embedder=backend, n_components=6)
    adata = estimator.preprocess(toy_adata, n_top_genes=None)
    graph = estimator.data_to_graph(adata)

    coords = estimator.graph_to_embeddings(graph)

    assert backend.embedding_.shape == (adata.n_obs, 3)
    assert coords.shape == (adata.n_obs, 3)


def test_estimator_rejects_both_embedder_and_legacy_method():
    with pytest.raises(ValueError, match="either embedder= or embedding_method="):
        ClaspEstimator(embedder=ClaspGraphEmbedder(), embedding_method="spectral")


def test_tuning_clones_and_preserves_configured_embedder(monkeypatch, toy_adata):
    import clasp.optimization as optimization
    import clasp.metrics

    configured = IteratedMinimumSpanningTreeEmbedder(
        n_msts=4, n_epochs=0, n_components=3, random_state=37
    )
    estimator = ClaspEstimator(embedder=configured)
    seen_embedders = []

    def fake_embed(self, adata, **kwargs):
        seen_embedders.append(self.embedder)
        adata.obsm["X_clasp"] = np.zeros((adata.n_obs, 3), dtype=float)
        return adata.obsm["X_clasp"]

    def fake_score(*args, **kwargs):
        return pd.DataFrame(
            [{"knn_label_agreement": 0.5, "batch_mixing": 0.5, "graph_density": 0.5}]
        )

    def fake_latent_bayesopt(objective_fn, search_space, **kwargs):
        score = objective_fn({})
        return {
            "best_params": {},
            "best_score": score,
            "history": pd.DataFrame([{"score": score}]),
            "latent_points": np.zeros((1, 1)),
            "observed_points": np.zeros((1, 1)),
            "observed_scores": np.array([score]),
            "models": {},
        }

    monkeypatch.setattr(ClaspEstimator, "embed", fake_embed)
    monkeypatch.setattr(clasp.metrics, "score_embedding", fake_score)
    monkeypatch.setattr(optimization, "latent_bayesopt", fake_latent_bayesopt)
    params = TuneParams(
        preprocess=PreprocessTuneParams(
            base={"n_top_genes": None, "min_gene_counts": 0, "normalize": False}
        ),
        estimator=EstimatorTuneParams(base={"n_components": 4}),
        graph=GraphTuneParams(base={"n_neighbors": 3}),
        pca_bo=LatentBOTuneParams(n_initial=1, n_iterations=0, embedding_model="pca"),
        run_gplvm_refinement=False,
    )

    estimator.tune(toy_adata, params)

    assert len(seen_embedders) == 1
    assert seen_embedders[0] is not configured
    assert seen_embedders[0].get_params() == configured.get_params()


def test_estimator_accepts_balanced_preset_by_name():
    estimator = ClaspEstimator(preset="balanced")

    assert "balanced" in available_presets()
    assert estimator.n_components == 80
    assert estimator.n_neighbors == 24
    assert estimator.n_inter_edges == 3
    assert estimator.assignment_quantile == 0.32
    assert estimator.hubness_k == 12
    assert estimator.edge_weighting == "distance"
    assert estimator.inter_edge_mode == "assignment"
    assert estimator.mutual_neighbors is True
    assert estimator.preprocess_defaults["n_top_genes"] == 1300


def test_estimator_accepts_trajectory_preset_by_name():
    estimator = ClaspEstimator(preset="trajectory")

    assert estimator.n_components == 90
    assert estimator.n_neighbors == 28
    assert estimator.intra_fraction == 0.58
    assert estimator.n_inter_edges == 4
    assert estimator.assignment_quantile == 0.30
    assert estimator.hubness_k == 13
    assert estimator.inter_edge_mode == "propagate_neighbors"
    assert estimator.mutual_neighbors is True


def test_estimator_preset_preserves_explicit_constructor_overrides():
    estimator = ClaspEstimator(preset="balanced", n_neighbors=10, inter_edge_mode="propagate_neighbors")

    assert estimator.n_components == 80
    assert estimator.n_neighbors == 10
    assert estimator.inter_edge_mode == "propagate_neighbors"


def test_estimator_rejects_unknown_preset():
    with pytest.raises(ValueError, match="Unknown CLASP preset"):
        ClaspEstimator(preset="not-a-preset")


def test_estimator_graph_to_embeddings_accepts_call_overrides(toy_adata):
    estimator = ClaspEstimator(n_components=6, embedding_method="spectral")
    adata = estimator.preprocess(toy_adata, n_top_genes=None)
    graph = estimator.data_to_graph(adata)

    coords = estimator.graph_to_embeddings(graph, method="spectral", n_components=3, random_state=5)

    assert coords.shape == (adata.n_obs, 3)


def test_estimator_data_to_graph_accepts_call_overrides(toy_adata):
    estimator = ClaspEstimator(n_components=6, n_neighbors=6, embedding_method="spectral")
    adata = estimator.preprocess(toy_adata, n_top_genes=None)

    graph = estimator.data_to_graph(
        adata,
        n_neighbors=4,
        intra_fraction=0.25,
        n_inter_edges=2,
        assignment_quantile=1.0,
        hubness_correction="none",
        hubness_k=3,
        rank_correction=False,
        edge_weighting="binary",
        inter_edge_mode="assignment",
        mutual_neighbors=False,
        neighbor_mode="distance",
        symmetrize=False,
    )

    params = adata.uns["clasp"]["graph"]["parameters"]
    assert graph.shape == (adata.n_obs, adata.n_obs)
    assert params["n_neighbors"] == 4
    assert params["intra_fraction"] == 0.25
    assert params["n_inter_edges"] == 2
    assert params["assignment_quantile"] == 1.0
    assert params["hubness_correction"] == "none"
    assert params["hubness_k"] == 3
    assert params["rank_correction"] is False
    assert params["edge_weighting"] == "binary"
    assert params["inter_edge_mode"] == "assignment"
    assert params["mutual_neighbors"] is False
    assert params["neighbor_mode"] == "distance"
    assert params["symmetrize"] is False


def test_estimator_embed_combines_graph_and_vector(toy_adata):
    estimator = ClaspEstimator(n_components=6, embedding_method="spectral")
    adata = estimator.preprocess(toy_adata, n_top_genes=None)

    coords = estimator.embed(adata)

    assert coords.shape == (adata.n_obs, 2)


def test_estimator_embed_accepts_graph_overrides(toy_adata):
    estimator = ClaspEstimator(n_components=6, embedding_method="spectral")
    adata = estimator.preprocess(toy_adata, n_top_genes=None)

    coords = estimator.embed(
        adata,
        n_neighbors=4,
        intra_fraction=0.25,
        assignment_quantile=1.0,
        hubness_correction="none",
        hubness_k=3,
        rank_correction=False,
        edge_weighting="binary",
        inter_edge_mode="assignment",
        mutual_neighbors=False,
        neighbor_mode="distance",
        embedding_method="spectral",
        embedding_components=3,
    )

    assert coords.shape == (adata.n_obs, 3)
    assert adata.uns["clasp"]["graph"]["parameters"]["n_neighbors"] == 4
    assert adata.uns["clasp"]["graph"]["parameters"]["hubness_correction"] == "none"
    assert adata.uns["clasp"]["graph"]["parameters"]["rank_correction"] is False
    assert adata.uns["clasp"]["graph"]["parameters"]["edge_weighting"] == "binary"
    assert adata.uns["clasp"]["graph"]["parameters"]["inter_edge_mode"] == "assignment"
    assert adata.uns["clasp"]["graph"]["parameters"]["mutual_neighbors"] is False
    assert adata.uns["clasp"]["graph"]["parameters"]["neighbor_mode"] == "distance"


def test_estimator_plot_wraps_embedding_pair(toy_adata):
    estimator = ClaspEstimator()
    toy_adata.obsm["X_clasp"] = toy_adata.X[:, :2]

    axes = estimator.plot(toy_adata)

    assert len(axes) == 2
    assert axes[0].get_title() == "X_clasp by batch"
    assert axes[1].get_title() == "X_clasp by label"
    plt.close(axes[0].figure)


def test_estimator_tune_returns_and_saves_parameters(monkeypatch, tmp_path, toy_adata):
    import clasp.optimization as optimization

    def fake_latent_bayesopt(objective_fn, search_space, **kwargs):
        params = {
            "n_top_genes": 8,
            "n_components": 4,
            "n_neighbors": 4,
        }
        return {
            "best_params": params,
            "best_score": 0.7,
            "history": pd.DataFrame([{**params, "score": 0.7}]),
            "latent_points": np.zeros((1, 1)),
            "observed_points": np.zeros((1, 1)),
            "observed_scores": np.array([0.7]),
            "models": {},
        }

    monkeypatch.setattr(optimization, "latent_bayesopt", fake_latent_bayesopt)
    estimator = ClaspEstimator(n_components=4, embedding_method="spectral")
    params = TuneParams(
        preprocess=PreprocessTuneParams(
            base={"n_top_genes": 10, "min_gene_counts": 0, "normalize": False},
            fixed={"max_cells": 12},
            search_space={"n_top_genes": {"type": "int", "bounds": [5, 12]}},
        ),
        estimator=EstimatorTuneParams(
            base={"n_components": 3},
            search_space={"n_components": {"type": "int", "bounds": [2, 5]}},
        ),
        graph=GraphTuneParams(
            base={"n_neighbors": 3, "assignment_quantile": 1.0},
            search_space={"n_neighbors": {"type": "int", "bounds": [2, 5]}},
        ),
        pca_bo=LatentBOTuneParams(n_initial=4, n_iterations=1, embedding_model="pca"),
        embedding_method="spectral",
        embedding_epochs=None,
    )

    result = estimator.tune(toy_adata, params)
    path = result.save(tmp_path / "params.json")

    assert result.preprocess_params["n_top_genes"] == 8
    assert result.preprocess_params["max_cells"] == 12
    assert result.estimator_params["n_components"] == 4
    assert result.graph_params["n_neighbors"] == 4
    assert path.exists()

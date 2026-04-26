from __future__ import annotations

import builtins
import warnings

import numpy as np
import pytest
from scipy import sparse

from clasp.embedding import embed_graph
from clasp.graph import build_clasp_graph
from clasp.preprocessing import ensure_pca


def test_embedding_returns_two_dimensions(toy_adata):
    ensure_pca(toy_adata, n_components=6)
    graph = build_clasp_graph(toy_adata, n_neighbors=6)
    coords = embed_graph(graph, method="spectral", n_components=2)
    assert coords.shape == (toy_adata.n_obs, 2)


def test_auto_embedding_falls_back_when_umap_unavailable(monkeypatch, toy_adata):
    ensure_pca(toy_adata, n_components=6)
    graph = build_clasp_graph(toy_adata, n_neighbors=6)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "umap":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    coords = embed_graph(graph, method="auto", n_components=2)
    assert coords.shape == (toy_adata.n_obs, 2)


def test_umap_method_errors_when_umap_unavailable(monkeypatch, toy_adata):
    ensure_pca(toy_adata, n_components=6)
    graph = build_clasp_graph(toy_adata, n_neighbors=6)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "umap":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="umap-learn"):
        embed_graph(graph, method="umap", n_components=2)


def test_umap_embedding_suppresses_expected_noise(capsys, toy_adata):
    pytest.importorskip("umap")
    ensure_pca(toy_adata, n_components=6)
    graph = build_clasp_graph(toy_adata, n_neighbors=6)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        coords = embed_graph(graph, method="umap", n_components=2, n_epochs=10)

    captured = capsys.readouterr()
    messages = "\n".join(str(item.message) for item in caught)
    assert coords.shape == (toy_adata.n_obs, 2)
    assert "using precomputed metric" not in messages
    assert "n_jobs value" not in messages
    assert "omp_set_nested" not in captured.err


def test_umap_embedding_handles_zero_distance_edges():
    pytest.importorskip("umap")
    graph = sparse.csr_matrix(
        np.array(
            [
                [0.0, 1.0, 0.5, 0.0, 0.0],
                [1.0, 0.0, 0.5, 0.0, 0.0],
                [0.5, 0.5, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.5, 0.0, 1.0],
                [0.0, 0.0, 0.5, 1.0, 0.0],
            ]
        )
    )

    coords = embed_graph(graph, method="umap", n_components=2, n_epochs=10, n_neighbors=2)

    assert coords.shape == (5, 2)
    assert np.isfinite(coords).all()

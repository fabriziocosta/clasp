from __future__ import annotations

import builtins

import pytest

from scalp_lite.embedding import embed_graph
from scalp_lite.graph import build_scalp_graph
from scalp_lite.preprocessing import ensure_pca


def test_embedding_returns_two_dimensions(toy_adata):
    ensure_pca(toy_adata, n_components=6)
    graph = build_scalp_graph(toy_adata, n_neighbors=6)
    coords = embed_graph(graph, method="spectral", n_components=2)
    assert coords.shape == (toy_adata.n_obs, 2)


def test_auto_embedding_falls_back_when_umap_unavailable(monkeypatch, toy_adata):
    ensure_pca(toy_adata, n_components=6)
    graph = build_scalp_graph(toy_adata, n_neighbors=6)
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
    graph = build_scalp_graph(toy_adata, n_neighbors=6)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "umap":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="umap-learn"):
        embed_graph(graph, method="umap", n_components=2)

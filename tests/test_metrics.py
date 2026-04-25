from __future__ import annotations

from scalp_lite.embedding import embed_graph
from scalp_lite.graph import build_scalp_graph
from scalp_lite.metrics import score_embedding
from scalp_lite.preprocessing import ensure_pca


def test_metrics_return_one_row_dataframe_with_expected_columns(toy_adata):
    ensure_pca(toy_adata, n_components=6)
    graph = build_scalp_graph(toy_adata, n_neighbors=6)
    toy_adata.obsm["X_scalp"] = embed_graph(graph, method="spectral")
    scores = score_embedding(toy_adata, graph=graph)

    assert len(scores) == 1
    assert {
        "embedding_key",
        "n_obs",
        "n_dims",
        "batch_mixing",
        "batch_silhouette",
        "knn_label_agreement",
        "label_silhouette",
        "graph_density",
        "runtime_seconds",
    }.issubset(scores.columns)

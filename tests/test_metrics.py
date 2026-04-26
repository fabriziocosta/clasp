from __future__ import annotations

from clasp.embedding import embed_graph
from clasp.graph import build_clasp_graph
from clasp.metrics import score_embedding
from clasp.preprocessing import ensure_pca


def test_metrics_return_one_row_dataframe_with_expected_columns(toy_adata):
    ensure_pca(toy_adata, n_components=6)
    graph = build_clasp_graph(toy_adata, n_neighbors=6)
    toy_adata.obsm["X_clasp"] = embed_graph(graph, method="spectral")
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

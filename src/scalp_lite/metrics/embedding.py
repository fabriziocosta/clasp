from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder


def _encode(values) -> np.ndarray:
    return LabelEncoder().fit_transform(np.asarray(values).astype(str))


def knn_label_agreement(X: np.ndarray, labels, *, n_neighbors: int = 5) -> float:
    """Fraction of nearest-neighbor labels matching each point's label."""
    labels = _encode(labels)
    if len(labels) < 2:
        return np.nan
    k = min(n_neighbors + 1, len(labels))
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    _, indices = nn.kneighbors(X)
    neighbor_labels = labels[indices[:, 1:]]
    return float((neighbor_labels == labels[:, None]).mean())


def batch_mixing(X: np.ndarray, batches, *, n_neighbors: int = 5) -> float:
    """Fraction of nearest neighbors from a different batch."""
    batches = _encode(batches)
    if len(batches) < 2:
        return np.nan
    k = min(n_neighbors + 1, len(batches))
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    _, indices = nn.kneighbors(X)
    neighbor_batches = batches[indices[:, 1:]]
    return float((neighbor_batches != batches[:, None]).mean())


def _safe_silhouette(X: np.ndarray, labels) -> float:
    encoded = _encode(labels)
    if len(np.unique(encoded)) < 2 or len(encoded) <= len(np.unique(encoded)):
        return np.nan
    return float(silhouette_score(X, encoded))


def score_embedding(
    adata: ad.AnnData,
    *,
    embedding_key: str = "X_scalp",
    batch_key: str = "batch",
    label_key: str | None = "label",
    graph: sparse.spmatrix | None = None,
    n_neighbors: int = 5,
) -> pd.DataFrame:
    """Return one-row dataframe with SCALP-lite embedding metrics."""
    if embedding_key not in adata.obsm:
        raise KeyError(f"Missing embedding in obsm: {embedding_key!r}")
    if batch_key not in adata.obs:
        raise KeyError(f"Missing batch obs column: {batch_key!r}")

    X = np.asarray(adata.obsm[embedding_key])
    row: dict[str, float | str | int | None] = {
        "embedding_key": embedding_key,
        "n_obs": int(adata.n_obs),
        "n_dims": int(X.shape[1]),
        "batch_mixing": batch_mixing(X, adata.obs[batch_key], n_neighbors=n_neighbors),
        "batch_silhouette": _safe_silhouette(X, adata.obs[batch_key]),
    }

    if label_key is not None and label_key in adata.obs:
        row["knn_label_agreement"] = knn_label_agreement(X, adata.obs[label_key], n_neighbors=n_neighbors)
        row["label_silhouette"] = _safe_silhouette(X, adata.obs[label_key])
    else:
        row["knn_label_agreement"] = np.nan
        row["label_silhouette"] = np.nan

    if graph is not None:
        graph = sparse.csr_matrix(graph)
        row["graph_density"] = float(graph.nnz / (graph.shape[0] * graph.shape[1]))
    else:
        row["graph_density"] = np.nan

    metadata = adata.uns.get("scalp_lite", {}).get("graph", {})
    row["runtime_seconds"] = metadata.get("runtime_seconds", np.nan)
    return pd.DataFrame([row])

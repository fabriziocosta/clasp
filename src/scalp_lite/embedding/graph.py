from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.manifold import SpectralEmbedding


def _umap_available():
    try:
        import umap  # noqa: F401
    except ImportError:
        return False
    return True


def embed_graph(
    graph: sparse.spmatrix,
    *,
    method: str = "auto",
    n_components: int = 2,
    random_state: int = 0,
    **kwargs,
) -> np.ndarray:
    """Embed a graph into low dimensions using UMAP when available, else spectral embedding."""
    if graph.shape[0] != graph.shape[1]:
        raise ValueError("graph must be square.")
    if n_components < 1:
        raise ValueError("n_components must be >= 1.")

    graph = sparse.csr_matrix(graph)
    method = method.lower()
    if method == "auto":
        method = "umap" if _umap_available() else "spectral"

    if method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise ImportError("UMAP embedding requires optional dependency `umap-learn`.") from exc
        distances = graph.copy().astype(float)
        distances.data = 1.0 / np.maximum(distances.data, 1e-12) - 1.0
        dense = distances.toarray()
        max_dist = np.nanmax(dense[np.isfinite(dense)]) if dense.size else 1.0
        dense[dense == 0] = max_dist * 2
        np.fill_diagonal(dense, 0)
        return umap.UMAP(
            n_components=n_components,
            metric="precomputed",
            random_state=random_state,
            **kwargs,
        ).fit_transform(dense)

    if method == "spectral":
        return SpectralEmbedding(
            n_components=n_components,
            affinity="precomputed",
            random_state=random_state,
            **kwargs,
        ).fit_transform(graph)

    raise ValueError("method must be one of: 'auto', 'umap', 'spectral'.")

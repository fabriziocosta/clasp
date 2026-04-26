from __future__ import annotations

import numpy as np


def _mean_k_smallest(values: np.ndarray, *, k: int, axis: int) -> np.ndarray:
    if k < 1:
        raise ValueError("k must be >= 1.")
    if values.shape[axis] == 0:
        return np.zeros(values.shape[1 - axis], dtype=float)
    k = min(k, values.shape[axis])
    partitioned = np.partition(values, kth=k - 1, axis=axis)
    return np.take(partitioned, np.arange(k), axis=axis).mean(axis=axis)


def csls_distances(distances: np.ndarray, *, k: int = 10) -> np.ndarray:
    """Return cross-domain similarity local scaling distances.

    CSLS is usually written for similarities as
    ``2 * s(x, y) - r_x - r_y``. For distances, which are minimized here,
    the equivalent corrected distance is ``2 * d(x, y) - r_x - r_y``,
    where ``r`` is the mean distance to the local cross-domain neighborhood.
    """
    distances = np.asarray(distances, dtype=float)
    if distances.ndim != 2:
        raise ValueError("distances must be a 2D matrix.")
    if distances.size == 0:
        return distances.copy()

    finite = np.where(np.isfinite(distances), distances, np.inf)
    row_scale = _mean_k_smallest(finite, k=k, axis=1)
    col_scale = _mean_k_smallest(finite, k=k, axis=0)
    corrected = 2.0 * distances - row_scale[:, None] - col_scale[None, :]
    return corrected


def rank_distances(distances: np.ndarray) -> np.ndarray:
    """Return reciprocal rank distances for a 2D distance matrix.

    Lower values remain better. Each entry is the sum of its row-wise rank and
    column-wise rank, so the transform works for both square within-batch
    matrices and rectangular cross-batch matrices.
    """
    distances = np.asarray(distances, dtype=float)
    if distances.ndim != 2:
        raise ValueError("distances must be a 2D matrix.")
    if distances.size == 0:
        return distances.copy()

    finite = np.isfinite(distances)
    row_order = np.argsort(distances, axis=1, kind="mergesort")
    row_ranks = np.empty_like(row_order, dtype=float)
    row_ids = np.arange(distances.shape[0])[:, None]
    row_ranks[row_ids, row_order] = np.arange(distances.shape[1], dtype=float)

    col_order = np.argsort(distances, axis=0, kind="mergesort")
    col_ranks = np.empty_like(col_order, dtype=float)
    col_ids = np.arange(distances.shape[1])[None, :]
    col_ranks[col_order, col_ids] = np.arange(distances.shape[0], dtype=float)[:, None]

    ranks = row_ranks + col_ranks
    ranks[~finite] = np.inf
    return ranks


def correct_distances(distances: np.ndarray, *, hubness_correction: str = "csls", hubness_k: int = 10, rank_correction: bool = True) -> np.ndarray:
    """Apply configured distance corrections before graph edge selection."""
    if hubness_correction == "csls":
        distances = csls_distances(distances, k=hubness_k)
    elif hubness_correction != "none":
        raise ValueError("hubness_correction must be one of: 'none', 'csls'.")
    if rank_correction:
        distances = rank_distances(distances)
    return distances


def shifted_distance_weights(distances: np.ndarray) -> np.ndarray:
    """Convert possibly negative corrected distances to positive edge weights."""
    distances = np.asarray(distances, dtype=float)
    if distances.size == 0:
        return distances.astype(np.float32)
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return np.zeros_like(distances, dtype=np.float32)
    shifted = distances - min(float(finite.min()), 0.0)
    return (1.0 / (1.0 + np.maximum(shifted, 0.0))).astype(np.float32)


def edge_weights(distances: np.ndarray, *, edge_weighting: str = "distance") -> np.ndarray:
    """Return graph edge weights from corrected distances."""
    if edge_weighting == "distance":
        return shifted_distance_weights(distances)
    if edge_weighting == "binary":
        return np.ones_like(np.asarray(distances), dtype=np.float32)
    raise ValueError("edge_weighting must be one of: 'distance', 'binary'.")

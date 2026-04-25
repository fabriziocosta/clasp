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

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances

from scalp_lite.graph.hubness import correct_distances, edge_weights
from scalp_lite.graph.params import _coerce_int


def _iterated_assignment(distances: np.ndarray, n_repeats: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    working = distances.copy()
    all_rows: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    all_distances: list[np.ndarray] = []

    for _ in range(max(0, n_repeats)):
        if not np.isfinite(working).any():
            break
        rows, cols = linear_sum_assignment(working)
        vals = working[rows, cols]
        keep = np.isfinite(vals)
        rows, cols, vals = rows[keep], cols[keep], vals[keep]
        if len(rows) == 0:
            break
        all_rows.append(rows)
        all_cols.append(cols)
        all_distances.append(vals)
        working[rows, cols] = np.inf

    if not all_rows:
        empty = np.array([], dtype=int)
        return empty, empty, np.array([], dtype=float)
    return np.concatenate(all_rows), np.concatenate(all_cols), np.concatenate(all_distances)


def build_inter_batch_graph(
    X_left: np.ndarray,
    X_right: np.ndarray,
    *,
    n_inter_edges: int = 1,
    metric: str = "euclidean",
    assignment_quantile: float | None = 0.95,
    hubness_correction: str = "csls",
    hubness_k: int = 10,
    rank_correction: bool = True,
    edge_weighting: str = "distance",
) -> sparse.csr_matrix:
    """Build a sparse cross-batch graph using repeated Hungarian assignment."""
    n_inter_edges = _coerce_int(n_inter_edges, name="n_inter_edges", minimum=0)
    hubness_k = _coerce_int(hubness_k, name="hubness_k", minimum=1)
    shape = (X_left.shape[0], X_right.shape[0])
    if shape[0] == 0 or shape[1] == 0 or n_inter_edges <= 0:
        return sparse.csr_matrix(shape, dtype=np.float32)

    distances = correct_distances(
        pairwise_distances(X_left, X_right, metric=metric),
        hubness_correction=hubness_correction,
        hubness_k=hubness_k,
        rank_correction=rank_correction,
    )

    rows, cols, vals = _iterated_assignment(distances, n_inter_edges)

    if assignment_quantile is not None and len(vals) > 0:
        if not 0 < assignment_quantile <= 1:
            raise ValueError("assignment_quantile must be in (0, 1] or None.")
        cutoff = np.quantile(vals, assignment_quantile)
        keep = vals <= cutoff
        rows, cols, vals = rows[keep], cols[keep], vals[keep]

    weights = edge_weights(vals, edge_weighting=edge_weighting)
    return sparse.csr_matrix((weights, (rows, cols)), shape=shape)

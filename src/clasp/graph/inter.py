from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances

from clasp.graph.hubness import correct_distances, edge_weights
from clasp.graph.params import _coerce_int


def _finite_assignment_costs(distances: np.ndarray) -> np.ndarray:
    """Return finite costs for Hungarian assignment while preserving invalid edges.

    `linear_sum_assignment` raises when the matrix contains infeasible rows or
    columns. We replace non-finite entries with a large finite penalty for the
    optimizer, then filter those penalty assignments against the original
    distance matrix after solving.
    """
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return distances.copy()
    lo = float(finite.min())
    hi = float(finite.max())
    penalty = hi + max(hi - lo, 1.0) * (max(distances.shape) + 1)
    return np.where(np.isfinite(distances), distances, penalty)


def _iterated_assignment(distances: np.ndarray, n_repeats: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    working = distances.copy()
    all_rows: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    all_distances: list[np.ndarray] = []

    for _ in range(max(0, n_repeats)):
        if not np.isfinite(working).any():
            break
        rows, cols = linear_sum_assignment(_finite_assignment_costs(working))
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


def _edge_matrix(
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    *,
    shape: tuple[int, int],
    edge_weighting: str,
) -> sparse.csr_matrix:
    if len(rows) == 0:
        return sparse.csr_matrix(shape, dtype=np.float32)

    weights = edge_weights(vals, edge_weighting=edge_weighting)
    keys = rows.astype(np.int64) * shape[1] + cols.astype(np.int64)
    order = np.argsort(keys, kind="mergesort")
    keys = keys[order]
    rows = rows[order]
    cols = cols[order]
    weights = weights[order]
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
    rows = rows[starts]
    cols = cols[starts]
    weights = np.maximum.reduceat(weights, starts).astype(np.float32)
    return sparse.csr_matrix((weights, (rows, cols)), shape=shape)


def _propagate_assigned_neighbors(
    assigned_rows: np.ndarray,
    assigned_cols: np.ndarray,
    neighbor_distances: np.ndarray,
    *,
    n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(assigned_rows) == 0 or neighbor_distances.shape[1] <= 1 or n_neighbors <= 0:
        empty = np.array([], dtype=int)
        return empty, empty, np.array([], dtype=float)

    n_edges = min(n_neighbors, neighbor_distances.shape[1] - 1)
    candidates = neighbor_distances[assigned_cols]
    indices = np.argpartition(candidates, kth=n_edges - 1, axis=1)[:, :n_edges]
    selected_distances = np.take_along_axis(candidates, indices, axis=1)
    order = np.argsort(selected_distances, axis=1)
    indices = np.take_along_axis(indices, order, axis=1)
    selected_distances = np.take_along_axis(selected_distances, order, axis=1)

    rows = np.repeat(assigned_rows, n_edges)
    cols = indices.reshape(-1)
    vals = selected_distances.reshape(-1)
    keep = np.isfinite(vals)
    return rows[keep], cols[keep], vals[keep]


def build_inter_batch_graph(
    X_left: np.ndarray,
    X_right: np.ndarray,
    *,
    n_neighbors: int = 5,
    n_inter_edges: int = 1,
    metric: str = "euclidean",
    assignment_quantile: float | None = 0.95,
    hubness_correction: str = "csls",
    hubness_k: int = 10,
    rank_correction: bool = True,
    edge_weighting: str = "distance",
    inter_edge_mode: str = "propagate_neighbors",
) -> sparse.csr_matrix:
    """Build a sparse directed cross-batch graph using Hungarian assignment.

    In the default ``propagate_neighbors`` mode, each retained assigned cell in
    ``X_left`` inherits nearest-neighbor edges from its assigned partner in
    ``X_right``. The assigned pair itself is not linked directly. The
    ``assignment`` mode links each retained assigned pair directly.
    """
    n_neighbors = _coerce_int(n_neighbors, name="n_neighbors", minimum=0)
    n_inter_edges = _coerce_int(n_inter_edges, name="n_inter_edges", minimum=0)
    hubness_k = _coerce_int(hubness_k, name="hubness_k", minimum=1)
    if inter_edge_mode not in {"propagate_neighbors", "assignment"}:
        raise ValueError("inter_edge_mode must be one of: 'propagate_neighbors', 'assignment'.")
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

    if inter_edge_mode == "assignment":
        return _edge_matrix(rows, cols, vals, shape=shape, edge_weighting=edge_weighting)

    right_distances = pairwise_distances(X_right, metric=metric)
    np.fill_diagonal(right_distances, np.inf)
    right_distances = correct_distances(
        right_distances,
        hubness_correction=hubness_correction,
        hubness_k=min(hubness_k, max(X_right.shape[0] - 1, 1)),
        rank_correction=rank_correction,
    )
    rows, cols, vals = _propagate_assigned_neighbors(
        rows,
        cols,
        right_distances,
        n_neighbors=n_neighbors,
    )
    return _edge_matrix(rows, cols, vals, shape=shape, edge_weighting=edge_weighting)

"""Helpers for the native single-cell IMSTE notebook."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath
from scipy import sparse
from scipy.spatial import ConvexHull, QhullError
from sklearn.manifold import trustworthiness
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


DATA_FILE = "human_pancreas_norm_complexBatch.h5ad"
DEFAULT_EMBEDDER_OPTIONS = {
    "n_msts": 64,
    "n_components": 2,
    "n_epochs": 1500,
    "batch_size": 131_072,
    "learning_rate": 0.05,
    "negative_ratio": 30,
    "lambda_rep": 0.5,
    "random_state": 42,
    "epsilon": 1e-4,
    "logistic_margin": 1.0,
    "logistic_temperature": 0.5,
    "mst_method": "famst",
    "mst_neighbors": 32,
    "mst_inter_component_edges": 5,
    "mst_max_neighbors": 256,
    "progress_callback": None,
    "mst_progress_callback": None,
    "mst_representatives_per_component": 10,
}


def _repository_root() -> Path:
    """Find the repository containing this notebook helper."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "data" / DATA_FILE).is_file():
            return parent
    raise FileNotFoundError(
        f"Could not find data/{DATA_FILE} above {Path(__file__).resolve()}"
    )



def available_datasets() -> list[Path]:
    """Return all top-level H5AD datasets available in the repository data dir."""
    data_dir = _repository_root() / "data"
    excluded_terms = ("pca", "scalp", "clasp")
    datasets = sorted(
        path
        for path in data_dir.glob("*.h5ad")
        if not any(term in path.stem.lower() for term in excluded_terms)
    )
    if not datasets:
        raise FileNotFoundError(f"No original high-dimensional H5AD datasets found in {data_dir}")
    return datasets


def available_obs_columns(path: str | Path) -> list[str]:
    """List metadata columns in an H5AD file without loading their values."""
    with h5py.File(path, "r") as h5:
        if "obs" not in h5 or not isinstance(h5["obs"], h5py.Group):
            return []
        columns = []
        for key, node in h5["obs"].items():
            if key.startswith("__"):
                continue
            if isinstance(node, h5py.Dataset) or (
                isinstance(node, h5py.Group)
                and {"codes", "categories"}.issubset(node.keys())
            ):
                columns.append(key)
        return columns


def default_label_key(obs: pd.DataFrame) -> str:
    """Choose a likely categorical label column for classification and plots."""
    preferred = (
        "celltype", "cell_type", "final_annotation", "clusters_coarse",
        "clusters_fine", "clusters", "cluster", "leiden", "louvain",
        "Stage", "major_cell_sets", "cell_sets",
    )
    for key in preferred:
        if key in obs and obs[key].nunique(dropna=True) > 1:
            return key

    for key in obs.columns:
        n_values = obs[key].nunique(dropna=True)
        if 1 < n_values <= min(100, max(2, len(obs) // 10)):
            return key
    raise ValueError(
        "No categorical-looking label column was found. Set LABEL_KEY manually "
        "to a suitable column in obs."
    )

def _decode_h5_strings(values: np.ndarray) -> np.ndarray:
    """Decode HDF5 byte strings while leaving numeric arrays unchanged."""
    values = np.asarray(values)
    if values.dtype.kind in {"S", "O", "U"}:
        return np.asarray(
            [value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else value
             for value in values],
            dtype=object,
        )
    return values


def _read_h5ad_matrix(node: h5py.Dataset | h5py.Group, row_indices: np.ndarray) -> np.ndarray:
    """Read selected rows from a dense H5AD matrix or CSR/CSC matrix group."""
    if isinstance(node, h5py.Dataset):
        return np.asarray(node[row_indices])

    encoding = node.attrs.get("encoding-type", "")
    if isinstance(encoding, bytes):
        encoding = encoding.decode("utf-8")
    shape = tuple(int(value) for value in node.attrs["shape"])
    if encoding == "csr_matrix":
        matrix = sparse.csr_matrix(
            (node["data"][:], node["indices"][:], node["indptr"][:]),
            shape=shape,
        )
    elif encoding == "csc_matrix":
        matrix = sparse.csc_matrix(
            (node["data"][:], node["indices"][:], node["indptr"][:]),
            shape=shape,
        )
    else:
        raise ValueError(f"Unsupported H5AD matrix encoding: {encoding!r}")
    return matrix[row_indices].toarray()


def _read_h5ad_obs_column(node: h5py.Dataset | h5py.Group, row_indices: np.ndarray):
    """Read a regular or categorical AnnData obs column."""
    if isinstance(node, h5py.Group) and {"codes", "categories"}.issubset(node.keys()):
        categories = _decode_h5_strings(node["categories"][:])
        codes = node["codes"][row_indices].astype(np.int64, copy=False)
        values = np.full(len(row_indices), None, dtype=object)
        valid = codes >= 0
        values[valid] = categories[codes[valid]]
        return pd.Categorical(values, categories=categories)
    return _decode_h5_strings(node[row_indices])


def load_dataset(
    path: str | Path | None = None,
    *,
    matrix_key: str = "obsm/X_pca",
    obs_keys: tuple[str, ...] = ("tech", "celltype"),
    max_cells: int | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Load one H5AD representation as float32 NumPy plus selected metadata.

    ``matrix_key`` may be ``X``, ``layers/<name>``, or ``obsm/<name>``.
    ``max_cells`` selects a reproducible subset. The H5AD file is opened
    read-only and is never modified.
    """
    path = Path(path) if path is not None else _repository_root() / "data" / DATA_FILE
    with h5py.File(path, "r") as h5:
        if matrix_key not in {"X", "layers", "obsm"} and not matrix_key.startswith(
            ("layers/", "obsm/")
        ):
            raise ValueError("matrix_key must be X, layers/<name>, or obsm/<name>.")
        matrix_node = h5
        for part in matrix_key.split("/"):
            matrix_node = matrix_node[part]
        n_obs = int(
            matrix_node.shape[0]
            if isinstance(matrix_node, h5py.Dataset)
            else matrix_node.attrs["shape"][0]
        )
        if max_cells is None or max_cells >= n_obs:
            row_indices = np.arange(n_obs, dtype=np.int64)
        else:
            if max_cells < 2:
                raise ValueError("max_cells must be at least 2 or None.")
            row_indices = np.sort(
                np.random.default_rng(random_state).choice(
                    n_obs, size=max_cells, replace=False
                )
            )
        X = _read_h5ad_matrix(matrix_node, row_indices)

        obs_group = h5["obs"]
        obs_columns = {
            key: _read_h5ad_obs_column(obs_group[key], row_indices)
            for key in obs_keys
            if key in obs_group
        }
        obs_index_key = obs_group.attrs.get("_index", "_index")
        if isinstance(obs_index_key, bytes):
            obs_index_key = obs_index_key.decode("utf-8")
        if obs_index_key in obs_group:
            obs_names = _decode_h5_strings(obs_group[obs_index_key][row_indices]).astype(str)
        else:
            obs_names = row_indices.astype(str)

    X = np.asarray(X, dtype=np.float32, order="C")
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("Loaded representation must be a finite 2D matrix.")
    return X, pd.DataFrame(obs_columns, index=obs_names), row_indices


def load_embedder_class():
    """Import IMSTE from the environment or an optional source checkout."""
    try:
        from imste import IteratedMinimumSpanningTreeEmbedder
    except ImportError:
        imste_root = Path(
            os.environ.get("IMSTE_REPO", Path.home() / "Code" / "mst_embedding")
        )
        if not (imste_root / "imste").is_dir():
            raise ImportError(
                "Install imste or set IMSTE_REPO to the mst_embedding source directory."
            )
        sys.path.insert(0, str(imste_root))
        from imste import IteratedMinimumSpanningTreeEmbedder
    return IteratedMinimumSpanningTreeEmbedder


def default_embedder_params() -> dict:
    """Return notebook defaults, selecting Apple MPS when it is available."""
    try:
        import torch

        device = "mps" if torch.backends.mps.is_available() else "auto"
    except ImportError:
        device = "auto"
    return {**DEFAULT_EMBEDDER_OPTIONS, "device": device}


def fit_embedding(X: np.ndarray, params: dict | None = None):
    """Fit IMSTE and report its runtime and trustworthiness."""
    embedder_class = load_embedder_class()
    params = default_embedder_params() if params is None else dict(params)
    estimator = embedder_class(**params)
    embedding = estimator.fit_transform(X)
    score = trustworthiness(X, embedding, n_neighbors=10)
    print(f"IMSTE source: {embedder_class.__module__}")
    print(f"Device: {estimator.device_}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Graph construction: {estimator.graph_construction_time_:.2f} s")
    print(f"Optimization: {estimator.embedding_optimization_time_:.2f} s")
    print(f"Total fit: {estimator.fit_time_:.2f} s")
    print(f"Trustworthiness (10 neighbors): {score:.4f}")
    return estimator, embedding


def plot_embedding(
    embedding: np.ndarray,
    obs: pd.DataFrame,
    *,
    color_keys: tuple[str, ...] = ("tech", "celltype"),
    point_size: float = 7,
):
    """Plot the embedding colored by available metadata columns."""
    color_keys = [key for key in color_keys if key in obs]
    if not color_keys:
        raise ValueError("None of the requested color columns are present in obs.")
    fig, axes = plt.subplots(
        1,
        len(color_keys),
        figsize=(1.3 * point_size * len(color_keys), point_size),
        squeeze=False,
    )
    for ax, key in zip(axes.ravel(), color_keys):
        labels = pd.Categorical(obs[key])
        points = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=labels.codes,
            cmap="tab20",
            s=point_size,
            alpha=0.8,
            linewidths=0,
        )
        ax.set_title(f"IMSTE colored by {key}")
        ax.set_xlabel("IMSTE 1")
        ax.set_ylabel("IMSTE 2")
        handles, _ = points.legend_elements(num=len(labels.categories))
        ax.legend(
            handles,
            [str(value) for value in labels.categories],
            title=key,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=False,
        )
    fig.tight_layout()
    return fig, axes



def cross_validate_embedding_knn(
    embedding: np.ndarray,
    obs: pd.DataFrame,
    *,
    label_key: str = "celltype",
    n_neighbors: int = 5,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Cross-validate a kNN classifier using only the 2D embedding.

    Rows with missing labels are ignored. The split count is reduced when a
    class has fewer examples than requested, while classes with fewer than two
    examples raise an error because stratified cross-validation is impossible.
    """
    if (
        not isinstance(n_neighbors, (int, np.integer))
        or isinstance(n_neighbors, bool)
        or n_neighbors < 1
    ):
        raise ValueError("n_neighbors must be a positive integer.")
    if (
        not isinstance(n_splits, (int, np.integer))
        or isinstance(n_splits, bool)
        or n_splits < 2
    ):
        raise ValueError("n_splits must be an integer of at least 2.")
    if label_key not in obs:
        raise KeyError(f"Label column {label_key!r} is not present in obs.")

    coordinates = np.asarray(embedding, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError("embedding must contain at least two coordinates per row.")
    coordinates = np.ascontiguousarray(coordinates[:, :2])
    if not np.isfinite(coordinates).all():
        raise ValueError("embedding coordinates must be finite.")
    if len(obs) != len(coordinates):
        raise ValueError("obs rows must match the embedding rows.")

    labels = pd.Categorical(obs[label_key]).remove_unused_categories()
    valid = labels.codes >= 0
    X = coordinates[valid]
    y = labels.codes[valid]
    classes = np.asarray(labels.categories)
    if len(classes) == 0:
        raise ValueError("At least one labeled class is required for CV.")
    counts = np.bincount(y, minlength=len(classes))
    min_class_count = int(counts.min())
    if min_class_count < 2:
        raise ValueError("Each class needs at least two labeled instances for CV.")
    effective_splits = min(int(n_splits), min_class_count)
    min_train_size = len(y) - int(np.ceil(len(y) / effective_splits))
    if n_neighbors > min_train_size:
        raise ValueError(
            f"n_neighbors={n_neighbors} exceeds the smallest CV training fold "
            f"({min_train_size} instances)."
        )

    cv = StratifiedKFold(
        n_splits=effective_splits, shuffle=True, random_state=random_state
    )
    predictions = cross_val_predict(
        KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=1),
        X,
        y,
        cv=cv,
        n_jobs=-1,
    )
    matrix = confusion_matrix(y, predictions, labels=np.arange(len(classes)))
    correct_mask = np.zeros(len(labels), dtype=bool)
    correct_mask[valid] = y == predictions
    return {
        "accuracy": accuracy_score(y, predictions),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "n_neighbors": int(n_neighbors),
        "n_splits": effective_splits,
        "classes": classes,
        "y_true": y,
        "y_pred": predictions,
        "valid_mask": valid,
        "correct_mask": correct_mask,
        "confusion_matrix": matrix,
    }


def plot_knn_confusion_matrix(results: dict, *, normalize: str = "true"):
    """Plot a confusion matrix from ``cross_validate_embedding_knn`` results."""
    if normalize not in {"true", "pred", "all", None}:
        raise ValueError("normalize must be 'true', 'pred', 'all', or None.")
    matrix = results["confusion_matrix"].astype(float)
    if normalize is not None:
        denominator = {
            "true": matrix.sum(axis=1, keepdims=True),
            "pred": matrix.sum(axis=0, keepdims=True),
            "all": np.asarray([[matrix.sum()]]),
        }[normalize]
        matrix = np.divide(
            matrix, denominator, out=np.zeros_like(matrix), where=denominator != 0
        )
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=results["classes"],
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    display.plot(
        ax=ax,
        cmap="Blues",
        values_format=".2f" if normalize else "d",
        colorbar=False,
    )
    ax.set_title(
        f"{results['n_splits']}-fold CV kNN on 2D embedding "
        f"(k={results['n_neighbors']})"
    )
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    return fig, ax


def _iterated_mst_edges_2d(
    embedding: np.ndarray,
    n_trees: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build exact edge-disjoint Euclidean MSTs directly on 2D coordinates."""
    n_samples = len(embedding)
    if n_samples < 2:
        raise ValueError("At least two embedding points are needed to build an MST.")
    if n_trees > n_samples // 2:
        raise ValueError(
            f"At most {n_samples // 2} edge-disjoint spanning trees can be built "
            f"from {n_samples} points."
        )

    # Keep only one availability bit per possible edge instead of retaining a
    # dense pairwise-distance matrix. Recompute each Prim distance row in 2D.
    available = np.full(
        (n_samples, (n_samples + 7) // 8), 0xFF, dtype=np.uint8
    )
    sample_ids = np.arange(n_samples)
    available[sample_ids, sample_ids >> 3] &= np.bitwise_not(
        (1 << (sample_ids & 7)).astype(np.uint8)
    )
    edges: list[tuple[int, int]] = []
    ranks: list[int] = []

    for rank in range(1, n_trees + 1):
        in_tree = np.zeros(n_samples, dtype=bool)
        best = np.full(n_samples, np.inf, dtype=np.float64)
        parent = np.full(n_samples, -1, dtype=np.intp)
        best[0] = 0.0

        for _ in range(n_samples):
            node = int(np.argmin(np.where(in_tree, np.inf, best)))
            if not np.isfinite(best[node]):
                raise ValueError(
                    f"Could not construct MST rank {rank}; reduce n_trees."
                )
            in_tree[node] = True

            if parent[node] != -1:
                other = int(parent[node])
                edges.append((other, node))
                ranks.append(rank)
                available[other, node >> 3] &= np.uint8(
                    0xFF ^ (1 << (node & 7))
                )
                available[node, other >> 3] &= np.uint8(
                    0xFF ^ (1 << (other & 7))
                )

            remaining = ~in_tree
            available_row = np.unpackbits(
                available[node], bitorder="little", count=n_samples
            ).astype(bool)
            connectable = remaining & available_row
            delta = embedding - embedding[node]
            squared_distances = np.einsum("ij,ij->i", delta, delta)
            improve = connectable & (squared_distances < best)
            best[improve] = squared_distances[improve]
            parent[improve] = node

    return (
        np.asarray(edges, dtype=np.intp).reshape(-1, 2),
        np.asarray(ranks, dtype=np.int32),
    )


def plot_mst_edges(
    embedding: np.ndarray,
    obs: pd.DataFrame,
    *,
    n_trees: int = 3,
    color_key: str = "celltype",
    cmap: str = "tab20",
    linewidth: float = 0.8,
):
    """Build and plot iterated MSTs from the fitted 2D embedding.

    Each edge joins the two endpoint colors by averaging their RGB values. Its
    opacity is ``1 / rank``. No node markers are drawn.
    """
    if (
        not isinstance(n_trees, (int, np.integer))
        or isinstance(n_trees, bool)
        or n_trees < 1
    ):
        raise ValueError("n_trees must be a positive integer.")
    if color_key not in obs:
        raise KeyError(f"Color column {color_key!r} is not present in obs.")

    coordinates = np.asarray(embedding, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError("embedding must contain at least two coordinates per row.")
    coordinates = np.ascontiguousarray(coordinates[:, :2])
    if not np.isfinite(coordinates).all():
        raise ValueError("embedding coordinates must be finite.")
    if len(obs) != len(coordinates):
        raise ValueError("obs rows must match the embedding rows.")

    selected_edges, selected_ranks = _iterated_mst_edges_2d(
        coordinates, n_trees
    )
    labels = pd.Categorical(obs[color_key])
    colors = plt.get_cmap(cmap, max(len(labels.categories), 1))
    category_colors = np.asarray(
        [to_rgba(colors(index))[:3] for index in range(len(labels.categories))],
        dtype=float,
    )
    endpoint_colors = np.empty((len(labels), 3), dtype=float)
    valid_labels = labels.codes >= 0
    endpoint_colors[valid_labels] = category_colors[labels.codes[valid_labels]]
    endpoint_colors[~valid_labels] = to_rgba("#808080")[:3]

    blended = (
        endpoint_colors[selected_edges[:, 0]]
        + endpoint_colors[selected_edges[:, 1]]
    ) / 2
    edge_colors = [
        (*color, 1.0 / rank)
        for color, rank in zip(blended, selected_ranks)
    ]
    segments = coordinates[selected_edges]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.add_collection(
        LineCollection(
            segments,
            colors=edge_colors,
            linewidths=linewidth,
            capstyle="round",
        )
    )
    ax.autoscale()
    ax.margins(0.03)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(
        f"{n_trees} exact iterated MSTs on the 2D embedding, "
        f"colored by blended {color_key} labels"
    )
    ax.set_xlabel("Embedding 1")
    ax.set_ylabel("Embedding 2")
    fig.tight_layout()
    return fig, ax


def plot_class_neighborhood_hulls(
    embedding: np.ndarray,
    obs: pd.DataFrame,
    *,
    k: int = 5,
    color_key: str = "celltype",
    cmap: str = "tab20",
    alpha: float = 0.5,
    include_mask: np.ndarray | None = None,
):
    """Plot the union of same-class local convex hulls in the 2D embedding.

    For each point, its neighborhood contains the point itself and its ``k``
    nearest other points from the same class. The convex hulls of those
    neighborhoods are combined into one filled path per class, so overlaps
    within a class receive the requested alpha only once. When ``include_mask``
    is supplied, excluded points are omitted from both the hulls and neighbor
    searches. Neighborhoods without three non-collinear coordinates are skipped.
    """
    if (
        not isinstance(k, (int, np.integer))
        or isinstance(k, bool)
        or k < 1
    ):
        raise ValueError("k must be a positive integer.")
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")
    if color_key not in obs:
        raise KeyError(f"Color column {color_key!r} is not present in obs.")

    coordinates = np.asarray(embedding, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError("embedding must contain at least two coordinates per row.")
    coordinates = np.ascontiguousarray(coordinates[:, :2])
    if not np.isfinite(coordinates).all():
        raise ValueError("embedding coordinates must be finite.")
    if len(obs) != len(coordinates):
        raise ValueError("obs rows must match the embedding rows.")

    labels = pd.Categorical(obs[color_key])
    if include_mask is None:
        include_mask = np.ones(len(coordinates), dtype=bool)
    else:
        include_mask = np.asarray(include_mask, dtype=bool)
        if include_mask.shape != (len(coordinates),):
            raise ValueError("include_mask must have one boolean per embedding row.")
    plot_mask = include_mask & (labels.codes >= 0)
    category_colors = plt.get_cmap(cmap, max(len(labels.categories), 1))
    fig, ax = plt.subplots(figsize=(8, 7))
    legend_handles = []

    for category_index, category in enumerate(labels.categories):
        class_indices = np.flatnonzero(
            (labels.codes == category_index) & include_mask
        )
        if len(class_indices) < 3:
            continue

        class_coordinates = coordinates[class_indices]
        n_neighbors = min(k + 1, len(class_indices))
        neighbors = NearestNeighbors(n_neighbors=n_neighbors)
        neighbors.fit(class_coordinates)
        neighborhood_indices = neighbors.kneighbors(
            class_coordinates, return_distance=False
        )

        path_vertices = []
        path_codes = []
        seen_neighborhoods = set()
        for neighborhood in neighborhood_indices:
            neighborhood_key = tuple(sorted(map(int, neighborhood)))
            if neighborhood_key in seen_neighborhoods:
                continue
            seen_neighborhoods.add(neighborhood_key)
            neighborhood_coordinates = class_coordinates[neighborhood]
            try:
                hull = ConvexHull(neighborhood_coordinates)
            except QhullError:
                continue

            polygon = neighborhood_coordinates[hull.vertices]
            polygon = np.vstack((polygon, polygon[0]))
            codes = np.full(len(polygon), MplPath.LINETO, dtype=np.uint8)
            codes[0] = MplPath.MOVETO
            codes[-1] = MplPath.CLOSEPOLY
            path_vertices.append(polygon)
            path_codes.append(codes)

        color = category_colors(category_index)
        if path_vertices:
            compound_path = MplPath(
                np.concatenate(path_vertices), np.concatenate(path_codes)
            )
            ax.add_patch(
                PathPatch(
                    compound_path,
                    facecolor=color,
                    edgecolor="none",
                    alpha=alpha,
                )
            )
        legend_handles.append(
            Patch(facecolor=color, edgecolor="none", alpha=alpha, label=str(category))
        )

    if plot_mask.any():
        ax.update_datalim(coordinates[plot_mask])
    else:
        ax.update_datalim(coordinates)
    ax.autoscale_view()
    ax.margins(0.03)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"Unions of {k}-nearest-neighbor hulls by {color_key}")
    ax.set_xlabel("Embedding 1")
    ax.set_ylabel("Embedding 2")
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            title=color_key,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=False,
        )
    fig.tight_layout()
    return fig, ax

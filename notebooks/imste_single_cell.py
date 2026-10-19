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
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch, PathPatch
from matplotlib.path import Path as MplPath
from scipy import sparse
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError, cKDTree
from scipy.stats import gaussian_kde
from sklearn.manifold import trustworthiness
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


DATA_FILE = "human_pancreas_norm_complexBatch.h5ad"


def combine_feature_blocks(
    original_train: np.ndarray,
    original_test: np.ndarray,
    partition_train: np.ndarray,
    partition_test: np.ndarray,
    *,
    normalize_blocks: bool = True,
    alpha: float = 0.5,
) -> tuple[np.ndarray, dict[str, float]]:
    """Combine original and partition features with configurable block weights.

    When normalization is enabled, each block is scaled to unit root mean
    squared row norm on the training data before applying weights ``alpha``
    and ``1 - alpha``.
    The same training-derived scales are applied to test rows, without fitting
    any scale on the test data.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    blocks = {
        "original": (np.asarray(original_train), np.asarray(original_test)),
        "partition": (np.asarray(partition_train), np.asarray(partition_test)),
    }
    block_weights = {"original": alpha, "partition": 1.0 - alpha}
    scales: dict[str, float] = {}
    scaled_test_blocks = []
    for name, (train_block, test_block) in blocks.items():
        if train_block.ndim != 2 or test_block.ndim != 2:
            raise ValueError(f"{name} feature blocks must be two-dimensional.")
        if train_block.shape[1] != test_block.shape[1]:
            raise ValueError(f"Train and test {name} feature widths must match.")
        if normalize_blocks:
            mean_squared_norm = float(np.mean(np.sum(train_block**2, axis=1)))
            rms_norm = np.sqrt(mean_squared_norm)
            norm_scale = 1.0 / rms_norm if rms_norm > 0 else 1.0
        else:
            norm_scale = 1.0
        scale = norm_scale * block_weights[name]
        scales[name] = scale
        scaled_test_blocks.append(test_block * scale)
    return np.hstack(scaled_test_blocks), scales


def plot_density_parent_tree(
    embedding: np.ndarray,
    obs: pd.DataFrame,
    *,
    label_key: str,
    ax=None,
    point_size: float = 8,
    point_alpha: float = 0.7,
    edge_alpha: float = 0.65,
    cross_label_edge_alpha: float = 0.25,
    edge_alpha_exponent: float = 1.5,
    linewidth: float = 0.7,
    k: int = 1,
    edge_length_std_threshold: float = 2.0,
    legend_position: str = "bottom",
    label_colors: dict[str, object] | None = None,
):
    """Plot a KDE density-parent graph over the labeled 2D embedding.

    Each point connects to its ``k`` nearest points with higher KDE density,
    or all such points when fewer than ``k`` exist. Equal-density points are
    ordered by row index to ensure every non-root point has a parent. Edge
    color is the mean of its endpoint label colors; opacity decreases as
    ``edge_alpha / link_rank**edge_alpha_exponent`` for each point's ordered
    parent links.
    """
    coordinates = np.asarray(embedding, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("embedding must be a two-dimensional coordinate array.")
    if len(obs) != len(coordinates):
        raise ValueError("obs rows must match the embedding rows.")
    if label_key not in obs:
        raise KeyError(f"Label column {label_key!r} is not present in obs.")
    if len(coordinates) < 2 or not np.isfinite(coordinates).all():
        raise ValueError("embedding must contain at least two finite coordinates.")
    if legend_position not in {"inside", "right", "bottom"}:
        raise ValueError("legend_position must be 'inside', 'right', or 'bottom'.")
    if (
        not np.isfinite(edge_length_std_threshold)
        or edge_length_std_threshold < 0
    ):
        raise ValueError("edge_length_std_threshold must be finite and non-negative.")
    if not np.isfinite(edge_alpha_exponent) or edge_alpha_exponent < 0:
        raise ValueError("edge_alpha_exponent must be finite and non-negative.")
    if (
        not np.isfinite(cross_label_edge_alpha)
        or not 0.0 <= cross_label_edge_alpha <= 1.0
    ):
        raise ValueError("cross_label_edge_alpha must be between 0 and 1.")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer.")
    k = min(int(k), len(coordinates) - 1)

    density = gaussian_kde(coordinates.T)(coordinates.T)
    density_order = np.lexsort((np.arange(len(coordinates)), density))
    density_rank = np.empty(len(coordinates), dtype=np.intp)
    density_rank[density_order] = np.arange(len(coordinates))

    parents = np.full((len(coordinates), k), -1, dtype=np.intp)
    unresolved = density_order[:-1].copy()
    search_tree = cKDTree(coordinates)
    neighbor_count = min(max(2, k + 1), len(coordinates))
    while unresolved.size:
        _, neighbors = search_tree.query(
            coordinates[unresolved], k=neighbor_count, workers=-1
        )
        if neighbor_count == 1:
            neighbors = neighbors[:, None]
        higher_density = density_rank[neighbors] > density_rank[unresolved, None]
        enough_parents = higher_density.sum(axis=1) >= k
        all_neighbors_seen = neighbor_count == len(coordinates)
        resolved_mask = enough_parents | all_neighbors_seen
        if np.any(resolved_mask):
            resolved_neighbors = neighbors[resolved_mask]
            resolved_higher_density = higher_density[resolved_mask]
            for row_index, point_index in enumerate(unresolved[resolved_mask]):
                higher_neighbors = resolved_neighbors[row_index][
                    resolved_higher_density[row_index]
                ]
                parents[point_index, :min(k, len(higher_neighbors))] = higher_neighbors[:k]
        unresolved = unresolved[~resolved_mask]
        if unresolved.size:
            neighbor_count = min(2 * neighbor_count, len(coordinates))

    labels = pd.Categorical(obs[label_key])
    palette = plt.get_cmap("tab20", max(len(labels.categories), 1))
    if label_colors is None:
        point_colors = palette(
            np.clip(labels.codes, 0, max(len(labels.categories) - 1, 0))
        )
        point_colors[labels.codes < 0] = (0.5, 0.5, 0.5, 1.0)
    else:
        missing_color = (0.5, 0.5, 0.5, 1.0)
        point_colors = np.asarray([
            label_colors.get(str(value), missing_color)
            if not pd.isna(value) else missing_color
            for value in obs[label_key]
        ], dtype=float)
    edge_mask = parents >= 0
    child_grid, edge_rank_grid = np.indices(parents.shape)
    child_indices = child_grid[edge_mask]
    parent_indices = parents[edge_mask]
    edge_ranks = edge_rank_grid[edge_mask] + 1
    edge_lengths = np.linalg.norm(
        coordinates[child_indices] - coordinates[parent_indices], axis=1
    )
    mean_edge_length = float(edge_lengths.mean())
    std_edge_length = float(edge_lengths.std())
    edge_length_cutoff = (
        mean_edge_length + edge_length_std_threshold * std_edge_length
    )
    keep_edge = edge_lengths <= edge_length_cutoff
    filtered_children = child_indices[keep_edge]
    filtered_parents = parent_indices[keep_edge]
    segments = np.stack(
        (coordinates[filtered_children], coordinates[filtered_parents]), axis=1
    )
    edge_colors = (
        point_colors[filtered_children] + point_colors[filtered_parents]
    ) / 2
    filtered_edge_ranks = edge_ranks[keep_edge]
    cross_label_edge = (
        labels.codes[filtered_children] != labels.codes[filtered_parents]
    )
    base_edge_alphas = np.where(
        cross_label_edge, cross_label_edge_alpha, edge_alpha
    )
    edge_alphas = base_edge_alphas / filtered_edge_ranks**edge_alpha_exponent
    edge_colors[:, 3] *= edge_alphas
    omitted_edges = int((~keep_edge).sum())

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))
    ax.add_collection(
        LineCollection(
            segments,
            colors=edge_colors,
            linewidths=linewidth,
            zorder=1,
        )
    )
    ax.scatter(
        coordinates[:, 0], coordinates[:, 1], c=point_colors,
        s=point_size, alpha=point_alpha, linewidths=0, zorder=2,
    )
    ax.set_title(
        f"KDE density-parent graph (k={k}); mean edge {mean_edge_length:.3g}; "
        f"{omitted_edges} edges omitted (>{edge_length_std_threshold:g} SD)"
    )
    ax.set_xlabel("IMSTE 1")
    ax.set_ylabel("IMSTE 2")
    ax.set_aspect("equal", adjustable="box")

    handles = [
        Line2D(
            [], [], marker="o", linestyle="none",
            color=(label_colors[str(value)] if label_colors is not None else palette(index)),
            markersize=6, label=str(value)
        )
        for index, value in enumerate(labels.categories)
    ]
    if np.any(labels.codes < 0):
        handles.append(
            Line2D(
                [], [], marker="o", linestyle="none", color="0.5",
                markersize=6, label="(missing)"
            )
        )
    if legend_position == "right":
        legend_options = {
            "bbox_to_anchor": (1.02, 1), "loc": "upper left", "frameon": False
        }
    elif legend_position == "bottom":
        legend_options = {
            "bbox_to_anchor": (0.5, -0.16),
            "loc": "upper center",
            "ncol": min(7, max(1, len(labels.categories))),
            "frameon": False,
        }
    else:
        legend_options = {
            "loc": "upper right", "frameon": True, "framealpha": 0.8
        }
    ax.legend(
        handles,
        [handle.get_label() for handle in handles],
        title=label_key,
        **legend_options,
    )
    return ax, {
        "density": density,
        "parent_indices": parents,
        "edge_lengths": edge_lengths,
        "edge_ranks": edge_ranks,
        "edge_alphas": edge_alphas,
        "cross_label_edge": cross_label_edge,
        "mean_edge_length": mean_edge_length,
        "std_edge_length": std_edge_length,
        "edge_length_cutoff": edge_length_cutoff,
        "omitted_edges": omitted_edges,
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
    excluded_terms = ("pca", "pcca", "scalp", "clasp")
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
        if "obs" not in h5:
            return []
        obs_node = h5["obs"]
        if isinstance(obs_node, h5py.Dataset):
            return [
                key for key in (obs_node.dtype.names or ())
                if not key.startswith("__")
            ]
        if not isinstance(obs_node, h5py.Group):
            return []
        columns = []
        for key, node in obs_node.items():
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


def _h5_obs_column_keys(obs_node: h5py.Group | h5py.Dataset) -> set[str]:
    if isinstance(obs_node, h5py.Dataset):
        return set(obs_node.dtype.names or ())
    if isinstance(obs_node, h5py.Group):
        return set(obs_node.keys())
    return set()


def _h5_dataframe_index_key(obs_node: h5py.Group | h5py.Dataset) -> str | None:
    """Resolve the obs index column from either H5AD dataframe storage layout."""
    available_keys = _h5_obs_column_keys(obs_node)
    value = obs_node.attrs.get("_index", "_index")
    if isinstance(value, np.ndarray):
        if value.size != 1:
            value = None
        else:
            value = value.reshape(-1)[0]
    if isinstance(value, np.void):
        names = value.dtype.names or ()
        if len(names) == 1:
            value = value[names[0]]
        else:
            value = None
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8")
    if isinstance(value, (str, np.str_)):
        key = str(value)
        if key in available_keys:
            return key
    for key in ("_index", "index"):
        if key in available_keys:
            return key
    return None

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


def _read_h5ad_obs_column(
    obs_node: h5py.Group | h5py.Dataset,
    key: str,
    row_indices: np.ndarray,
):
    """Read a regular or categorical AnnData obs column."""
    if isinstance(obs_node, h5py.Dataset):
        if obs_node.dtype.names and key in obs_node.dtype.names:
            return _decode_h5_strings(obs_node.fields(key)[row_indices])
        raise KeyError(f"Obs column {key!r} is not present in the structured obs dataset.")

    node = obs_node[key]
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
    matrix_key: str = "X",
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
            key: _read_h5ad_obs_column(obs_group, key, row_indices)
            for key in obs_keys
            if key in _h5_obs_column_keys(obs_group)
        }
        obs_index_key = _h5_dataframe_index_key(obs_group)
        if obs_index_key is not None:
            obs_names = _decode_h5_strings(
                _read_h5ad_obs_column(obs_group, obs_index_key, row_indices)
            ).astype(str)
        else:
            obs_names = row_indices.astype(str)

    X = np.asarray(X, dtype=np.float32, order="C")
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("Loaded representation must be a finite 2D matrix.")
    return X, pd.DataFrame(obs_columns, index=obs_names), row_indices


def load_dataset_with_train_test(
    path: str | Path,
    *,
    matrix_key: str = "X",
    obs_keys: tuple[str, ...] = ("tech", "celltype"),
    max_test_cells: int = 4000,
    train_instances_per_class: int = 50,
    min_cells_per_class: int = 50,
    label_key: str | None = None,
    random_state: int = 42,
):
    """Load disjoint per-class training rows and a capped test subset.

    Classes with fewer than ``min_cells_per_class`` total labeled rows are
    excluded. Up to ``train_instances_per_class`` rows are sampled per
    retained class, while reserving one row per class for testing. The test set
    includes one row from each retained class and is filled uniformly up to
    ``max_test_cells``. Only selected matrix rows are loaded into memory.
    """
    if not isinstance(max_test_cells, (int, np.integer)) or max_test_cells < 1:
        raise ValueError("max_test_cells must be a positive integer.")
    if (
        not isinstance(train_instances_per_class, (int, np.integer))
        or train_instances_per_class < 1
    ):
        raise ValueError("train_instances_per_class must be a positive integer.")
    if not isinstance(min_cells_per_class, (int, np.integer)) or min_cells_per_class < 1:
        raise ValueError("min_cells_per_class must be a positive integer.")

    path = Path(path)
    rng = np.random.default_rng(random_state)
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
        all_rows = np.arange(n_obs, dtype=np.int64)
        obs_group = h5["obs"]
        full_obs = pd.DataFrame(
            {
                key: _read_h5ad_obs_column(obs_group, key, all_rows)
                for key in obs_keys
                if key in _h5_obs_column_keys(obs_group)
            }
        )
        if label_key is None:
            label_key = default_label_key(full_obs)
        if label_key not in full_obs:
            raise KeyError(f"Label column {label_key!r} is not present in the H5AD obs.")

        labels = full_obs[label_key]
        class_counts = labels.value_counts(dropna=True)
        excluded_class_counts = class_counts[
            class_counts < int(min_cells_per_class)
        ].sort_index()
        retained_labels = class_counts[
            class_counts >= int(min_cells_per_class)
        ].index
        retained_rows = all_rows[labels.isin(retained_labels).to_numpy()]
        train_rows = []
        reserved_test_rows = []
        for label in retained_labels:
            class_rows = all_rows[(labels == label).to_numpy()]
            shuffled = rng.permutation(class_rows)
            # Keep one labeled cell aside for the held-out set.
            n_train = min(int(train_instances_per_class), len(shuffled) - 1)
            train_rows.extend(shuffled[:n_train])
            reserved_test_rows.append(shuffled[n_train])

        train_rows = np.asarray(train_rows, dtype=np.int64)
        reserved_test_rows = np.asarray(reserved_test_rows, dtype=np.int64)
        if len(retained_labels) == 0:
            raise ValueError(
                f"No classes have at least {min_cells_per_class} labeled cells."
            )
        if max_test_cells < len(reserved_test_rows):
            raise ValueError(
                f"max_test_cells={max_test_cells} is too small to reserve one test "
                f"cell for each of {len(reserved_test_rows)} classes."
            )
        remaining_rows = np.setdiff1d(retained_rows, train_rows, assume_unique=False)
        extra_candidates = np.setdiff1d(
            remaining_rows, reserved_test_rows, assume_unique=False
        )
        n_test = min(int(max_test_cells), len(remaining_rows))
        n_extra = n_test - len(reserved_test_rows)
        if n_extra > 0:
            extra_test_rows = rng.choice(extra_candidates, size=n_extra, replace=False)
            test_rows = np.concatenate((reserved_test_rows, extra_test_rows))
        else:
            test_rows = reserved_test_rows.copy()

        obs_index_key = _h5_dataframe_index_key(obs_group)

        def read_rows(rows):
            sorted_rows = np.sort(rows)
            reorder = np.searchsorted(sorted_rows, rows)
            matrix = _read_h5ad_matrix(matrix_node, sorted_rows)[reorder]
            columns = {
                key: _read_h5ad_obs_column(obs_group, key, sorted_rows)
                for key in obs_keys
                if key in _h5_obs_column_keys(obs_group)
            }
            if obs_index_key is not None:
                names = _decode_h5_strings(
                    _read_h5ad_obs_column(obs_group, obs_index_key, sorted_rows)
                ).astype(str)
            else:
                names = sorted_rows.astype(str)
            observations = pd.DataFrame(columns, index=names).iloc[reorder].copy()
            for column in observations.select_dtypes(include="category"):
                observations[column] = observations[column].cat.remove_unused_categories()
            return np.asarray(matrix, dtype=np.float32, order="C"), observations

        X_train, obs_train = read_rows(train_rows)
        X_test, obs_test = read_rows(test_rows)

    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError("Loaded representations must contain only finite values.")
    return (
        X_train,
        obs_train,
        train_rows,
        X_test,
        obs_test,
        test_rows,
        label_key,
        excluded_class_counts,
    )


def load_embedder_class():
    """Import IMSTE from the environment or an optional source checkout."""
    try:
        from imste import IteratedMinimumSpanningTreeEmbedder
    except ImportError:
        configured_root = os.environ.get("IMSTE_REPO")
        candidate_roots = (
            [Path(configured_root)]
            if configured_root
            else [
                Path.home() / "Code" / "IteratedMinimumSpanningTree",
                Path.home() / "Code" / "mst_embedding",
            ]
        )
        imste_root = next(
            (root for root in candidate_roots if (root / "imste").is_dir()),
            None,
        )
        if imste_root is None:
            raise ImportError(
                "Install imste or set IMSTE_REPO to the IMSTE repository root "
                "(the directory containing the imste package)."
            )
        sys.path.insert(0, str(imste_root))
        from imste import IteratedMinimumSpanningTreeEmbedder
    return IteratedMinimumSpanningTreeEmbedder


def fit_embedding(
    X: np.ndarray,
    params: dict,
    *,
    auto_mst_fallback: bool = True,
    fallback_dense_memory_gb: float = 1.0,
):
    """Fit IMSTE with bounded FAMST retries and a best-effort MST fallback.

    A FAMST candidate-capacity failure first retries with a larger neighbor
    graph. If that still fails, exact Prim is used when its dense distance
    matrix fits the configured memory budget. For larger datasets, the number
    of trees is reduced one at a time until FAMST succeeds.
    """
    embedder_class = load_embedder_class()
    params = dict(params)
    if not np.isfinite(fallback_dense_memory_gb) or fallback_dense_memory_gb < 0:
        raise ValueError("fallback_dense_memory_gb must be non-negative and finite.")

    def run_fit(fit_params):
        estimator = embedder_class(**fit_params)
        return estimator, estimator.fit_transform(X)

    requested_n_msts = int(params.get("n_msts", 15))
    fallback_used = "none"
    try:
        estimator, embedding = run_fit(params)
    except ValueError as initial_error:
        capacity_failure = "FAMST candidate graph could not supply" in str(initial_error)
        if (
            not auto_mst_fallback
            or params.get("mst_method", "famst") != "famst"
            or not capacity_failure
        ):
            raise

        n_samples = len(X)
        original_cap = params.get("mst_max_neighbors")
        if original_cap is None:
            starting_neighbors = max(
                int(params.get("mst_neighbors", 15)), 2 * requested_n_msts
            )
            original_cap = max(4 * starting_neighbors, 4 * requested_n_msts)
        retry_cap = min(
            max(n_samples - 2, 1),
            max(2 * int(original_cap), 1024),
        )
        retry_params = {
            **params,
            "mst_neighbors": retry_cap,
            "mst_max_neighbors": retry_cap,
        }
        last_capacity_error = initial_error
        if retry_cap > int(original_cap):
            print(
                f"FAMST could not build {requested_n_msts} trees; "
                f"retrying with mst_neighbors={retry_cap}."
            )
            try:
                estimator, embedding = run_fit(retry_params)
                fallback_used = f"FAMST retry at {retry_cap} neighbors"
            except ValueError as retry_error:
                if "FAMST candidate graph could not supply" not in str(retry_error):
                    raise
                last_capacity_error = retry_error
            else:
                retry_params = None

        if retry_params is not None:
            dense_memory_gb = n_samples * n_samples * 8 / (1024**3)
            if dense_memory_gb <= fallback_dense_memory_gb:
                exact_n_msts = requested_n_msts
                print(
                    f"FAMST retry failed; trying exact Prim with "
                    f"{exact_n_msts} trees (estimated distance matrix "
                    f"{dense_memory_gb:.2f} GiB)."
                )
                prim_params = {
                    **params,
                    "mst_method": "prim",
                    "n_msts": exact_n_msts,
                }
                prim_succeeded = False
                while exact_n_msts >= 1:
                    prim_params["n_msts"] = exact_n_msts
                    try:
                        estimator, embedding = run_fit(prim_params)
                        fallback_used = f"exact Prim with {exact_n_msts} trees"
                        prim_succeeded = True
                        break
                    except MemoryError:
                        print(
                            "Exact Prim exceeded available memory; "
                            "continuing with sparse FAMST fallbacks."
                        )
                        break
                    except ValueError as prim_error:
                        if "Could not connect all samples in tree" not in str(prim_error):
                            raise
                        exact_n_msts -= 1
                if not prim_succeeded and exact_n_msts < 1:
                    raise last_capacity_error
                if prim_succeeded:
                    retry_params = None

        if retry_params is not None:
            # Dense Prim would exceed the configured budget. Keep FAMST sparse
            # and retain as many trees as its candidate graph can support.
            for effective_n_msts in range(requested_n_msts - 1, 0, -1):
                reduced_params = {
                    **retry_params,
                    "n_msts": effective_n_msts,
                }
                print(f"Retrying FAMST with {effective_n_msts} trees.")
                try:
                    estimator, embedding = run_fit(reduced_params)
                except ValueError as reduced_error:
                    if "FAMST candidate graph could not supply" not in str(reduced_error):
                        raise
                    last_capacity_error = reduced_error
                else:
                    fallback_used = (
                        f"FAMST with {effective_n_msts} trees "
                        f"and {retry_cap} neighbors"
                    )
                    break
            else:
                raise ValueError(
                    "Automatic MST fallback could not construct even one tree. "
                    f"FAMST failure: {last_capacity_error}"
                ) from last_capacity_error

    estimator.requested_n_msts_ = requested_n_msts
    estimator.effective_n_msts_ = int(estimator.n_msts)
    estimator.mst_fallback_ = fallback_used
    score = trustworthiness(X, embedding, n_neighbors=10)
    print(f"IMSTE source: {embedder_class.__module__}")
    print(
        f"MST trees: {estimator.effective_n_msts_} / {requested_n_msts} requested; "
        f"method: {estimator.mst_method}; fallback: {fallback_used}"
    )
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
    point_alpha: float = 0.8,
    ax=None,
    legend_position: str | None = None,
    label_colors: dict[str, object] | None = None,
):
    """Plot the embedding colored by available metadata columns."""
    color_keys = [key for key in color_keys if key in obs]
    if not color_keys:
        raise ValueError("None of the requested color columns are present in obs.")
    if ax is None:
        fig, axes = plt.subplots(
            1,
            len(color_keys),
            figsize=(1.3 * point_size * len(color_keys), point_size),
            squeeze=False,
        )
        created_figure = True
    else:
        axes = np.atleast_1d(ax).ravel()
        if len(axes) != len(color_keys):
            raise ValueError("Provide one axes object for each color key.")
        fig = axes[0].figure
        axes = axes.reshape(1, -1)
        created_figure = False
    if legend_position is None:
        legend_position = "right" if created_figure else "inside"
    if legend_position not in {"inside", "right", "bottom"}:
        raise ValueError("legend_position must be 'inside', 'right', or 'bottom'.")

    for ax, key in zip(axes.ravel(), color_keys):
        labels = pd.Categorical(obs[key])
        if label_colors is None:
            points = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=labels.codes,
                cmap="tab20",
                s=point_size,
                alpha=point_alpha,
                linewidths=0,
                zorder=3,
            )
            handles, _ = points.legend_elements(num=len(labels.categories))
        else:
            missing_color = (0.5, 0.5, 0.5, 1.0)
            point_colors = [
                label_colors.get(str(value), missing_color)
                if not pd.isna(value) else missing_color
                for value in obs[key]
            ]
            ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=point_colors,
                s=point_size,
                alpha=point_alpha,
                linewidths=0,
                zorder=3,
            )
            handles = [
                Line2D(
                    [], [], marker="o", linestyle="none",
                    color=label_colors[str(value)], markersize=6, label=str(value)
                )
                for value in labels.categories
            ]
            if obs[key].isna().any():
                handles.append(Line2D(
                    [], [], marker="o", linestyle="none", color=missing_color,
                    markersize=6, label="(missing)"
                ))
        ax.set_title(f"IMSTE colored by {key}")
        ax.set_xlabel("IMSTE 1")
        ax.set_ylabel("IMSTE 2")
        ax.set_aspect("equal", adjustable="box")
        if legend_position == "right":
            legend_options = {
                "bbox_to_anchor": (1.02, 1), "loc": "upper left", "frameon": False
            }
        elif legend_position == "bottom":
            legend_options = {
                "bbox_to_anchor": (0.5, -0.16),
                "loc": "upper center",
                "ncol": min(7, max(1, len(labels.categories))),
                "frameon": False,
            }
        else:
            legend_options = {
                "loc": "upper right", "frameon": True, "framealpha": 0.8
            }
        ax.legend(
            handles,
            [handle.get_label() for handle in handles],
            title=key,
            **legend_options,
        )
    if created_figure:
        fig.tight_layout()
    return fig, axes


def interactive_label_guided_embedding(
    embedding: np.ndarray,
    obs: pd.DataFrame,
    label_key: str,
    *,
    initial_k: int = 10,
    initial_iterations: int = 20,
    initial_distance_fraction: float = 0.25,
    max_k: int = 50,
    max_iterations: int = 200,
    point_size: float = 6,
):
    """Show an interactive label-guided adjustment of a 2D embedding.

    For each point, the viewer separately finds its ``k`` nearest same-label
    points and its ``k`` nearest different-label points in the original
    embedding. During each iteration it moves toward the same-label neighbors
    and repels from different-label neighbors only when they are closer than a
    small separation margin. Balance 0 applies only different-label repulsion;
    balance 1 applies only the same-label springs. The movement slider
    interpolates between the original and adjusted coordinates. Neighbor
    identities remain fixed during the adjustment. Each same-label neighbor
    pair uses a symmetric spring whose rest length is the selected fraction of
    its original distance.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
        from html import escape as html_escape
    except ImportError as error:
        raise ImportError(
            "Install ipywidgets to use the interactive embedding viewer."
        ) from error

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
    n_samples = len(coordinates)
    if n_samples < 2:
        raise ValueError("At least two points are required for nearest neighbors.")
    if not isinstance(max_k, (int, np.integer)) or isinstance(max_k, bool) or max_k < 1:
        raise ValueError("max_k must be a positive integer.")
    max_k = min(int(max_k), n_samples - 1)
    if not isinstance(max_iterations, (int, np.integer)) or isinstance(max_iterations, bool) or max_iterations < 0:
        raise ValueError("max_iterations must be a non-negative integer.")

    labels = pd.Categorical(obs[label_key])
    label_codes = labels.codes
    same_neighbors = np.zeros((n_samples, max_k), dtype=np.intp)
    different_neighbors = np.zeros((n_samples, max_k), dtype=np.intp)
    same_neighbor_mask = np.zeros((n_samples, max_k), dtype=bool)
    different_neighbor_mask = np.zeros((n_samples, max_k), dtype=bool)
    for label_code in np.unique(label_codes):
        class_rows = np.flatnonzero(label_codes == label_code)
        other_rows = np.flatnonzero(label_codes != label_code)

        n_same = min(max_k, max(0, len(class_rows) - 1))
        if n_same:
            same_model = NearestNeighbors(n_neighbors=n_same).fit(
                coordinates[class_rows]
            )
            # kneighbors() with X omitted excludes each fitted row itself.
            same_neighbors[class_rows, :n_same] = class_rows[
                same_model.kneighbors(return_distance=False)
            ]
            same_neighbor_mask[class_rows, :n_same] = True

        n_different = min(max_k, len(other_rows))
        if n_different:
            different_model = NearestNeighbors(n_neighbors=n_different).fit(
                coordinates[other_rows]
            )
            different_neighbors[class_rows, :n_different] = other_rows[
                different_model.kneighbors(
                    coordinates[class_rows], return_distance=False
                )
            ]
            different_neighbor_mask[class_rows, :n_different] = True
    # Scale updates to the visible embedding, not to the often tiny nearest-
    # neighbor spacing in dense single-cell plots.
    global_scale = float(np.sqrt(np.mean(np.var(coordinates, axis=0))))
    if not np.isfinite(global_scale) or global_scale <= 0:
        global_scale = max(float(np.ptp(coordinates, axis=0).mean()), 1e-8)
    step_limit = 0.025 * global_scale
    repulsion_radius = 0.08 * global_scale

    default_k = min(max(1, int(initial_k)), max_k)
    default_iterations = min(max(0, int(initial_iterations)), int(max_iterations))
    k_control = widgets.IntSlider(
        value=default_k,
        min=1,
        max=max_k,
        step=1,
        description="",
        continuous_update=False,
        layout=widgets.Layout(width="480px"),
    )
    iteration_control = widgets.IntSlider(
        value=default_iterations,
        min=0,
        max=int(max_iterations),
        step=1,
        description="",
        continuous_update=False,
        layout=widgets.Layout(width="480px"),
    )
    balance_control = widgets.FloatSlider(
        value=0.5,
        min=0.0,
        max=1.0,
        step=0.05,
        description="",
        readout_format=".2f",
        continuous_update=False,
        layout=widgets.Layout(width="480px"),
    )
    distance_control = widgets.FloatSlider(
        value=float(np.clip(initial_distance_fraction, 0.05, 1.0)),
        min=0.05,
        max=1.0,
        step=0.05,
        description="",
        readout_format=".2f",
        continuous_update=False,
        layout=widgets.Layout(width="480px"),
    )
    movement_control = widgets.FloatSlider(
        value=1.0,
        min=0.0,
        max=1.0,
        step=0.05,
        description="",
        readout_format=".2f",
        continuous_update=False,
        layout=widgets.Layout(width="480px"),
    )
    status = widgets.HTML(
        value='<span style="color:#666">Ready</span>',
        layout=widgets.Layout(margin="4px 0 0 420px"),
    )

    cache: dict[tuple[int, int, float, float], np.ndarray] = {}
    pair_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def same_label_pairs(k: int):
        if int(k) in pair_cache:
            return pair_cache[int(k)]
        mask = same_neighbor_mask[:, : int(k)]
        rows = np.broadcast_to(np.arange(n_samples)[:, None], mask.shape)[mask]
        columns = same_neighbors[:, : int(k)][mask]
        keys = np.unique(
            np.minimum(rows, columns).astype(np.int64) * n_samples
            + np.maximum(rows, columns)
        )
        left = (keys // n_samples).astype(np.intp)
        right = (keys % n_samples).astype(np.intp)
        initial_distances = np.linalg.norm(
            coordinates[right] - coordinates[left], axis=1
        )
        pair_cache.clear()
        pair_cache[int(k)] = (left, right, initial_distances)
        return pair_cache[int(k)]

    def adjusted_coordinates(
        k: int,
        n_iterations: int,
        balance: float,
        distance_fraction: float,
    ) -> np.ndarray:
        balance = float(balance)
        distance_fraction = float(distance_fraction)
        key = (int(k), int(n_iterations), balance, distance_fraction)
        if key in cache:
            return cache[key]
        pair_left, pair_right, initial_pair_distances = same_label_pairs(k)
        rest_lengths = np.maximum(
            initial_pair_distances * distance_fraction,
            0.005 * global_scale,
        )
        different_indices = different_neighbors[:, : int(k)]
        different_mask = different_neighbor_mask[:, : int(k)]
        current = coordinates.copy()

        for _ in range(int(n_iterations)):
            pair_delta = current[pair_right] - current[pair_left]
            pair_distances = np.linalg.norm(pair_delta, axis=1)
            pair_direction = pair_delta / np.maximum(
                pair_distances[:, None], np.finfo(np.float64).eps
            )
            coincident = pair_distances <= np.finfo(np.float64).eps
            if coincident.any():
                # Give exact overlaps a stable direction so their nonzero rest
                # length can separate them instead of leaving a zero force.
                angles = (
                    (pair_left[coincident].astype(np.int64) * 73856093)
                    ^ (pair_right[coincident].astype(np.int64) * 19349663)
                ) % 1_000_003
                angles = angles * (2.0 * np.pi / 1_000_003)
                pair_direction[coincident] = np.column_stack(
                    (np.cos(angles), np.sin(angles))
                )
            spring_magnitude = pair_distances - rest_lengths
            spring_force = spring_magnitude[:, None] * pair_direction
            same_label_force = np.column_stack(
                (
                    np.bincount(pair_left, weights=spring_force[:, 0], minlength=n_samples)
                    - np.bincount(pair_right, weights=spring_force[:, 0], minlength=n_samples),
                    np.bincount(pair_left, weights=spring_force[:, 1], minlength=n_samples)
                    - np.bincount(pair_right, weights=spring_force[:, 1], minlength=n_samples),
                )
            )
            nearby_different = current[different_indices]
            away_vectors = current[:, None, :] - nearby_different
            neighbor_distances = np.linalg.norm(away_vectors, axis=2)
            close_different = different_mask & (neighbor_distances < repulsion_radius)
            shortfall = np.maximum(repulsion_radius - neighbor_distances, 0.0)
            unit_away = away_vectors / np.maximum(
                neighbor_distances[:, :, None], np.finfo(np.float64).eps
            )
            repulsion = unit_away * shortfall[:, :, None] * 0.5
            repulsion[~close_different] = 0.0
            repulsion_count = close_different.sum(axis=1, keepdims=True)
            mean_repulsion = np.divide(
                repulsion.sum(axis=1),
                repulsion_count,
                out=np.zeros_like(current),
                where=repulsion_count != 0,
            )
            force = (
                balance * 3.0 * same_label_force
                + (1.0 - balance) * mean_repulsion * 0.5
            )
            force_size = np.linalg.norm(force, axis=1)
            scale = np.minimum(
                1.0,
                step_limit / np.maximum(force_size, np.finfo(np.float64).tiny),
            )
            current += force * scale[:, None]

        cache.clear()
        cache[key] = current
        return current

    def render_update(
        k: int,
        iterations: int,
        balance: float,
        distance_fraction: float,
        movement: float,
    ):
        adjusted = adjusted_coordinates(k, iterations, balance, distance_fraction)
        shown = coordinates + float(movement) * (adjusted - coordinates)
        median_shift = float(
            np.median(np.linalg.norm(shown - coordinates, axis=1))
        )
        palette = plt.get_cmap("tab20", max(len(labels.categories), 1))
        point_colors = palette(np.clip(label_codes, 0, max(len(labels.categories) - 1, 0)))
        point_colors[label_codes < 0] = (0.5, 0.5, 0.5, 1.0)
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(
            shown[:, 0],
            shown[:, 1],
            c=point_colors,
            s=point_size,
            alpha=0.65,
            linewidths=0,
        )
        ax.set_title(
            f"Label-guided embedding: k={k}, {iterations} iterations, "
            f"balance={balance:.2f}, same-label rest distance="
            f"{distance_fraction:.2f}× original, movement={movement:.2f}, "
            f"median shift={median_shift:.3g}"
        )
        ax.set_xlabel("IMSTE 1")
        ax.set_ylabel("IMSTE 2")
        ax.set_aspect("equal", adjustable="datalim")
        legend_handles = [
            Patch(facecolor=palette(index), edgecolor="none", label=str(label))
            for index, label in enumerate(labels.categories)
        ]
        if np.any(label_codes < 0):
            legend_handles.append(
                Patch(facecolor="0.5", edgecolor="none", label="(missing)")
            )
        if legend_handles:
            ax.legend(
                handles=legend_handles,
                title=label_key,
                bbox_to_anchor=(0.5, -0.12),
                loc="upper center",
                ncol=min(6, len(legend_handles)),
                frameon=False,
            )
        fig.tight_layout()
        display(fig)
        plt.close(fig)

    def update(
        k: int,
        iterations: int,
        balance: float,
        distance_fraction: float,
        movement: float,
    ):
        status.value = (
            '<span style="color:#9a6700;font-weight:600">'
            '⏳ Computing…</span>'
        )
        try:
            render_update(k, iterations, balance, distance_fraction, movement)
        except Exception as error:
            status.value = (
                '<span style="color:#b42318;font-weight:600">'
                f'Failed: {html_escape(str(error))}</span>'
            )
            raise
        status.value = '<span style="color:#188038;font-weight:600">✓ Done</span>'

    output = widgets.interactive_output(
        update,
        {
            "k": k_control,
            "iterations": iteration_control,
            "balance": balance_control,
            "distance_fraction": distance_control,
            "movement": movement_control,
        },
    )
    label_width = "420px"
    controls = widgets.VBox(
        [
            widgets.HBox(
                [
                    widgets.HTML(
                        value=(
                            '<div style="text-align:right;padding-right:12px;'
                            f'white-space:nowrap">{html_escape(label)}</div>'
                        ),
                        layout=widgets.Layout(width=label_width),
                    ),
                    control,
                ],
                layout=widgets.Layout(align_items="center"),
            )
            for label, control in (
                ("k per group:", k_control),
                ("Iterations:", iteration_control),
                ("Balance (repel → same-label spring):", balance_control),
                ("Rest distance (fraction of original):", distance_control),
                ("Movement:", movement_control),
            )
        ] + [status],
        layout=widgets.Layout(align_items="flex-start"),
    )
    viewer = widgets.VBox([controls, output])
    return viewer


def plot_cell_counts(
    obs: pd.DataFrame,
    label_key: str,
    *,
    log_scale: bool = True,
    figsize: tuple[float, float] = (10, 4),
):
    """Plot the number of observations per label as a vertical bar chart."""
    if label_key not in obs:
        raise KeyError(f"Label column {label_key!r} is not present in obs.")
    counts = obs[label_key].value_counts(dropna=False).sort_values(ascending=False)
    counts = counts[counts > 0]
    tick_labels = [
        "(missing)" if pd.isna(value) else str(value) for value in counts.index
    ]
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(np.arange(len(counts)), counts.to_numpy())
    ax.bar_label(bars, labels=[f"{count:,}" for count in counts], padding=3)
    ax.set_xticks(np.arange(len(counts)), tick_labels, rotation=90)
    ax.set_ylabel("Number of cells" + (" (log scale)" if log_scale else ""))
    ax.set_xlabel(label_key)
    ax.set_title(f"Cell counts per {label_key}")
    if log_scale:
        ax.set_yscale("log")
        lower, upper = ax.get_ylim()
        ax.set_ylim(lower, upper * 1.2)
    fig.tight_layout()
    return fig, ax



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
    original_codes = np.asarray(labels.codes)
    original_classes = np.asarray(labels.categories)
    if len(original_classes) == 0:
        raise ValueError("At least one labeled class is required for CV.")

    original_counts = np.bincount(
        original_codes[original_codes >= 0], minlength=len(original_classes)
    )
    eligible_classes = np.flatnonzero(original_counts >= 2)
    if len(eligible_classes) < 2:
        raise ValueError("At least two classes need two or more instances for CV.")
    class_map = np.full(len(original_classes), -1, dtype=np.intp)
    class_map[eligible_classes] = np.arange(len(eligible_classes))
    valid = (original_codes >= 0) & (class_map[np.maximum(original_codes, 0)] >= 0)
    X = coordinates[valid]
    y = class_map[original_codes[valid]]
    classes = original_classes[eligible_classes]
    counts = np.bincount(y, minlength=len(classes))
    min_class_count = int(counts.min())
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
        n_jobs=1,
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
        "n_excluded": int(len(labels) - valid.sum()),
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
    """Build ranked Euclidean MSTs directly on 2D coordinates.

    Fresh edges are preferred. If they cannot connect a residual component,
    reuse the shortest edge crossing that cut and increment its rank by one.
    """
    n_samples = len(embedding)
    if n_samples < 2:
        raise ValueError("At least two embedding points are needed to build an MST.")

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
    edge_last_rank: dict[tuple[int, int], int] = {}

    for rank in range(1, n_trees + 1):
        in_tree = np.zeros(n_samples, dtype=bool)
        best = np.full(n_samples, np.inf, dtype=np.float64)
        parent = np.full(n_samples, -1, dtype=np.intp)
        parent_reused = np.zeros(n_samples, dtype=bool)
        best[0] = 0.0

        for _ in range(n_samples):
            node = int(np.argmin(np.where(in_tree, np.inf, best)))
            if not np.isfinite(best[node]):
                # No fresh edge crosses this cut. Reuse the shortest geometric
                # edge and let its increasing rank reduce its plotting opacity.
                left_nodes = np.flatnonzero(in_tree)
                right_nodes = np.flatnonzero(~in_tree)
                minimum = np.inf
                other = node = -1
                for left in left_nodes:
                    delta = embedding[right_nodes] - embedding[left]
                    squared = np.einsum("ij,ij->i", delta, delta)
                    offset = int(np.argmin(squared))
                    if squared[offset] < minimum:
                        minimum = float(squared[offset])
                        other = int(left)
                        node = int(right_nodes[offset])
                best[node] = minimum
                parent[node] = other
                parent_reused[node] = True
            in_tree[node] = True

            if parent[node] != -1:
                other = int(parent[node])
                edge = (min(other, node), max(other, node))
                edge_rank = (
                    edge_last_rank[edge] + 1 if parent_reused[node] else rank
                )
                edge_last_rank[edge] = edge_rank
                edges.append((other, node))
                ranks.append(edge_rank)
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
            improve = connectable & (squared_distances <= best)
            best[improve] = squared_distances[improve]
            parent[improve] = node
            parent_reused[improve] = False

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


def _add_compound_path_union_outline(
    ax, path, color, *, linewidth: float = 1.0, alpha: float = 0.95
):
    """Draw only the exterior outline of a compound polygon path.

    Matplotlib strokes each subpath independently. Rasterizing the filled
    compound path and contouring its mask gives the outline of the union, with
    overlaps between neighborhood hulls left unoutlined.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    width, height = 1600, 1400
    xmin, ymin, xmax, ymax = path.get_extents().extents
    xpad = max((xmax - xmin) * 0.01, np.finfo(float).eps)
    ypad = max((ymax - ymin) * 0.01, np.finfo(float).eps)
    xmin, xmax = xmin - xpad, xmax + xpad
    ymin, ymax = ymin - ypad, ymax + ypad

    raster_figure = Figure(figsize=(8, 7), dpi=200)
    raster_canvas = FigureCanvasAgg(raster_figure)
    raster_ax = raster_figure.add_axes((0, 0, 1, 1))
    raster_ax.set_xlim(xmin, xmax)
    raster_ax.set_ylim(ymin, ymax)
    raster_ax.set_axis_off()
    raster_ax.add_patch(
        PathPatch(path, facecolor="black", edgecolor="none", antialiased=True)
    )
    raster_canvas.draw()
    pixels = np.asarray(raster_canvas.buffer_rgba())
    union_mask = (pixels[:, :, 0] < 128)[::-1, :]
    component_labels, n_components = ndimage.label(
        union_mask, structure=np.ones((3, 3), dtype=bool)
    )

    x_grid = np.linspace(xmin, xmax, width)
    y_grid = np.linspace(ymin, ymax, height)
    ax.contour(
        x_grid,
        y_grid,
        union_mask.astype(np.float32),
        levels=(0.5,),
        colors=[(*color[:3], alpha)],
        linewidths=linewidth,
    )
    return component_labels, n_components, x_grid, y_grid


def _fit_component_ellipses(
    points: np.ndarray,
    *,
    pixel_area: float,
    n_std: float,
    max_ellipses: int,
    min_precision: float,
    min_recall: float,
    random_state: int,
):
    """Fit one to ``max_ellipses`` ellipses, splitting when one overcovers.

    Precision estimates the fraction of the total fitted ellipse area occupied
    by the component. Recall estimates the fraction of component pixels inside
    their assigned ellipse. The smallest split meeting both thresholds wins;
    otherwise the best precision/recall balance is used.
    """
    sample_size = min(len(points), 10_000)
    sample_indices = np.linspace(0, len(points) - 1, sample_size, dtype=int)
    sample = points[sample_indices]
    candidates = []

    for n_ellipses in range(1, min(max_ellipses, len(sample) // 3) + 1):
        if n_ellipses == 1:
            sample_labels = np.zeros(len(sample), dtype=np.intp)
            centers = sample.mean(axis=0, keepdims=True)
        else:
            model = KMeans(
                n_clusters=n_ellipses,
                n_init=10,
                random_state=random_state,
            ).fit(sample)
            sample_labels = model.labels_
            centers = model.cluster_centers_

        ellipses = []
        enclosed_count = 0
        total_ellipse_area = 0.0
        valid = True
        for cluster_id, center in enumerate(centers):
            cluster = sample[sample_labels == cluster_id]
            if len(cluster) < 3:
                valid = False
                break
            covariance = np.cov(cluster, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = np.maximum(eigenvalues[order], 0.0)
            if eigenvalues[1] <= np.finfo(float).eps:
                valid = False
                break
            eigenvectors = eigenvectors[:, order]
            inverse_covariance = (
                eigenvectors / eigenvalues
            ) @ eigenvectors.T
            delta = cluster - center
            mahalanobis_squared = np.einsum(
                "ij,jk,ik->i", delta, inverse_covariance, delta
            )
            enclosed_count += int((mahalanobis_squared <= n_std**2).sum())
            total_ellipse_area += (
                np.pi * n_std**2 * np.sqrt(eigenvalues[0] * eigenvalues[1])
            )
            angle = np.degrees(
                np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
            )
            ellipses.append(
                (
                    center,
                    2 * n_std * np.sqrt(eigenvalues[0]),
                    2 * n_std * np.sqrt(eigenvalues[1]),
                    angle,
                )
            )
        if not valid:
            continue

        precision = min(1.0, (len(points) * pixel_area) / total_ellipse_area)
        recall = enclosed_count / len(sample)
        score = 2 * precision * recall / max(precision + recall, 1e-12)
        candidates.append((n_ellipses, precision, recall, score, ellipses))
        if precision >= min_precision and recall >= min_recall:
            return ellipses, n_ellipses, precision, recall

    if not candidates:
        return [], 0, 0.0, 0.0
    best = max(candidates, key=lambda candidate: candidate[3] - 0.02 * (candidate[0] - 1))
    return best[4], best[0], best[1], best[2]


def plot_class_neighborhood_hulls(
    embedding: np.ndarray,
    obs: pd.DataFrame,
    *,
    k: int = 5,
    color_key: str = "celltype",
    cmap: str = "tab20",
    alpha_range: tuple[float, float] = (0.2, 0.9),
    cv_neighbors: int = 5,
    cv_splits: int = 5,
    cv_random_state: int = 42,
    distance_std_threshold: float | None = 1.0,
    ax=None,
    ellipse_ax=None,
    ellipse_overlay_ax=None,
    ellipse_n_std: float = 2.0,
    ellipse_max_per_patch: int = 3,
    ellipse_min_precision: float = 0.65,
    ellipse_min_recall: float = 0.8,
    legend_position: str | None = None,
):
    """Plot the union of same-class local convex hulls in the 2D embedding.

    For each point, its neighborhood contains the point itself and its ``k``
    nearest other points from the same class. The convex hulls of those
    neighborhoods are combined into one filled path per class, so overlaps
    within a class receive that class's alpha only once. A stratified
    cross-validated kNN classifier runs on the 2D coordinates first; incorrectly
    classified and CV-ineligible points are omitted. Then, by default, points
    whose mean distance to their same-class ``k`` nearest neighbors exceeds
    that class's mean plus one standard deviation are also removed. Set
    ``distance_std_threshold=None`` to disable the distance filter. Class patch
    fill alpha is scaled linearly by the number of retained cells between
    ``alpha_range[0]`` and ``alpha_range[1]``.
    Neighborhoods without three non-collinear coordinates are skipped.
    Ellipses are split into up to ``ellipse_max_per_patch`` clusters when the
    estimated area precision or pixel coverage falls below its threshold.
    """
    if (
        not isinstance(k, (int, np.integer))
        or isinstance(k, bool)
        or k < 1
    ):
        raise ValueError("k must be a positive integer.")
    if (
        len(alpha_range) != 2
        or not np.isfinite(alpha_range).all()
        or not 0 <= alpha_range[0] <= alpha_range[1] <= 1
    ):
        raise ValueError("alpha_range must be a (min, max) pair in [0, 1].")
    if distance_std_threshold is not None and (
        not np.isfinite(distance_std_threshold) or distance_std_threshold < 0
    ):
        raise ValueError("distance_std_threshold must be non-negative or None.")
    if not np.isfinite(ellipse_n_std) or ellipse_n_std <= 0:
        raise ValueError("ellipse_n_std must be positive and finite.")
    if (
        not isinstance(ellipse_max_per_patch, (int, np.integer))
        or isinstance(ellipse_max_per_patch, bool)
        or ellipse_max_per_patch < 1
    ):
        raise ValueError("ellipse_max_per_patch must be a positive integer.")
    if not 0 <= ellipse_min_precision <= 1 or not 0 <= ellipse_min_recall <= 1:
        raise ValueError("ellipse precision and recall thresholds must be in [0, 1].")
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
    cv_results = cross_validate_embedding_knn(
        coordinates,
        obs,
        label_key=color_key,
        n_neighbors=cv_neighbors,
        n_splits=cv_splits,
        random_state=cv_random_state,
    )
    cv_correct_mask = cv_results["correct_mask"]
    category_colors = plt.get_cmap(cmap, max(len(labels.categories), 1))
    created_figure = ax is None
    if created_figure:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure
    ellipse_axes = [
        ellipse_axis
        for ellipse_axis in (ellipse_ax, ellipse_overlay_ax)
        if ellipse_axis is not None
    ]
    if any(ellipse_axis.figure is not fig for ellipse_axis in ellipse_axes):
        raise ValueError("ax and ellipse axes must belong to the same figure.")
    if legend_position is None:
        legend_position = "right" if created_figure else "inside"
    if legend_position not in {"inside", "right", "bottom"}:
        raise ValueError("legend_position must be 'inside', 'right', or 'bottom'.")
    class_patches = []
    filtered_plot_mask = np.zeros(len(coordinates), dtype=bool)
    filtered_outliers = 0

    for category_index, category in enumerate(labels.categories):
        class_indices = np.flatnonzero(
            (labels.codes == category_index) & cv_correct_mask
        )
        if len(class_indices) < 2:
            continue

        class_coordinates = coordinates[class_indices]
        n_same_class_neighbors = min(k, len(class_indices) - 1)
        distance_search = NearestNeighbors(n_neighbors=n_same_class_neighbors + 1)
        distance_search.fit(class_coordinates)
        neighbor_distances, neighbor_indices = distance_search.kneighbors(
            class_coordinates, return_distance=True
        )
        mean_neighbor_distances = np.empty(len(class_indices), dtype=np.float64)
        for local_index, (distances, neighbors) in enumerate(
            zip(neighbor_distances, neighbor_indices)
        ):
            other_mask = neighbors != local_index
            mean_neighbor_distances[local_index] = distances[other_mask][
                :n_same_class_neighbors
            ].mean()

        if distance_std_threshold is None:
            inlier_mask = np.ones(len(class_indices), dtype=bool)
        else:
            cutoff = mean_neighbor_distances.mean() + (
                distance_std_threshold * mean_neighbor_distances.std()
            )
            inlier_mask = mean_neighbor_distances <= cutoff
        filtered_outliers += int((~inlier_mask).sum())
        class_indices = class_indices[inlier_mask]
        class_coordinates = class_coordinates[inlier_mask]
        if len(class_indices) < 3:
            continue
        filtered_plot_mask[class_indices] = True

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
            class_patches.append(
                (category_index, category, color, compound_path, len(class_indices))
            )

    legend_handles = []
    ellipse_legend_handles = []
    n_ellipses = 0
    n_split_patches = 0
    split_patches_by_class = {}
    class_counts = [record[4] for record in class_patches]
    if class_counts:
        min_count, max_count = min(class_counts), max(class_counts)
        for _, category, color, compound_path, n_cells in class_patches:
            if min_count == max_count:
                patch_alpha = (alpha_range[0] + alpha_range[1]) / 2
            else:
                patch_alpha = alpha_range[0] + (
                    (n_cells - min_count) / (max_count - min_count)
                ) * (alpha_range[1] - alpha_range[0])
            ax.add_patch(
                PathPatch(
                    compound_path,
                    facecolor=(*color[:3], patch_alpha),
                    edgecolor="none",
                )
            )
            border_alpha = min(patch_alpha + 0.1, 0.95)
            component_labels, n_components, x_grid, y_grid = _add_compound_path_union_outline(
                ax, compound_path, color, linewidth=1.0, alpha=border_alpha
            )
            legend_handles.append(
                Patch(
                    facecolor=(*color[:3], patch_alpha),
                    edgecolor=(*color[:3], border_alpha),
                    linewidth=1.0,
                    label=str(category),
                )
            )
            if ellipse_axes:
                class_drew_ellipse = False
                for component_id in range(1, n_components + 1):
                    rows, columns = np.nonzero(component_labels == component_id)
                    if len(rows) < 3:
                        continue
                    component_points = np.column_stack(
                        (x_grid[columns], y_grid[rows])
                    )
                    pixel_area = (
                        (x_grid[1] - x_grid[0]) * (y_grid[1] - y_grid[0])
                    )
                    fitted, split_count, _, _ = _fit_component_ellipses(
                        component_points,
                        pixel_area=pixel_area,
                        n_std=ellipse_n_std,
                        max_ellipses=ellipse_max_per_patch,
                        min_precision=ellipse_min_precision,
                        min_recall=ellipse_min_recall,
                        random_state=cv_random_state,
                    )
                    if split_count > 1:
                        n_split_patches += 1
                        split_patches_by_class[category] = (
                            split_patches_by_class.get(category, 0) + 1
                        )
                    if split_count > 1:
                        ellipse_paths = []
                        for center, width, height, angle in fitted:
                            ellipse = Ellipse(
                                center, width=width, height=height, angle=angle
                            )
                            ellipse_paths.append(
                                ellipse.get_path().transformed(ellipse.get_transform())
                            )
                        ellipse_union_path = MplPath.make_compound_path(*ellipse_paths)
                        for target_ax in ellipse_axes:
                            target_ax.add_patch(
                                PathPatch(
                                    ellipse_union_path,
                                    facecolor=(*color[:3], 0.12),
                                    edgecolor="none",
                                    zorder=1,
                                )
                            )
                            _add_compound_path_union_outline(
                                target_ax,
                                ellipse_union_path,
                                color,
                                linewidth=1.2,
                                alpha=0.9,
                            )
                    else:
                        for center, width, height, angle in fitted:
                            for target_ax in ellipse_axes:
                                target_ax.add_patch(
                                    Ellipse(
                                        center,
                                        width=width,
                                        height=height,
                                        angle=angle,
                                        facecolor=(*color[:3], 0.12),
                                        edgecolor=(*color[:3], 0.9),
                                        linewidth=1.2,
                                        zorder=1,
                                    )
                                )
                    if fitted:
                        class_drew_ellipse = True
                    n_ellipses += len(fitted)
                if class_drew_ellipse:
                    ellipse_legend_handles.append(
                        Patch(
                            facecolor=(*color[:3], 0.12),
                            edgecolor=(*color[:3], 0.9),
                            linewidth=1.2,
                            label=str(category),
                        )
                    )

    if filtered_plot_mask.any():
        ax.update_datalim(coordinates[filtered_plot_mask])
    ax.autoscale_view()
    ax.margins(0.03)
    ax.set_aspect("equal", adjustable="datalim")
    if distance_std_threshold is None:
        filter_description = "distance filter off"
    else:
        filter_description = (
            f"{filtered_outliers} cells filtered at class mean + "
            f"{distance_std_threshold:g} SD"
        )
    ax.set_title(
        f"Unions of {k}-nearest-neighbor hulls by {color_key}\n"
        f"{filter_description}; 2D CV accuracy "
        f"{cv_results['accuracy']:.3f} (balanced "
        f"{cv_results['balanced_accuracy']:.3f})"
    )
    ax.set_xlabel("Embedding 1")
    ax.set_ylabel("Embedding 2")
    if legend_handles:
        if legend_position == "right":
            legend_options = {
                "bbox_to_anchor": (1.02, 1), "loc": "upper left", "frameon": False
            }
        elif legend_position == "bottom":
            legend_options = {
                "bbox_to_anchor": (0.5, -0.16),
                "loc": "upper center",
                "ncol": min(5, len(legend_handles)),
                "frameon": False,
            }
        else:
            legend_options = {
                "loc": "upper right", "frameon": True, "framealpha": 0.8
            }
        ax.legend(
            handles=legend_handles,
            title=color_key,
            **legend_options,
        )
    if ellipse_axes:
        split_summary = ", ".join(
            f"{category}: {count}"
            for category, count in split_patches_by_class.items()
        ) or "none"
        for target_ax in ellipse_axes:
            target_ax.set_title(
                f"{ellipse_n_std:g}σ ellipses: {n_ellipses} total, "
                f"{n_split_patches} patches split; classes: {split_summary}"
            )
            target_ax.set_xlabel("Embedding 1")
            target_ax.set_ylabel("Embedding 2")
            target_ax.set_aspect("equal", adjustable="datalim")
        if ellipse_ax is not None and ellipse_legend_handles:
            if legend_position == "bottom":
                ellipse_legend_options = {
                    "bbox_to_anchor": (0.5, -0.16),
                    "loc": "upper center",
                    "ncol": min(5, len(ellipse_legend_handles)),
                    "frameon": False,
                }
            elif legend_position == "right":
                ellipse_legend_options = {
                    "bbox_to_anchor": (1.02, 1),
                    "loc": "upper left",
                    "frameon": False,
                }
            else:
                ellipse_legend_options = {
                    "loc": "upper right",
                    "frameon": True,
                    "framealpha": 0.8,
                }
            ellipse_ax.legend(
                handles=ellipse_legend_handles,
                title=color_key,
                **ellipse_legend_options,
            )
    if created_figure:
        fig.tight_layout()
    return fig, ax

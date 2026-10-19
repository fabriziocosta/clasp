from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


class LabelNeighborGraph:
    """Build and plot directed label transitions from vector data.

    ``fit(X, labels)`` accepts dense or sparse sample-by-feature data of any
    dimensionality. For each source label A, the A→B edge weight is the mean
    fraction of each A sample's ``k`` nearest neighbors that have label B.
    Edges below ``min_edge_fraction`` are omitted from ``graph_``.
    """

    def __init__(
        self,
        k: int = 30,
        *,
        metric: str = "euclidean",
        min_edge_fraction: float = 0.0,
        min_node_size: float = 1000.0,
        max_node_size: float = 2500.0,
        max_edge_width: float = 8.0,
    ) -> None:
        self.k = k
        self.metric = metric
        self.min_edge_fraction = float(min_edge_fraction)
        self.min_node_size = float(min_node_size)
        self.max_node_size = float(max_node_size)
        self.max_edge_width = max_edge_width
        if (
            not np.isfinite(self.min_edge_fraction)
            or self.min_edge_fraction < 0.0
            or self.min_edge_fraction > 1.0
        ):
            raise ValueError("min_edge_fraction must be between 0 and 1")
        if (
            not np.isfinite(self.min_node_size)
            or not np.isfinite(self.max_node_size)
            or self.min_node_size <= 0.0
            or self.max_node_size < self.min_node_size
        ):
            raise ValueError("node sizes must be finite, positive, and max_node_size >= min_node_size")

    def fit(self, X, labels) -> "LabelNeighborGraph":
        """Construct the label-level directed graph from vector data ``X``."""
        if sparse.issparse(X):
            vectors = sparse.csr_matrix(X, dtype=float, copy=True)
            finite = np.isfinite(vectors.data).all()
        else:
            vectors = np.asarray(X, dtype=float)
            finite = np.isfinite(vectors).all()
        if vectors.ndim != 2 or vectors.shape[1] < 1:
            raise ValueError("X must have shape (n_samples, n_features) with n_features >= 1")
        if not finite:
            raise ValueError("X must contain only finite values")

        label_values = (
            pd.Series(labels, dtype="object")
            .astype("string")
            .fillna("Missing")
            .to_numpy(dtype=str)
        )
        if len(label_values) != vectors.shape[0]:
            raise ValueError("labels must have one value for each row in X")
        if vectors.shape[0] < 2:
            raise ValueError("at least two samples are required to build a k-NN graph")
        if isinstance(self.k, (bool, np.bool_)) or not isinstance(self.k, (int, np.integer)) or self.k < 1:
            raise ValueError("k must be a positive integer")

        n_neighbors = min(int(self.k), vectors.shape[0] - 1)
        label_names = list(dict.fromkeys(label_values.tolist()))
        label_to_index = {label: index for index, label in enumerate(label_names)}
        encoded_labels = np.fromiter(
            (label_to_index[label] for label in label_values),
            dtype=int,
            count=len(label_values),
        )
        label_counts = np.bincount(encoded_labels, minlength=len(label_names))
        fractions = np.zeros((len(label_names), len(label_names)), dtype=float)

        neighbor_model = NearestNeighbors(
            n_neighbors=n_neighbors + 1,
            metric=self.metric,
        ).fit(vectors)
        neighbor_indices = neighbor_model.kneighbors(vectors, return_distance=False)
        for sample_index, candidates in enumerate(neighbor_indices):
            neighbors = candidates[candidates != sample_index][:n_neighbors]
            if len(neighbors) != n_neighbors:
                raise RuntimeError("Could not find the requested number of non-self neighbors")
            target_counts = np.bincount(
                encoded_labels[neighbors],
                minlength=len(label_names),
            )
            fractions[encoded_labels[sample_index]] += target_counts / n_neighbors
        fractions /= label_counts[:, None]

        graph = nx.DiGraph()
        for label, count in zip(label_names, label_counts):
            graph.add_node(label, count=int(count))
        for source_index, source in enumerate(label_names):
            for target_index, target in enumerate(label_names):
                fraction = float(fractions[source_index, target_index])
                if fraction >= self.min_edge_fraction and fraction > 0:
                    graph.add_edge(source, target, weight=fraction)

        self.graph_ = graph
        self.transition_matrix_ = fractions
        self.label_names_ = tuple(label_names)
        self.labels_ = label_values.copy()
        self.label_counts_ = dict(zip(label_names, label_counts.astype(int).tolist()))
        self.n_samples_fit_ = vectors.shape[0]
        self.n_features_in_ = vectors.shape[1]
        self.n_neighbors_ = n_neighbors
        self.positions_ = None
        return self

    def set_positions(self, positions: Mapping[str, Sequence[float]]) -> "LabelNeighborGraph":
        """Set explicit 2D positions for the fitted label nodes."""
        if not hasattr(self, "graph_"):
            raise ValueError("Call fit(X, labels) before setting positions.")
        normalized = {str(label): np.asarray(position, dtype=float) for label, position in positions.items()}
        missing = set(self.label_names_) - set(normalized)
        extra = set(normalized) - set(self.label_names_)
        if missing or extra:
            raise ValueError(
                f"Position labels must match fitted graph nodes; missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        for label, position in normalized.items():
            if position.shape != (2,) or not np.isfinite(position).all():
                raise ValueError(f"Position for {label!r} must be a finite 2D coordinate")
        self.positions_ = normalized
        return self

    def set_positions_from_embedding(
        self,
        embedding_2d,
        labels=None,
    ) -> "LabelNeighborGraph":
        """Place each label at its mean coordinate in a 2D sample embedding."""
        if not hasattr(self, "graph_"):
            raise ValueError("Call fit(X, labels) before setting positions.")
        coordinates = np.asarray(embedding_2d, dtype=float)
        if coordinates.shape != (self.n_samples_fit_, 2) or not np.isfinite(coordinates).all():
            raise ValueError(
                f"embedding_2d must be finite with shape ({self.n_samples_fit_}, 2)"
            )
        label_values = self.labels_ if labels is None else (
            pd.Series(labels, dtype="object")
            .astype("string")
            .fillna("Missing")
            .to_numpy(dtype=str)
        )
        if len(label_values) != self.n_samples_fit_:
            raise ValueError("labels must have one value for each fitted sample")
        if set(label_values.tolist()) != set(self.label_names_):
            raise ValueError("Position labels must match the labels used to fit the graph")
        positions = {
            label: coordinates[label_values == label].mean(axis=0)
            for label in self.label_names_
        }
        return self.set_positions(positions)

    def export_positions(self, path: str | Path) -> Path:
        """Save current node positions as a JSON file."""
        if self.positions_ is None:
            raise ValueError("Set or compute positions before exporting them.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "clasp-label-neighbor-positions-v1",
            "positions": {
                label: np.asarray(position, dtype=float).tolist()
                for label, position in self.positions_.items()
            },
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path

    def import_positions(self, path: str | Path) -> "LabelNeighborGraph":
        """Load node positions from a JSON file exported by this class."""
        payload = json.loads(Path(path).read_text())
        if not isinstance(payload, dict) or payload.get("format") != "clasp-label-neighbor-positions-v1":
            raise ValueError("Unsupported label-position file format")
        positions = payload.get("positions")
        if not isinstance(positions, dict):
            raise ValueError("Label-position file must contain a positions mapping")
        return self.set_positions(positions)

    def plot(
        self,
        *,
        ax: Axes | None = None,
        title: str | None = None,
        show_self_links: bool = False,
        show_node_counts: bool = False,
        node_alpha: float = 0.7,
        label_colors: Mapping[str, object] | None = None,
    ) -> Axes:
        """Draw the fitted graph using a Kamada–Kawai layout.

        Set ``show_self_links=True`` to draw within-label neighbor fractions.
        Self-transition weights remain available in ``graph_`` either way.
        """
        if not hasattr(self, "graph_"):
            raise ValueError("Call fit(X, labels) before plot().")
        if not np.isfinite(node_alpha) or not 0.0 <= node_alpha <= 1.0:
            raise ValueError("node_alpha must be between 0 and 1")

        label_names = list(self.label_names_)
        graph = self.graph_
        retained_fractions = np.zeros_like(self.transition_matrix_)
        label_to_index = {label: index for index, label in enumerate(label_names)}
        for source, target, data in graph.edges(data=True):
            retained_fractions[label_to_index[source], label_to_index[target]] = data["weight"]
        layout_graph = nx.Graph()
        layout_graph.add_nodes_from(label_names)
        if self.positions_ is not None:
            positions = self.positions_
        elif len(label_names) == 1:
            positions = {label_names[0]: np.zeros(2)}
        else:
            # Use the strongest directed transition as the undirected spring
            # strength for layout; combining both directions would inflate it.
            pair_strengths = np.maximum(retained_fractions, retained_fractions.T)
            positive_strengths = pair_strengths[pair_strengths > 0]
            disconnected_distance = (
                1.25 / positive_strengths.min() if positive_strengths.size else 1.0
            )
            for source_index, source in enumerate(label_names):
                for target_index in range(source_index + 1, len(label_names)):
                    target = label_names[target_index]
                    strength = float(pair_strengths[source_index, target_index])
                    length = 1.0 / strength if strength > 0 else disconnected_distance
                    layout_graph.add_edge(source, target, length=length)
            positions = nx.kamada_kawai_layout(layout_graph, weight="length")
        self.positions_ = positions

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 7))
        min_count = min(self.label_counts_.values())
        max_count = max(self.label_counts_.values())
        if max_count == min_count:
            node_sizes = [self.max_node_size] * len(label_names)
        else:
            count_span = max_count - min_count
            size_span = self.max_node_size - self.min_node_size
            node_sizes = [
                self.min_node_size
                + size_span * (self.label_counts_[label] - min_count) / count_span
                for label in label_names
            ]
        if label_colors is None:
            cmap = plt.get_cmap("tab20")
            node_colors = [cmap(index % cmap.N) for index in range(len(label_names))]
        else:
            missing_colors = set(label_names) - set(label_colors)
            if missing_colors:
                raise ValueError(
                    f"label_colors is missing graph labels: {sorted(missing_colors)}"
                )
            node_colors = [label_colors[label] for label in label_names]
        edges = [
            (source, target, data)
            for source, target, data in graph.edges(data=True)
            if show_self_links or source != target
        ]
        if edges:
            nx.draw_networkx_edges(
                graph,
                positions,
                edgelist=[(source, target) for source, target, _ in edges],
                width=[self.max_edge_width * data["weight"] for _, _, data in edges],
                edge_color="black",
                alpha=1.0,
                arrows=True,
                arrowstyle="-|>",
                arrowsize=13,
                connectionstyle="arc3,rad=0.12",
                node_size=node_sizes,
                ax=ax,
            )
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=label_names,
            node_size=node_sizes,
            node_color=node_colors,
            edgecolors="black",
            linewidths=0.8,
            alpha=node_alpha,
            ax=ax,
        )
        node_labels = {
            label: (
                f"{label}\n(n={self.label_counts_[label]:,})"
                if show_node_counts
                else label
            )
            for label in label_names
        }
        nx.draw_networkx_labels(
            graph,
            positions,
            labels=node_labels,
            font_size=9,
            ax=ax,
        )
        ax.set_title(title or f"Label transitions (k={self.n_neighbors_})")
        ax.set_axis_off()
        return ax


def plot_label_neighbor_graph(
    X,
    labels,
    *,
    k: int = 30,
    min_edge_fraction: float = 0.0,
    min_node_size: float = 1000.0,
    max_node_size: float = 2500.0,
    ax: Axes | None = None,
    title: str | None = None,
    show_self_links: bool = False,
    show_node_counts: bool = False,
    node_alpha: float = 0.7,
) -> nx.DiGraph:
    """Convenience wrapper to fit and plot a :class:`LabelNeighborGraph`."""
    model = LabelNeighborGraph(
        k=k,
        min_edge_fraction=min_edge_fraction,
        min_node_size=min_node_size,
        max_node_size=max_node_size,
    ).fit(X, labels)
    model.plot(
        ax=ax,
        title=title,
        show_self_links=show_self_links,
        show_node_counts=show_node_counts,
        node_alpha=node_alpha,
    )
    return model.graph_

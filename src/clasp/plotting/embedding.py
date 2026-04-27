from __future__ import annotations

from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _embedding_frame(
    adata: ad.AnnData,
    *,
    embedding_key: str,
    color_key: str,
    order: np.ndarray | None = None,
) -> pd.DataFrame:
    if embedding_key not in adata.obsm:
        raise KeyError(f"Missing embedding in obsm: {embedding_key!r}")
    if color_key not in adata.obs:
        raise KeyError(f"Missing obs column: {color_key!r}")

    X = adata.obsm[embedding_key]
    if X.shape[1] < 2:
        raise ValueError("Embedding must have at least two dimensions.")

    if order is None:
        order = np.arange(adata.n_obs)

    return pd.DataFrame(
        {
            "x": X[order, 0],
            "y": X[order, 1],
            color_key: adata.obs[color_key].astype(str).to_numpy()[order],
        }
    )


def _plot_order(n_obs: int, *, shuffle: bool, random_state: int | None) -> np.ndarray:
    order = np.arange(n_obs)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(order)
    return order


def _save_figure(
    fig,
    filename: str | Path | None,
    *,
    dpi: int,
    savefig_kwargs: dict | None,
) -> Path | None:
    if filename is None:
        return None

    path = Path(filename)
    if not path.suffix:
        path = path.with_suffix(".png")
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "dpi": dpi,
        "bbox_inches": "tight",
        "facecolor": "white",
    }
    if savefig_kwargs:
        kwargs.update(savefig_kwargs)
    fig.savefig(path, **kwargs)
    return path


def _plot_embedding_on_axis(
    adata: ad.AnnData,
    *,
    embedding_key: str,
    color_key: str,
    ax,
    title: str | None,
    s: float,
    alpha: float,
    legend: bool | str,
    palette=None,
    order: np.ndarray | None = None,
    legend_markerscale: float = 2.5,
):
    df = _embedding_frame(adata, embedding_key=embedding_key, color_key=color_key, order=order)
    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue=color_key,
        palette=palette,
        s=s,
        alpha=alpha,
        linewidth=0,
        ax=ax,
        legend=legend,
    )
    ax.set_title(title or f"{embedding_key} colored by {color_key}")
    ax.set_xlabel("1")
    ax.set_ylabel("2")
    ax.set_xticks([])
    ax.set_yticks([])
    if legend:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            frameon=False,
            title=color_key,
            markerscale=legend_markerscale,
            scatterpoints=1,
        )
    return ax


def plot_embedding(
    adata: ad.AnnData,
    *,
    embedding_key: str = "X_clasp",
    color_key: str = "batch",
    ax=None,
    filename: str | Path | None = None,
    save_dpi: int = 300,
    savefig_kwargs: dict | None = None,
    s: float = 12,
    alpha: float = 0.85,
    shuffle: bool = True,
    random_state: int | None = 0,
    legend_markerscale: float = 2.5,
):
    """Plot a 2D embedding stored in `adata.obsm` colored by an obs column.

    When `filename` is provided, the figure is also saved. Paths without a
    suffix are saved as high-resolution PNG files.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    plotted_ax = _plot_embedding_on_axis(
        adata,
        embedding_key=embedding_key,
        color_key=color_key,
        ax=ax,
        title=None,
        s=s,
        alpha=alpha,
        legend=True,
        palette=None,
        order=_plot_order(adata.n_obs, shuffle=shuffle, random_state=random_state),
        legend_markerscale=legend_markerscale,
    )
    _save_figure(plotted_ax.figure, filename, dpi=save_dpi, savefig_kwargs=savefig_kwargs)
    return plotted_ax


def plot_embedding_pair(
    adata: ad.AnnData,
    *,
    embedding_key: str = "X_clasp",
    batch_key: str = "batch",
    label_key: str = "label",
    axes=None,
    figsize: tuple[float, float] = (15, 5),
    filename: str | Path | None = None,
    save_dpi: int = 300,
    savefig_kwargs: dict | None = None,
    s: float = 12,
    alpha: float = 0.85,
    batch_palette: str | list | dict | None = "viridis",
    label_palette: str | list | dict | None = "tab20",
    shuffle: bool = True,
    random_state: int | None = 0,
    legend_markerscale: float = 2.5,
):
    """Plot one embedding twice: colored by batch and by biological label.

    The default palettes mirror the paper-style qualitative figures: viridis
    for ordered batch/time slices and a high-contrast categorical palette for
    cell-type labels. When `filename` is provided, the figure is also saved.
    Paths without a suffix are saved as high-resolution PNG files.
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    axes = np.asarray(axes).ravel()
    if len(axes) != 2:
        raise ValueError("axes must contain exactly two matplotlib axes.")

    order = _plot_order(adata.n_obs, shuffle=shuffle, random_state=random_state)
    _plot_embedding_on_axis(
        adata,
        embedding_key=embedding_key,
        color_key=batch_key,
        ax=axes[0],
        title=f"{embedding_key} by {batch_key}",
        s=s,
        alpha=alpha,
        legend=True,
        palette=batch_palette,
        order=order,
        legend_markerscale=legend_markerscale,
    )
    _plot_embedding_on_axis(
        adata,
        embedding_key=embedding_key,
        color_key=label_key,
        ax=axes[1],
        title=f"{embedding_key} by {label_key}",
        s=s,
        alpha=alpha,
        legend=True,
        palette=label_palette,
        order=order,
        legend_markerscale=legend_markerscale,
    )
    _save_figure(axes[0].figure, filename, dpi=save_dpi, savefig_kwargs=savefig_kwargs)
    return axes

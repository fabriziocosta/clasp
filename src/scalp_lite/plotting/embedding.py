from __future__ import annotations

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _embedding_frame(adata: ad.AnnData, *, embedding_key: str, color_key: str) -> pd.DataFrame:
    if embedding_key not in adata.obsm:
        raise KeyError(f"Missing embedding in obsm: {embedding_key!r}")
    if color_key not in adata.obs:
        raise KeyError(f"Missing obs column: {color_key!r}")

    X = adata.obsm[embedding_key]
    if X.shape[1] < 2:
        raise ValueError("Embedding must have at least two dimensions.")

    return pd.DataFrame({"x": X[:, 0], "y": X[:, 1], color_key: adata.obs[color_key].astype(str).to_numpy()})


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
):
    df = _embedding_frame(adata, embedding_key=embedding_key, color_key=color_key)
    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue=color_key,
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
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, title=color_key)
    return ax


def plot_embedding(
    adata: ad.AnnData,
    *,
    embedding_key: str = "X_scalp",
    color_key: str = "batch",
    ax=None,
    s: float = 12,
    alpha: float = 0.85,
):
    """Plot a 2D embedding stored in `adata.obsm` colored by an obs column."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    return _plot_embedding_on_axis(
        adata,
        embedding_key=embedding_key,
        color_key=color_key,
        ax=ax,
        title=None,
        s=s,
        alpha=alpha,
        legend=True,
    )


def plot_embedding_pair(
    adata: ad.AnnData,
    *,
    embedding_key: str = "X_scalp",
    batch_key: str = "batch",
    label_key: str = "label",
    axes=None,
    figsize: tuple[float, float] = (12, 5),
    s: float = 12,
    alpha: float = 0.85,
):
    """Plot one embedding twice: colored by batch and by biological label."""
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    axes = np.asarray(axes).ravel()
    if len(axes) != 2:
        raise ValueError("axes must contain exactly two matplotlib axes.")

    _plot_embedding_on_axis(
        adata,
        embedding_key=embedding_key,
        color_key=batch_key,
        ax=axes[0],
        title=f"{embedding_key} by {batch_key}",
        s=s,
        alpha=alpha,
        legend=True,
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
    )
    return axes

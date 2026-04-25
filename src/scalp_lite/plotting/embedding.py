from __future__ import annotations

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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
    if embedding_key not in adata.obsm:
        raise KeyError(f"Missing embedding in obsm: {embedding_key!r}")
    if color_key not in adata.obs:
        raise KeyError(f"Missing obs column: {color_key!r}")

    X = adata.obsm[embedding_key]
    if X.shape[1] < 2:
        raise ValueError("Embedding must have at least two dimensions.")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], color_key: adata.obs[color_key].astype(str).to_numpy()})
    sns.scatterplot(data=df, x="x", y="y", hue=color_key, s=s, alpha=alpha, linewidth=0, ax=ax)
    ax.set_title(f"{embedding_key} colored by {color_key}")
    ax.set_xlabel("1")
    ax.set_ylabel("2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    return ax

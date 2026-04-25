from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from scalp_lite.plotting import plot_embedding_pair


def test_plot_embedding_pair_returns_two_axes(toy_adata):
    toy_adata.obsm["X_scalp"] = toy_adata.X[:, :2]

    axes = plot_embedding_pair(toy_adata, embedding_key="X_scalp", batch_key="batch", label_key="label")

    assert len(axes) == 2
    assert axes[0].get_title() == "X_scalp by batch"
    assert axes[1].get_title() == "X_scalp by label"
    plt.close(axes[0].figure)


def test_plot_embedding_pair_rejects_missing_label(toy_adata):
    toy_adata.obsm["X_scalp"] = toy_adata.X[:, :2]

    with pytest.raises(KeyError, match="missing"):
        plot_embedding_pair(toy_adata, embedding_key="X_scalp", batch_key="batch", label_key="missing")

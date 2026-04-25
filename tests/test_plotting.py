from __future__ import annotations

import matplotlib
import numpy as np
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


def test_plot_embedding_pair_uses_distinct_default_palettes(toy_adata):
    toy_adata.obsm["X_scalp"] = toy_adata.X[:, :2]

    axes = plot_embedding_pair(toy_adata, embedding_key="X_scalp", batch_key="batch", label_key="label")

    batch_color = axes[0].collections[0].get_facecolors()[0]
    label_color = axes[1].collections[0].get_facecolors()[0]
    assert tuple(batch_color) != tuple(label_color)
    plt.close(axes[0].figure)


def test_plot_embedding_pair_shuffles_draw_order_in_sync(toy_adata):
    toy_adata.obsm["X_scalp"] = np.column_stack([np.arange(toy_adata.n_obs), np.zeros(toy_adata.n_obs)])

    axes = plot_embedding_pair(
        toy_adata,
        embedding_key="X_scalp",
        batch_key="batch",
        label_key="label",
        shuffle=True,
        random_state=7,
    )

    expected_order = np.arange(toy_adata.n_obs)
    np.random.default_rng(7).shuffle(expected_order)
    batch_offsets = axes[0].collections[0].get_offsets()
    label_offsets = axes[1].collections[0].get_offsets()
    np.testing.assert_array_equal(batch_offsets[:, 0], expected_order)
    np.testing.assert_array_equal(label_offsets[:, 0], expected_order)
    plt.close(axes[0].figure)


def test_plot_embedding_pair_rejects_missing_label(toy_adata):
    toy_adata.obsm["X_scalp"] = toy_adata.X[:, :2]

    with pytest.raises(KeyError, match="missing"):
        plot_embedding_pair(toy_adata, embedding_key="X_scalp", batch_key="batch", label_key="missing")

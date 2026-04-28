from __future__ import annotations

import matplotlib
import numpy as np
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from clasp.plotting import plot_embedding_pair


def test_plot_embedding_pair_returns_two_axes(toy_adata):
    toy_adata.obsm["X_clasp"] = toy_adata.X[:, :2]

    axes = plot_embedding_pair(toy_adata, embedding_key="X_clasp", batch_key="batch", label_key="label")

    assert len(axes) == 2
    assert axes[0].get_title() == "X_clasp by batch"
    assert axes[1].get_title() == "X_clasp by label"
    plt.close(axes[0].figure)


def test_plot_embedding_pair_uses_distinct_default_palettes(toy_adata):
    toy_adata.obsm["X_clasp"] = toy_adata.X[:, :2]

    axes = plot_embedding_pair(toy_adata, embedding_key="X_clasp", batch_key="batch", label_key="label")

    batch_color = axes[0].collections[0].get_facecolors()[0]
    label_color = axes[1].collections[0].get_facecolors()[0]
    assert tuple(batch_color) != tuple(label_color)
    plt.close(axes[0].figure)


def test_plot_embedding_pair_shuffles_draw_order_in_sync(toy_adata):
    toy_adata.obsm["X_clasp"] = np.column_stack([np.arange(toy_adata.n_obs), np.zeros(toy_adata.n_obs)])

    axes = plot_embedding_pair(
        toy_adata,
        embedding_key="X_clasp",
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


def test_plot_embedding_pair_uses_larger_legend_markers(toy_adata):
    toy_adata.obsm["X_clasp"] = toy_adata.X[:, :2]

    axes = plot_embedding_pair(
        toy_adata,
        embedding_key="X_clasp",
        batch_key="batch",
        label_key="label",
        legend_markerscale=3.0,
    )

    assert axes[0].get_legend().markerscale == 3.0
    assert axes[1].get_legend().markerscale == 3.0
    plt.close(axes[0].figure)


def test_plot_embedding_pair_saves_high_resolution_png(toy_adata, tmp_path):
    toy_adata.obsm["X_clasp"] = toy_adata.X[:, :2]
    output_path = tmp_path / "nested" / "embedding.png"

    axes = plot_embedding_pair(
        toy_adata,
        embedding_key="X_clasp",
        batch_key="batch",
        label_key="label",
        filename=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    plt.close(axes[0].figure)


def test_plot_embedding_pair_defaults_missing_suffix_to_png(toy_adata, tmp_path):
    toy_adata.obsm["X_clasp"] = toy_adata.X[:, :2]
    output_path = tmp_path / "embedding"

    axes = plot_embedding_pair(
        toy_adata,
        embedding_key="X_clasp",
        batch_key="batch",
        label_key="label",
        filename=output_path,
    )

    assert output_path.with_suffix(".png").exists()
    plt.close(axes[0].figure)


def test_plot_embedding_pair_uses_pale_blue_unlabeled_panel_when_label_missing(toy_adata):
    toy_adata.obsm["X_clasp"] = toy_adata.X[:, :2]

    axes = plot_embedding_pair(toy_adata, embedding_key="X_clasp", batch_key="batch", label_key="missing")

    assert axes[1].get_title() == "X_clasp by missing"
    assert axes[1].get_legend().get_title().get_text() == "missing"
    assert axes[1].get_legend().texts[0].get_text() == "unlabeled"
    label_color = axes[1].collections[0].get_facecolors()[0]
    np.testing.assert_allclose(label_color[:3], np.array([0xB7, 0xD7, 0xEE]) / 255, atol=0.02)
    plt.close(axes[0].figure)

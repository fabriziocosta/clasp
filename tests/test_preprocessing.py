from __future__ import annotations

from scalp_lite.preprocessing import ensure_pca, split_batches


def test_pca_is_created_when_missing(toy_adata):
    assert "X_pca" not in toy_adata.obsm
    ensure_pca(toy_adata, n_components=5)
    assert toy_adata.obsm["X_pca"].shape == (toy_adata.n_obs, 5)


def test_batch_splitting_preserves_original_cell_order(toy_adata):
    splits = split_batches(toy_adata, batch_key="batch")
    assert [split.batch for split in splits] == ["batch_a", "batch_b"]
    assert splits[0].indices.tolist() == list(range(16))
    assert splits[1].indices.tolist() == list(range(16, 32))

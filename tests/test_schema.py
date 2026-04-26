from __future__ import annotations

import pytest

from clasp.io import AnnDataValidationError, read_h5ad, validate_adata


def test_schema_validation_rejects_missing_batch_key(toy_adata):
    del toy_adata.obs["batch"]
    with pytest.raises(AnnDataValidationError, match="batch"):
        validate_adata(toy_adata, batch_key="batch")


def test_read_h5ad_rejects_missing_input(tmp_path):
    missing_path = tmp_path / "missing.h5ad"
    with pytest.raises(FileNotFoundError, match="CLASP_INPUT_H5AD"):
        read_h5ad(missing_path)

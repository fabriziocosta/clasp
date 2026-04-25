from __future__ import annotations

import anndata as ad


class AnnDataValidationError(ValueError):
    """Raised when an AnnData object does not satisfy scalp-lite's schema."""


def validate_adata(
    adata: ad.AnnData,
    *,
    batch_key: str = "batch",
    label_key: str | None = None,
    rep_key: str | None = None,
    require_rep: bool = False,
    min_batches: int = 2,
) -> None:
    """Validate the AnnData fields required by the SCALP-lite workflow."""
    if adata.n_obs == 0:
        raise AnnDataValidationError("AnnData must contain at least one observation.")
    if adata.n_vars == 0 and (rep_key is None or rep_key not in adata.obsm):
        raise AnnDataValidationError("AnnData must contain variables or a requested representation.")
    if batch_key not in adata.obs:
        raise AnnDataValidationError(f"Missing required obs column: {batch_key!r}.")

    n_batches = adata.obs[batch_key].nunique(dropna=False)
    if n_batches < min_batches:
        raise AnnDataValidationError(f"Expected at least {min_batches} batches in obs[{batch_key!r}], found {n_batches}.")

    if label_key is not None and label_key not in adata.obs:
        raise AnnDataValidationError(f"Missing requested label obs column: {label_key!r}.")

    if require_rep and rep_key is not None and rep_key not in adata.obsm:
        raise AnnDataValidationError(f"Missing required representation in obsm: {rep_key!r}.")

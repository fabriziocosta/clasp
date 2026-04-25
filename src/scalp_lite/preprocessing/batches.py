from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np


@dataclass(frozen=True)
class BatchSplit:
    batch: object
    indices: np.ndarray
    offset: int


def split_batches(adata: ad.AnnData, *, batch_key: str = "batch") -> list[BatchSplit]:
    """Return stable batch splits in first-observed batch order."""
    if batch_key not in adata.obs:
        raise KeyError(f"Missing obs column: {batch_key!r}")

    values = np.asarray(adata.obs[batch_key])
    seen: set[object] = set()
    splits: list[BatchSplit] = []
    offset = 0
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        indices = np.flatnonzero(values == value)
        splits.append(BatchSplit(batch=value, indices=indices, offset=offset))
        offset += len(indices)
    return splits

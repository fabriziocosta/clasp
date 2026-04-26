from __future__ import annotations

from pathlib import Path

import anndata as ad


def read_h5ad(path: str | Path) -> ad.AnnData:
    """Read an AnnData object from a .h5ad file."""
    path = Path(path)
    if path.suffix != ".h5ad":
        raise ValueError(f"Expected a .h5ad file, got {path}")
    if not path.exists():
        raise FileNotFoundError(
            f"AnnData input file not found: {path}. "
            "Pass the path to an existing .h5ad file or set CLASP_INPUT_H5AD."
        )
    return ad.read_h5ad(path)


def save_h5ad(adata: ad.AnnData, path: str | Path, *, compression: str | None = "gzip") -> None:
    """Write an AnnData object to a .h5ad file."""
    path = Path(path)
    if path.suffix != ".h5ad":
        raise ValueError(f"Expected a .h5ad output path, got {path}")
    adata.write_h5ad(path, compression=compression)

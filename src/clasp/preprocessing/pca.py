from __future__ import annotations

import anndata as ad
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler


def _as_array(matrix) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def ensure_pca(
    adata: ad.AnnData,
    *,
    rep_key: str = "X_pca",
    n_components: int = 40,
    random_state: int = 0,
    copy: bool = False,
) -> ad.AnnData:
    """Ensure `adata.obsm[rep_key]` exists, computing PCA/SVD from `adata.X` if needed."""
    target = adata.copy() if copy else adata
    if rep_key in target.obsm:
        return target

    if target.n_obs < 2 or target.n_vars < 2:
        raise ValueError("PCA requires at least two observations and two variables.")

    n_components = min(n_components, target.n_obs - 1, target.n_vars)
    if n_components < 1:
        raise ValueError("Could not determine a valid PCA dimensionality.")

    if sparse.issparse(target.X):
        model = TruncatedSVD(n_components=n_components, random_state=random_state)
        target.obsm[rep_key] = model.fit_transform(target.X)
    else:
        X = StandardScaler(with_mean=True, with_std=True).fit_transform(_as_array(target.X))
        target.obsm[rep_key] = PCA(n_components=n_components, random_state=random_state).fit_transform(X)
    return target

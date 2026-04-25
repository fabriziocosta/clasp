from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import anndata as ad
import numpy as np
from scipy import sparse

from scalp_lite.embedding import embed_graph
from scalp_lite.graph import build_scalp_graph
from scalp_lite.io import read_h5ad, validate_adata
from scalp_lite.preprocessing import ensure_pca


def _optional_scanpy():
    try:
        import scanpy as sc
    except ImportError:
        return None
    return sc


def _looks_log_normalized(matrix) -> bool:
    sample = matrix.data if sparse.issparse(matrix) else np.asarray(matrix).ravel()
    if sample.size == 0:
        return False
    sample = sample[: min(sample.size, 100_000)]
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return False
    mostly_integer = np.mean(np.isclose(finite, np.round(finite))) > 0.98
    return bool(finite.max() < 50 and not mostly_integer)


def _matrix_variance(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        mean = np.asarray(matrix.mean(axis=0)).ravel()
        mean_sq = np.asarray(matrix.multiply(matrix).mean(axis=0)).ravel()
        return mean_sq - mean**2
    return np.asarray(matrix).var(axis=0)


def _stratified_indices(values, *, max_cells: int, random_state: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    values = np.asarray(values)
    unique = list(dict.fromkeys(values))
    groups = [np.flatnonzero(values == value) for value in unique]

    quotas = np.array([len(group) for group in groups], dtype=float)
    quotas = quotas / quotas.sum() * max_cells
    counts = np.floor(quotas).astype(int)
    counts = np.minimum(counts, [len(group) for group in groups])

    remainder = max_cells - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(quotas - counts))
        for idx in order:
            if remainder == 0:
                break
            if counts[idx] < len(groups[idx]):
                counts[idx] += 1
                remainder -= 1

    selected = [rng.choice(group, size=count, replace=False) for group, count in zip(groups, counts) if count > 0]
    if not selected:
        return np.array([], dtype=int)
    return np.sort(np.concatenate(selected))


@dataclass
class ScalpEstimator:
    """Object-oriented facade for the SCALP-lite workflow."""

    rep_key: str = "X_pca"
    batch_key: str = "batch"
    label_key: str = "label"
    n_neighbors: int = 15
    intra_fraction: float = 0.5
    n_inter_edges: int = 1
    metric: str = "euclidean"
    assignment_quantile: float | None = 0.95
    symmetrize: bool = True
    n_components: int = 40
    random_state: int = 0
    embedding_method: str = "auto"
    embedding_components: int = 2

    def input(self, path: str | Path) -> ad.AnnData:
        """Read an `.h5ad` file and return an AnnData object."""
        return read_h5ad(path)

    def preprocess(
        self,
        adata: ad.AnnData,
        *,
        n_top_genes: int | None = 2000,
        max_cells: int | None = None,
        min_gene_counts: int = 3,
        normalize: bool | str = "auto",
        hvg_flavor: str = "cell_ranger",
        copy: bool = True,
    ) -> ad.AnnData:
        """Normalize expression, select variable genes, optionally subsample cells, and ensure PCA exists."""
        result = adata.copy() if copy else adata
        sc = _optional_scanpy()
        changed_expression = False

        if min_gene_counts > 0 and sc is not None and result.n_vars > 0:
            gene_mask, _ = sc.pp.filter_genes(result, min_counts=min_gene_counts, inplace=False)
            if not np.all(gene_mask):
                result = result[:, gene_mask].copy()
                changed_expression = True

        should_normalize = normalize
        if normalize == "auto":
            should_normalize = not bool(result.uns.get("normlog", False)) and not _looks_log_normalized(result.X)
        if should_normalize:
            if sc is None:
                raise ImportError("Expression normalization requires optional dependency `scanpy`.")
            sc.pp.normalize_total(result, target_sum=1e4)
            sc.pp.log1p(result)
            result.uns["normlog"] = True
            changed_expression = True

        if n_top_genes is not None and result.n_vars > n_top_genes:
            if n_top_genes < 1:
                raise ValueError("n_top_genes must be >= 1 or None.")
            if sc is not None:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="`n_top_genes` > number of normalized dispersions.*")
                        warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)
                        hvg = sc.pp.highly_variable_genes(
                            result,
                            n_top_genes=n_top_genes,
                            flavor=hvg_flavor,
                            inplace=False,
                        )
                    selected = np.asarray(hvg["highly_variable"], dtype=bool)
                    score_column = "dispersions_norm" if "dispersions_norm" in hvg else "variances_norm"
                    scores = hvg[score_column].fillna(0).to_numpy() if score_column in hvg else np.zeros(result.n_vars)
                    result.var["scalp_lite_hvg_score"] = scores
                except Exception:
                    selected = np.zeros(result.n_vars, dtype=bool)
            if sc is None or not np.any(selected):
                variances = _matrix_variance(result.X)
                selected = np.zeros(result.n_vars, dtype=bool)
                selected[np.argsort(variances)[::-1][:n_top_genes]] = True
                result.var["scalp_lite_hvg_score"] = variances
            result = result[:, selected].copy()
            result.var["scalp_lite_selected"] = True
            changed_expression = True

        if max_cells is not None and result.n_obs > max_cells:
            if max_cells < 1:
                raise ValueError("max_cells must be >= 1 or None.")
            if self.batch_key in result.obs:
                indices = _stratified_indices(result.obs[self.batch_key], max_cells=max_cells, random_state=self.random_state)
            else:
                rng = np.random.default_rng(self.random_state)
                indices = np.sort(rng.choice(result.n_obs, size=max_cells, replace=False))
            result = result[indices].copy()

        if changed_expression and self.rep_key in result.obsm:
            del result.obsm[self.rep_key]
        ensure_pca(result, rep_key=self.rep_key, n_components=self.n_components, random_state=self.random_state)
        return result

    def data_to_graph(self, adata: ad.AnnData) -> sparse.csr_matrix:
        """Build and return the SCALP-lite graph for an AnnData object."""
        validate_adata(adata, batch_key=self.batch_key, rep_key=self.rep_key, require_rep=True)
        return build_scalp_graph(
            adata,
            rep_key=self.rep_key,
            batch_key=self.batch_key,
            n_neighbors=self.n_neighbors,
            intra_fraction=self.intra_fraction,
            n_inter_edges=self.n_inter_edges,
            metric=self.metric,
            assignment_quantile=self.assignment_quantile,
            symmetrize=self.symmetrize,
        )

    def graph_to_vector(self, graph: sparse.spmatrix) -> np.ndarray:
        """Embed a graph and return low-dimensional vectors."""
        return embed_graph(
            graph,
            method=self.embedding_method,
            n_components=self.embedding_components,
            random_state=self.random_state,
        )

    def embed(self, adata: ad.AnnData) -> np.ndarray:
        """Build the graph from AnnData and return embedding vectors."""
        graph = self.data_to_graph(adata)
        return self.graph_to_vector(graph)

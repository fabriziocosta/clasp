from __future__ import annotations

import time

import anndata as ad
import numpy as np
from scipy import sparse

from scalp_lite.graph.inter import build_inter_batch_graph
from scalp_lite.graph.intra import build_intra_batch_graph
from scalp_lite.io import validate_adata
from scalp_lite.preprocessing import split_batches


def build_scalp_graph(
    adata: ad.AnnData,
    *,
    rep_key: str = "X_pca",
    batch_key: str = "batch",
    n_neighbors: int = 15,
    intra_fraction: float = 0.5,
    n_inter_edges: int = 1,
    metric: str = "euclidean",
    assignment_quantile: float | None = 0.95,
    symmetrize: bool = True,
) -> sparse.csr_matrix:
    """Build the SCALP-lite integrated graph for an AnnData object."""
    validate_adata(adata, batch_key=batch_key, rep_key=rep_key, require_rep=True)
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be >= 1.")
    if not 0 <= intra_fraction <= 1:
        raise ValueError("intra_fraction must be between 0 and 1.")

    started = time.perf_counter()
    splits = split_batches(adata, batch_key=batch_key)
    X = np.asarray(adata.obsm[rep_key])
    batch_arrays = [X[split.indices] for split in splits]

    intra_neighbors = int(np.ceil(n_neighbors * intra_fraction))
    blocks: list[list[sparse.csr_matrix | None]] = [[None for _ in splits] for _ in splits]
    intra_edges = 0
    inter_edges = 0

    for i, Xi in enumerate(batch_arrays):
        block = build_intra_batch_graph(Xi, n_neighbors=intra_neighbors, metric=metric)
        blocks[i][i] = block
        intra_edges += block.nnz

    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            block = build_inter_batch_graph(
                batch_arrays[i],
                batch_arrays[j],
                n_inter_edges=n_inter_edges,
                metric=metric,
                assignment_quantile=assignment_quantile,
            )
            blocks[i][j] = block
            blocks[j][i] = block.T.tocsr()
            inter_edges += block.nnz * 2

    graph = sparse.bmat(blocks, format="csr", dtype=np.float32)
    graph.setdiag(0)
    graph.eliminate_zeros()
    if symmetrize:
        graph = graph.maximum(graph.T).tocsr()
        graph.setdiag(0)
        graph.eliminate_zeros()

    adata.uns.setdefault("scalp_lite", {})
    adata.uns["scalp_lite"]["graph"] = {
        "batch_key": batch_key,
        "rep_key": rep_key,
        "batch_order": [str(split.batch) for split in splits],
        "offsets": [int(split.offset) for split in splits],
        "sizes": [int(len(split.indices)) for split in splits],
        "parameters": {
            "n_neighbors": n_neighbors,
            "intra_fraction": intra_fraction,
            "n_inter_edges": n_inter_edges,
            "metric": metric,
            "assignment_quantile": assignment_quantile,
            "symmetrize": symmetrize,
        },
        "edge_counts": {
            "intra": int(intra_edges),
            "inter": int(inter_edges),
            "total": int(graph.nnz),
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    return graph

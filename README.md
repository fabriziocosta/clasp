# scalp-lite

`scalp-lite` is a small, dependency-light reimplementation of the core SCALP idea: integrate batches of single-cell data by combining within-batch nearest-neighbor edges with cross-batch Hungarian assignment edges, then embed and score the resulting graph.

It intentionally avoids the legacy research dependencies from the original `cellsaw` repository. The required interface is an in-memory `AnnData` object or a `.h5ad` file.

## Install

```bash
pip install -e ".[dev,umap]"
```

UMAP is optional. If `umap-learn` is not installed, `embed_graph(method="auto")` falls back to spectral embedding from scikit-learn.

## Quick Start

```python
from scalp_lite import (
    read_h5ad,
    validate_adata,
    ensure_pca,
    build_scalp_graph,
    embed_graph,
    score_embedding,
    plot_embedding_pair,
)

adata = read_h5ad("input.h5ad")
validate_adata(adata, batch_key="batch", label_key="label")
ensure_pca(adata, rep_key="X_pca", n_components=40)

graph = build_scalp_graph(
    adata,
    rep_key="X_pca",
    batch_key="batch",
    n_neighbors=15,
    intra_fraction=0.5,
    n_inter_edges=1,
)

adata.obsm["X_scalp"] = embed_graph(graph, method="auto")
scores = score_embedding(adata, embedding_key="X_scalp", batch_key="batch", label_key="label", graph=graph)

plot_embedding_pair(adata, embedding_key="X_scalp", batch_key="batch", label_key="label")
```

## AnnData Schema

Required:

- `adata.X`: expression matrix if PCA must be computed.
- `adata.obs["batch"]`: batch, dataset, or time-point identifier.

Optional:

- `adata.obs["label"]`: biological label used for visualization and metrics.
- `adata.obsm["X_pca"]`: precomputed representation used for graph construction.

## Notebooks

- `notebooks/01_visualize_embedding.ipynb`: load data, build graph, embed in 2D, and plot.
- `notebooks/02_evaluate_embedding.ipynb`: compute embedding quality metrics and export a CSV report.

The notebooks use `input.h5ad` as a placeholder. Either place your dataset at that path from the repository root or set:

```bash
export SCALP_INPUT_H5AD=/path/to/your_dataset.h5ad
```

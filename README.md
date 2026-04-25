# scalp-lite

`scalp-lite` is a small, dependency-light reimplementation of the core SCALP idea: integrate batches of single-cell data by combining within-batch nearest-neighbor edges with cross-batch Hungarian assignment edges, then embed and score the resulting graph.

It intentionally avoids the legacy research dependencies from the original `cellsaw` repository. The required interface is an in-memory `AnnData` object or a `.h5ad` file.

## Install

```bash
pip install -e ".[dev,umap,preprocess]"
```

UMAP is optional. If `umap-learn` is not installed, `embed_graph(method="auto")` falls back to spectral embedding from scikit-learn.

## Quick Start

```python
from scalp_lite import (
    ScalpEstimator,
    save_h5ad,
    score_embedding,
    plot_embedding_pair,
)

estimator = ScalpEstimator(batch_key="batch", label_key="label")
adata = estimator.input("input.h5ad")
adata = estimator.preprocess(adata, n_top_genes=2000, max_cells=None)

graph = estimator.data_to_graph(adata)
adata.obsm["X_scalp"] = estimator.graph_to_vector(graph)
scores = score_embedding(adata, embedding_key="X_scalp", batch_key="batch", label_key="label", graph=graph)

plot_embedding_pair(adata, embedding_key="X_scalp", batch_key="batch", label_key="label")
save_h5ad(adata, "scalp_lite_embedded.h5ad")
```

`plot_embedding_pair` uses a viridis palette for batches and a categorical `tab20` palette for labels by default.

When Scanpy is installed, `ScalpEstimator.preprocess` follows the standard single-cell preprocessing pattern used by the original project: filter genes with low counts, normalize each cell to 10,000 counts, apply `log1p`, select highly variable genes with `scanpy.pp.highly_variable_genes`, and compute PCA. If Scanpy is unavailable, it falls back to variance-based gene selection.

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

The notebooks default to the ignored local dataset `data/cellrank-pancreas.h5ad`, a CellRank pancreas development example with fine cell-type labels in `obs["clusters_fine"]`. Because this example has no real batch column, the visualization notebook creates deterministic artificial `batch` splits for smoke testing and maps `clusters_fine` to `label`.

To use a different dataset, set:

```bash
export SCALP_INPUT_H5AD=/path/to/your_dataset.h5ad
```

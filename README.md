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
    ScalpEstimator,
    score_embedding,
)

# batch_key is the technical batch/sample/time column; label_key is the biological cell-type column.
estimator = ScalpEstimator(batch_key="batch", label_key="label")
adata = estimator.to_data("input.h5ad")
adata = estimator.preprocess(
    adata,
    # Number of highly variable genes to keep before PCA.
    n_top_genes=2000,
    # Optional cell cap for faster experiments; None keeps all cells.
    max_cells=None,
    # Optional minimum detected genes per cell; None disables cell filtering.
    min_cell_genes=None,
    # Remove genes observed fewer than this many total counts.
    min_gene_counts=3,
    # "auto" normalizes only when data does not look already log-normalized.
    normalize="auto",
    # Library-size target used by scanpy.pp.normalize_total.
    target_sum=1e4,
    # Apply scanpy.pp.log1p after normalization.
    log1p=True,
    # Scanpy HVG flavor used for gene selection.
    hvg_flavor="cell_ranger",
    # Optional batch-aware HVG selection key.
    hvg_batch_key=None,
    # Create deterministic split_* batches only for single-batch smoke tests.
    create_artificial_batch=False,
    # Number of deterministic split_* batches if artificial batches are created.
    artificial_batch_count=3,
    # Ordered fallback columns for inferring label_key when obs[label_key] is absent.
    label_candidates=(
        "clusters_coarse",
        "cell_type",
        "lineages",
        "clusters_fine",
        "clusters",
        "leiden",
        "louvain",
    ),
    # Copy input AnnData before preprocessing.
    copy=True,
)

graph = estimator.data_to_graph(
    adata,
    # Total neighborhood scale used to build the integrated graph.
    n_neighbors=15,
    # Fraction of neighbors assigned to within-batch kNN edges.
    intra_fraction=0.5,
    # Number of repeated Hungarian assignment layers between batch pairs.
    n_inter_edges=1,
    # Keep only cross-batch assignments up to this distance quantile.
    assignment_quantile=0.95,
    # Apply CSLS hubness correction before kNN and Hungarian assignment.
    hubness_correction="csls",
    # Local neighborhood size used by CSLS.
    hubness_k=10,
    # Use binary for paper-compatible graph connectivity, or distance for weighted edges.
    edge_weighting="binary",
    # Symmetrize the final graph.
    symmetrize=True,
)
adata.obsm["X_scalp"] = estimator.graph_to_vector(graph)
scores = score_embedding(adata, embedding_key="X_scalp", batch_key="batch", label_key="label", graph=graph)

estimator.plot(adata, embedding_key="X_scalp")
estimator.save(adata, "scalp_lite_embedded.h5ad")
```

`estimator.plot` uses a viridis palette for batches and a categorical `tab20` palette for labels by default.

`ScalpEstimator.preprocess` follows the standard single-cell preprocessing pattern used by the original project: optionally filter cells and genes, normalize each cell to `target_sum`, optionally apply `log1p`, select highly variable genes with `scanpy.pp.highly_variable_genes`, and compute PCA.

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

The notebooks default to `data/pancreas_normalized.h5ad`, a real batch-integration example with five pancreas studies/platforms in `obs["study"]` and curated cell types in `obs["cell_type"]`. The visualization notebook writes `data/pancreas_normalized-scalp.h5ad`, which the evaluation notebook then reads.

To switch datasets, edit `selected_dataset` in the first notebook cell. Available local choices are `pancreas`, `zebrafish`, and `pbmc3k`.

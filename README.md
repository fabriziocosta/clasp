# CLASP

**CLASP** stands for **Cell integration via Linear Assignment and Sparse Pairing**.

`clasp` integrates batches of single-cell data by combining within-batch nearest-neighbor edges with cross-batch edges derived from Hungarian assignment, then embeds and scores the resulting graph.

The required interface is an in-memory `AnnData` object or a `.h5ad` file.

## Install

```bash
pip install -e ".[dev,umap]"
```

UMAP is optional. If `umap-learn` is not installed, `embed_graph(method="auto")` falls back to spectral embedding from scikit-learn.

## Documentation

- [CLASP Whitepaper](docs/WHITEPAPER.md): method overview, motivation, data model, graph construction, hubness correction, assignment strategy, embedding, and evaluation concepts.
- [Processing Pipeline](docs/PROCESSING_PIPELINE.md): dataset YAML configuration, notebook flow, download handling, preprocessing, tuning artifacts, embedding outputs, and recommended end-to-end workflow.
- [Latent Bayesian Optimization](docs/OPTIMIZATION.md): optional hyperparameter tuning with BoTorch/GPyTorch, search-space encoding, latent optimization, candidate repair, and saved optimized parameter files.

## Quick Start

```python
from clasp import (
    ClaspEstimator,
    score_embedding,
)

# Use preset="balanced" for optimized benchmark-derived defaults.
# Use preset="trajectory" for smoother CellRank-style temporal datasets.
estimator = ClaspEstimator(preset="balanced", batch_key="batch", label_key="label")
adata = estimator.to_data("input.h5ad")
adata = estimator.preprocess(
    adata,
    # Number of highly variable genes to keep before PCA.
    n_top_genes=estimator.preprocess_defaults["n_top_genes"],
    # Optional cell cap for faster experiments; None keeps all cells.
    max_cells=None,
    # Optional minimum detected genes per cell; None disables cell filtering.
    min_cell_genes=None,
    # Remove genes observed fewer than this many total counts.
    min_gene_counts=estimator.preprocess_defaults["min_gene_counts"],
    # "auto" normalizes only when data does not look already log-normalized.
    normalize="auto",
    # Library-size target used by scanpy.pp.normalize_total.
    target_sum=1e4,
    # Apply scanpy.pp.log1p after normalization.
    log1p=True,
    # HVG flavor used for gene selection.
    hvg_flavor=estimator.preprocess_defaults["hvg_flavor"],
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

graph = estimator.data_to_graph(adata)
adata.obsm["X_clasp"] = estimator.graph_to_embeddings(graph)
scores = score_embedding(adata, embedding_key="X_clasp", batch_key="batch", label_key="label", graph=graph)

estimator.plot(adata, embedding_key="X_clasp")
estimator.save(adata, "clasp_embedded.h5ad")
```

`estimator.plot` uses a viridis palette for batches and a categorical `tab20` palette for labels by default. It shuffles the draw order reproducibly so ordered labels or batches do not hide mixing in crowded regions.
Pass `filename="figures/my_embedding"` to also save the displayed plot as a high-resolution PNG, or provide an explicit suffix such as `.pdf` when vector output is smaller. Omitting `filename` keeps the default display-only behavior.

`ClaspEstimator.preprocess` follows the standard single-cell preprocessing pattern used by the original project: optionally filter cells and genes, normalize each cell to `target_sum`, optionally apply `log1p`, select highly variable genes with `scanpy.pp.highly_variable_genes`, and compute PCA.

## AnnData Schema

Required:

- `adata.X`: expression matrix if PCA must be computed.
- `adata.obs["batch"]`: batch, dataset, or time-point identifier.

Optional:

- `adata.obs["label"]`: biological label used for visualization and metrics.
- `adata.obsm["X_pca"]`: precomputed representation used for graph construction.

## Notebooks

- `notebooks/00_download_datasets.ipynb`: download registered `.h5ad` datasets into `data/`.
- `notebooks/01_optimize_all_datasets.ipynb`: run PCA latent BO plus GPLVM refinement for all paper datasets and save optimized configs.
- `notebooks/02_latent_bayesopt.ipynb`: optimize preprocessing PCA and graph parameters, then save them to `data/optimized_params/`.
- `notebooks/03_visualize_embedding.ipynb`: load optimized preprocessing and graph parameters, embed in 2D, plot, and save the embedded AnnData.
- `notebooks/04_evaluate_embedding.ipynb`: compute embedding quality metrics and export a CSV report.
- `notebooks/05_assignment_quantile_sweep.ipynb`: sweep graph parameters around the optimized baseline.
- `notebooks/06_integrated_pipeline.ipynb`: tune parameters, save them, embed, save the embedded AnnData file, and plot in one workflow.

The notebooks default to `data/pancreas_normalized.h5ad`, a real batch-integration example with five pancreas studies/platforms in `obs["study"]` and curated cell types in `obs["cell_type"]`. The optimization notebook writes optimized graph parameters, the visualization notebook writes `data/pancreas_normalized-clasp.h5ad`, and the evaluation notebook then reads that embedded file.

To switch datasets, edit `selected_dataset` in the first notebook cell. Available local choices are `pancreas`, `zebrafish`, and `pbmc3k`.

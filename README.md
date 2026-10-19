# CLASP

**CLASP** stands for **Cell integration via Linear Assignment and Sparse Pairing**.

`clasp` integrates batches of single-cell data by combining within-batch nearest-neighbor edges with cross-batch edges derived from Hungarian assignment, then embeds and scores the resulting graph.

The required interface is an in-memory `AnnData` object or a `.h5ad` file.

## Install

```bash
pip install -e ".[dev]"
```

IMSTE is installed as a required dependency from its GitHub repository. For development against local checkouts, install IMSTE editable first, then install CLASP:

```bash
pip install -e ".[dev,umap]"
pip install -e "/path/to/IteratedMinimumSpanningTree" --no-deps
```

`ClaspEstimator` uses IMSTE by default. UMAP and spectral embedding remain available through `ClaspGraphEmbedder` (UMAP requires the optional `umap-learn` dependency); the older `embedding_method=` argument remains as a deprecated compatibility shortcut.

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

`data_to_graph` returns a CLASP affinity graph. The estimator converts its distance-weighted affinities back to distances when the configured embedder is IMSTE. IMSTE requires `edge_weighting="distance"`; binary edges do not retain the distance ranking it uses to build spanning trees.

Configure IMSTE directly when you want to tune its optimizer and tree settings:

```python
from imste import IteratedMinimumSpanningTreeEmbedder
from clasp import ClaspEstimator

imste = IteratedMinimumSpanningTreeEmbedder(
    n_msts=15,
    n_epochs=300,
    n_components=2,
    random_state=42,
)
estimator = ClaspEstimator(embedder=imste, batch_key="batch", label_key="label")
```

To select a legacy graph embedder explicitly, pass `ClaspGraphEmbedder(method="umap")` or `ClaspGraphEmbedder(method="spectral")` as `embedder=`.

Build a directed label-neighbor graph from any sample-by-feature matrix, including 2D embeddings or higher-dimensional representations:

```python
from clasp import LabelNeighborGraph

label_graph = LabelNeighborGraph(k=30).fit(X, adata.obs["cell_type"])
label_graph.plot()
```

To share node positions between graphs, compute label centroids in a 2D embedding and export them. A graph fitted on another representation can import the same positions before plotting:

```python
umap_graph = LabelNeighborGraph(k=30).fit(umap_2d, labels)
umap_graph.set_positions_from_embedding(umap_2d)
umap_graph.export_positions("label_positions.json")

imste_graph = LabelNeighborGraph(k=30).fit(imste_data, labels)
imste_graph.import_positions("label_positions.json")
imste_graph.plot()
```

`estimator.plot` uses a viridis palette for batches and a categorical `tab20` palette for labels by default. It shuffles the draw order reproducibly so ordered labels or batches do not hide mixing in crowded regions.
Pass `filename="figures/my_embedding"` to also save the displayed plot as a high-resolution PNG, or provide an explicit suffix such as `.pdf` when vector output is smaller. Omitting `filename` keeps the default display-only behavior.

`ClaspEstimator.preprocess` follows the standard single-cell preprocessing pattern used by the original project: optionally filter cells and genes, normalize each cell to `target_sum`, optionally apply `log1p`, select highly variable genes with `scanpy.pp.highly_variable_genes`, and compute PCA.

## Command Line

The first Galaxy-oriented command is `clasp embed`. It reads an `.h5ad`, preprocesses it, builds the CLASP graph, writes an embedded `.h5ad`, and can optionally save a plot.

Show available commands:

```bash
clasp --help
```

Show all options for embedding:

```bash
clasp embed --help
```

Basic embedding:

```bash
clasp embed input.h5ad clasp_embedded.h5ad \
  --batch-key batch \
  --label-key label \
  --preset balanced \
  --max-cells 6000 \
  --figure figures/clasp_embedding.pdf
```

Use `--preset balanced` for the benchmark-derived general default. Use `--preset trajectory` for smoother CellRank-style temporal datasets.

Inputs and outputs:

- `input`: input AnnData `.h5ad`.
- `output`: output AnnData `.h5ad` with the CLASP embedding stored in `obsm["X_clasp"]` by default.
- `--figure`: optional plot path, for example `.pdf` or `.png`. If `--label-key` is absent from `adata.obs`, the label panel is drawn with all cells in one pale-blue unlabeled category.

Core options:

- `--batch-key`: `adata.obs` column for batch, sample, donor, dataset, or time point.
- `--label-key`: optional `adata.obs` column for biological labels used in plots. If missing, embedding still runs and plots use a pale-blue unlabeled panel.
- `--rep-key`: input representation in `adata.obsm`; default is `X_pca`.
- `--embedding-key`: output embedding key; default is `X_clasp`.
- `--random-state`: seed for reproducible preprocessing and embedding.
- `--embedding-method`: deprecated legacy backend selector (`auto`, `umap`, or `spectral`); omitted uses IMSTE.
- `--embedding-components`: number of output embedding dimensions.

Preprocessing overrides:

- `--n-top-genes`: number of highly variable genes to keep; use `none` to disable HVG selection.
- `--max-cells`: optional stratified cell cap for faster runs.
- `--min-cell-genes`: optional minimum detected genes per cell.
- `--min-gene-counts`: minimum total counts per gene.
- `--normalize`: `auto`, `true`, or `false`.
- `--target-sum`: library-size target for normalization.
- `--log1p`: `true` or `false`.
- `--hvg-flavor`: HVG method, for example `variance` or `cell_ranger`.
- `--create-artificial-batch`: `true` or `false`; useful only for smoke tests or single-batch files.
- `--artificial-batch-count`: number of deterministic `split_*` artificial batches.

Estimator and graph overrides:

- `--n-components`: PCA components used before graph construction.
- `--n-neighbors`: total neighborhood scale.
- `--intra-fraction`: fraction of neighbors assigned to within-batch kNN edges.
- `--n-inter-edges`: number of assignment layers between batch pairs.
- `--metric`: distance metric.
- `--assignment-quantile`: retained assignment distance quantile; use `none` to disable filtering.
- `--hubness-correction`: `csls` or `none`.
- `--hubness-k`: CSLS local neighborhood size.
- `--rank-correction`: `true` or `false`.
- `--edge-weighting`: `distance` or `binary`.
- `--inter-edge-mode`: `assignment` or `propagate_neighbors`.
- `--mutual-neighbors`: `true` or `false`.
- `--neighbor-mode`: `distance` or `rank`.
- `--symmetrize`: `true` or `false`.

Examples:

```bash
# Fast smoke test on a small subset.
clasp embed input.h5ad smoke.h5ad \
  --batch-key batch \
  --label-key label \
  --max-cells 1000 \
  --embedding-method spectral
```

```bash
# Temporal trajectory-oriented preset.
clasp embed input.h5ad trajectory_clasp.h5ad \
  --batch-key day \
  --label-key cell_type \
  --preset trajectory \
  --figure figures/trajectory_clasp.pdf
```

```bash
# Manual graph override while keeping the balanced preset for unspecified values.
clasp embed input.h5ad custom_clasp.h5ad \
  --batch-key batch \
  --label-key label \
  --preset balanced \
  --n-neighbors 30 \
  --n-inter-edges 4 \
  --assignment-quantile 0.5 \
  --mutual-neighbors true
```

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
- `notebooks/07_imste_single_cell.ipynb`: read an existing single-cell `.h5ad` in place, convert a stored representation to NumPy, and try the separate IMSTE embedder without writing a dataset copy.

The notebooks default to `data/pancreas_normalized.h5ad`, a real batch-integration example with five pancreas studies/platforms in `obs["study"]` and curated cell types in `obs["cell_type"]`. The optimization notebook writes optimized graph parameters, the visualization notebook writes `data/pancreas_normalized-clasp.h5ad`, and the evaluation notebook then reads that embedded file.

To switch datasets, edit `selected_dataset` in the first notebook cell. Available local choices are `pancreas`, `zebrafish`, and `pbmc3k`.

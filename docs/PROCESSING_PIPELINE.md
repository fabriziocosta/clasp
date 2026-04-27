# Processing Pipeline

This document describes the end-to-end CLASP workflow used by the notebooks and helper functions. The pipeline is name driven: a notebook chooses a dataset name, and the dataset-specific paths, labels, preprocessing defaults, graph defaults, and download metadata are loaded from YAML.

## Dataset Configuration

Dataset metadata lives in:

```text
src/clasp/datasets/*.yaml
```

Each file is keyed by its `name` field and contains the information needed to process that dataset:

```yaml
name: scib_pancreas
input: data/human_pancreas_norm_complexBatch.h5ad
embedded: data/human_pancreas_norm_complexBatch-clasp.h5ad
batch_key: tech
label_key: celltype
source: https://figshare.com/articles/dataset/12420968
paper_group: scIB benchmark
download:
  kind: figshare
  filename: human_pancreas_norm_complexBatch.h5ad
  article_id: 12420968
  preferred_file: human_pancreas_norm_complexBatch.h5ad
preprocess:
  normalize: false
  hvg_flavor: variance
  min_gene_counts: 0
  create_artificial_batch: false
graph:
  n_neighbors: 20
  intra_fraction: 0.5
  n_inter_edges: 5
  assignment_quantile: 0.35
  hubness_correction: csls
  hubness_k: 10
  rank_correction: true
  edge_weighting: distance
  mutual_neighbors: false
  neighbor_mode: distance
  symmetrize: true
```

`dataset_config(name)` loads this YAML and resolves paths relative to the project root:

```python
dataset = dataset_config("scib_pancreas")
```

The rest of the pipeline should use this returned `dataset` object rather than hardcoding paths or column names.

## Notebook Flow

The intended notebook order is:

1. `00_download_datasets.ipynb`: download configured datasets.
2. `01_latent_bayesopt.ipynb`: optimize preprocessing, estimator, and graph parameters for one selected dataset.
3. `02_visualize_embedding.ipynb`: load saved parameters, run CLASP, save the embedded `.h5ad`, and plot batch/label panels.
4. `03_evaluate_embedding.ipynb`: compute integration metrics for an embedded dataset.
5. `04_assignment_quantile_sweep.ipynb`: run small one-parameter sweeps for visual diagnostics.

The common pattern is:

```python
selected_dataset = "scib_pancreas"
dataset = dataset_config(selected_dataset)
estimator = make_estimator(dataset, random_state=0)
adata = estimator.to_data(dataset["input_path"])
```

## Download Step

Download metadata comes from the `download` block in the dataset YAML. `download_datasets(...)` supports:

- `kind: url`
- `kind: figshare`
- `kind: cellrank`

The paper download notebook uses `PAPER_DATASET_DOWNLOADS`, which is a curated list of dataset names. The helper resolves each name to YAML metadata, downloads into `data/`, and returns a mapping from dataset name to local path.

`summarize_downloads(...)` reports downloaded file paths and sizes by default. Full AnnData inspection is optional because several paper datasets are multi-GB files.

## Input Data

`ClaspEstimator.to_data(path)` wraps `.h5ad` loading:

```python
adata = estimator.to_data(dataset["input_path"])
```

The input must be an AnnData object with:

- `adata.X`: expression matrix, unless a representation already exists and preprocessing will not recompute it.
- `adata.obs[batch_key]`: the batch/domain column from the YAML.
- `adata.obs[label_key]`: optional biological labels for plotting and scoring.

## Preprocessing

Preprocessing is handled by:

```python
adata = estimator.preprocess(adata, **preprocess_params(dataset, **overrides))
```

The YAML `preprocess` block carries dataset-specific defaults. This matters because some downloaded benchmark files are already normalized and should not be normalized again.

The preprocessing sequence is:

1. Optionally create deterministic artificial batches if requested.
2. If `label_key` is missing, copy the first available label from `label_candidates`.
3. Optionally filter cells by `min_cell_genes`.
4. Optionally filter genes by `min_gene_counts`.
5. Normalize and log-transform if `normalize=True`, or if `normalize="auto"` decides the data are not already log-normalized.
6. Select highly variable genes by Scanpy HVG or variance fallback.
7. Optionally subsample cells, stratified by `batch_key` when present.
8. Remove stale PCA if expression changed.
9. Compute PCA into `adata.obsm[rep_key]`.

Important parameters:

- `n_top_genes`: number of genes retained before PCA.
- `max_cells`: optimization-time subsampling limit.
- `normalize`: `False`, `True`, or `"auto"`.
- `hvg_flavor`: Scanpy HVG flavor, or `"variance"` for a direct variance ranking.
- `hvg_batch_key`: optional batch-aware HVG key.
- `label_candidates`: ordered fallback columns for labels.

## Estimator Construction

`make_estimator(dataset, ...)` creates a `ClaspEstimator` with the YAML batch and label keys:

```python
estimator = make_estimator(dataset, n_components=80, random_state=0)
```

The estimator stores defaults for:

- representation key, usually `X_pca`
- batch and label keys
- graph parameters
- PCA component count
- embedding method and output dimension

## Graph Construction

Graph construction is split into:

```python
graph = estimator.data_to_graph(adata, **graph_params)
```

or, as part of embedding:

```python
adata.obsm["X_clasp"] = estimator.embed(adata, **graph_params)
```

The graph builder first validates that `adata.obs[batch_key]` and `adata.obsm[rep_key]` exist. It then splits cells by batch and builds a block sparse graph:

- diagonal blocks: within-batch neighbor graphs
- off-diagonal blocks: cross-batch graphs derived from linear assignment

### Within-Batch Edges

For each batch, CLASP builds a k-nearest-neighbor graph from PCA coordinates. The number of intra-batch neighbors is:

$$
k_{\mathrm{intra}} =
\left\lceil
n_{\mathrm{neighbors}} \cdot \rho_{\mathrm{intra}}
\right\rceil
$$

where `rho_intra` is `intra_fraction`.

Options include:

- `hubness_correction="csls"` to reduce high-dimensional neighbor hubs.
- `rank_correction=True` to transform corrected distances into local ranks.
- `mutual_neighbors=True` to retain only mutual neighbor relations.
- `edge_weighting="binary"` or `"distance"`.

### Cross-Batch Edges

For each pair of batches, CLASP computes pairwise distances in representation space, optionally applies CSLS and rank correction, then solves repeated linear assignment problems. The retained assignments define reliable cross-batch partners without allowing one cell to become a hub for many cells.

The default `inter_edge_mode="propagate_neighbors"` does not add direct assignment edges. If cell `x_i` in one batch is assigned to `x_j` in the other batch, `x_i` is linked to the nearest neighbors of `x_j` inside `x_j`'s batch, using those neighbor distances as the edge distances. The assigned pair itself is linked only if it appears through this propagated local-neighbor graph.

The number of propagated cross-batch neighbors is:

$$
k_{\mathrm{inter}} =
n_{\mathrm{neighbors}} - k_{\mathrm{intra}}
$$

Set `inter_edge_mode="assignment"` to recover the legacy behavior that links retained assigned pairs directly.

Key parameters:

- `n_inter_edges`: number of repeated assignment passes.
- `assignment_quantile`: keeps only confident assignments below the selected distance quantile.
- `hubness_k`: local neighborhood size for CSLS scaling.
- `inter_edge_mode`: `"propagate_neighbors"` for default neighbor inheritance, or `"assignment"` for direct assigned-pair edges.

The final graph is assembled with `scipy.sparse.bmat`, symmetrized if requested, and reordered back to the original AnnData cell order. Metadata about graph parameters, batch order, edge counts, and runtime is stored in:

```python
adata.uns["clasp"]["graph"]
```

## Embedding

`graph_to_vector(...)` embeds the integrated graph:

```python
embedding = estimator.graph_to_vector(graph, method="umap", n_components=2)
```

`embed(...)` combines graph construction and embedding:

```python
adata.obsm["X_clasp"] = estimator.embed(
    adata,
    **graph_params,
    embedding_method="umap",
    embedding_components=2,
)
```

Supported embedding methods are:

- `auto`: use UMAP if installed, otherwise spectral embedding.
- `umap`: convert graph weights to a precomputed distance matrix and run UMAP.
- `spectral`: run sklearn spectral embedding on the graph affinity.

Notebook optimization currently uses UMAP by default so the objective matches the visualization behavior.

## Plotting

The paired batch/label plot is wrapped by:

```python
estimator.plot(adata, embedding_key="X_clasp")
```

The plot function renders two synchronized panels:

- left: embedding colored by `batch_key`
- right: embedding colored by `label_key`

Rows are randomly permuted before plotting so ordered cell types do not hide mixing patterns in dense regions.

## Optimization

Notebook 01 optimizes a mixed parameter space containing:

- preprocessing parameters, such as `n_top_genes`
- estimator parameters, such as PCA `n_components`
- graph parameters, such as `n_neighbors`, `assignment_quantile`, `edge_weighting`, and `inter_edge_mode`

The objective:

1. Preprocesses a copy of the raw AnnData.
2. Builds and embeds the CLASP graph.
3. Scores the embedding.
4. Returns a scalar score.

The default scalar objective is:

$$
\mathrm{score}
=
\mathrm{knn\_label\_agreement}
+ 0.25 \cdot \mathrm{batch\_mixing}
- 0.05 \cdot \mathrm{graph\_density}
$$

The optimizer first runs PCA latent BO. Optional GPLVM refinement can then search a compact space around the PCA best parameters. The compact search is controlled by `compact_radii`, which defines the local radius around each best numeric parameter.

The best settings are saved to:

```text
data/optimized_params/<dataset>_graph_params.json
```

Downstream notebooks load this file with:

```python
optimized_params = load_or_default_params(selected_dataset, dataset)
```

If no optimization file exists, registry/YAML defaults are used.

## Saving And Reuse

The final embedded AnnData is saved with:

```python
estimator.save(adata, dataset["output_path"])
```

This file can then be used for plotting, evaluation, or downstream analysis without rebuilding the graph.

## Adding A New Dataset

To add a dataset:

1. Create `src/clasp/datasets/<name>.yaml`.
2. Set `input`, `embedded`, `batch_key`, and `label_key`.
3. Add a `download` block if the dataset should be downloadable from notebook 00.
4. Set preprocessing defaults that match the file state, especially `normalize`.
5. Set graph defaults if the dataset needs nonstandard behavior.
6. Use the new dataset by name:

```python
dataset = dataset_config("<name>")
```

No notebook code should need dataset-specific paths or obs-column names once the YAML file is present.

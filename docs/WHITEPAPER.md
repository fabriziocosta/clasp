# SCALP-lite Whitepaper

## Summary

SCALP-lite is a compact implementation of a graph-based strategy for integrating single-cell datasets across batches, donors, technologies, time points, or related biological conditions. The central idea is to build a cell-cell graph that preserves local neighborhoods inside each batch while adding sparse, globally optimized links across batches.

The method is intentionally simple:

1. Represent each cell in a common feature space, usually PCA.
2. Build a within-batch nearest-neighbor graph for every batch.
3. For every pair of batches, solve a linear assignment problem to match cells across batches.
4. Keep only confident cross-batch assignments.
5. Assemble all within-batch and cross-batch blocks into one sparse graph.
6. Use the graph for embedding, visualization, and integration metrics.

This design avoids directly forcing all cells into the same coordinates during graph construction. Instead, it creates an integrated neighborhood structure that downstream tools can embed or score.

## Motivation

Single-cell RNA-seq datasets often contain technical and biological variation that is confounded with batch labels. Standard batch correction works well when batches are replicates with high cell-type overlap, but harder cases arise when batches are time points, developmental stages, disease states, species, or technologies.

In those harder cases, fully mixing all batches can be wrong: some cells should align, while distinct biological states should remain separate. SCALP-lite addresses this by adding cross-batch links sparsely and globally. The Hungarian assignment step prevents one popular cell from becoming the nearest neighbor of many cells in another batch, while the distance cutoff removes weak assignments that likely represent non-overlapping biology.

## Data Model

Input is an `AnnData` object or `.h5ad` file.

Required fields:

- `adata.X`: expression matrix, needed if PCA must be computed.
- `adata.obs[batch_key]`: batch, dataset, donor, time point, or condition label.

Optional fields:

- `adata.obs[label_key]`: biological label used for evaluation and plotting.
- `adata.obsm[rep_key]`: precomputed representation, usually `X_pca`.

SCALP-lite works on `adata.obsm[rep_key]`. If that representation is missing, the project provides `ensure_pca()` to compute it from `adata.X`.

## Algorithm

Let there be `m` batches:

```text
X = {X_1, X_2, ..., X_m}
```

where `X_i` is the matrix of cells in batch `i`, represented in PCA or another shared embedding.

The output is a sparse adjacency matrix:

```text
G in R^(n x n)
```

where `n` is the total number of cells across all batches.

### Within-Batch Graph

For each batch, SCALP-lite builds a symmetric k-nearest-neighbor graph. These blocks preserve local biological structure inside a batch.

Edge weights are converted from distances as:

```text
weight = 1 / (1 + distance)
```

The number of within-batch neighbors is controlled by:

```text
intra_neighbors = ceil(n_neighbors * intra_fraction)
```

### Cross-Batch Graph

For each pair of batches, SCALP-lite computes all pairwise distances and solves a linear assignment problem:

```text
minimize sum distance(left_cell_i, right_cell_assignment_i)
```

This creates one-to-one matches between cells from the two batches. If `n_inter_edges > 1`, assignment is repeated after removing previous matches, creating multiple sparse matching layers.

Assignments can be filtered with `assignment_quantile`. For example, `0.95` drops the worst 5% of assignment distances for each batch pair.

Surviving assignment edges are weighted as:

```text
weight = 1 / (1 + assignment_distance)
```

### Block Assembly

The final graph is assembled as a block matrix:

```text
G = [
  G_11  G_12  ...  G_1m
  G_21  G_22  ...  G_2m
  ...   ...   ...  ...
  G_m1  G_m2  ...  G_mm
]
```

where:

- `G_ii` is the within-batch kNN graph for batch `i`.
- `G_ij` is the cross-batch assignment graph from batch `i` to batch `j`.
- `G_ji` is the transpose of `G_ij`.

The graph is optionally symmetrized and diagonal entries are removed.

## Pseudocode

```text
function BUILD_SCALP_GRAPH(
    adata,
    rep_key = "X_pca",
    batch_key = "batch",
    n_neighbors = 15,
    intra_fraction = 0.5,
    n_inter_edges = 1,
    metric = "euclidean",
    assignment_quantile = 0.95,
    symmetrize = true
):
    validate adata has batch_key and rep_key

    batches = split cells by adata.obs[batch_key]
    X = adata.obsm[rep_key]
    X_batches = [X[cells_in_batch] for each batch]

    intra_neighbors = ceil(n_neighbors * intra_fraction)
    initialize block matrix blocks[m][m]

    for each batch i:
        blocks[i][i] = BUILD_INTRA_BATCH_GRAPH(
            X_batches[i],
            n_neighbors = intra_neighbors,
            metric = metric
        )

    for each pair of batches i < j:
        cross = BUILD_INTER_BATCH_GRAPH(
            X_batches[i],
            X_batches[j],
            n_inter_edges = n_inter_edges,
            metric = metric,
            assignment_quantile = assignment_quantile
        )
        blocks[i][j] = cross
        blocks[j][i] = transpose(cross)

    G = sparse_block_matrix(blocks)
    remove diagonal entries from G

    if symmetrize:
        G = elementwise_max(G, transpose(G))
        remove diagonal entries from G

    store graph metadata in adata.uns["scalp_lite"]["graph"]
    return G
```

```text
function BUILD_INTRA_BATCH_GRAPH(X, n_neighbors, metric):
    if X has zero or one cell:
        return empty sparse graph

    k = min(n_neighbors + 1, number_of_cells)
    neighbors, distances = nearest_neighbors(X, k, metric)

    for each cell:
        add edges to its k - 1 nearest non-self neighbors
        weight each edge as 1 / (1 + distance)

    symmetrize graph with elementwise maximum
    remove diagonal entries
    return graph
```

```text
function BUILD_INTER_BATCH_GRAPH(
    X_left,
    X_right,
    n_inter_edges,
    metric,
    assignment_quantile
):
    if either batch is empty or n_inter_edges <= 0:
        return empty sparse graph

    D = pairwise_distances(X_left, X_right, metric)
    assignments = []

    repeat n_inter_edges times:
        rows, cols = linear_sum_assignment(D)
        keep only finite assigned distances
        append assigned pairs and distances to assignments
        set assigned D[rows, cols] to infinity

    if assignment_quantile is not None:
        cutoff = quantile(assignment_distances, assignment_quantile)
        keep only assignments with distance <= cutoff

    weight each assignment as 1 / (1 + distance)
    return sparse bipartite graph from X_left to X_right
```

## Parameters

- `rep_key`: representation used for graph construction. Default: `X_pca`.
- `batch_key`: observation column defining batches. Default: `batch`.
- `n_neighbors`: total neighborhood scale. Default: `15`.
- `intra_fraction`: fraction of `n_neighbors` allocated to within-batch edges. Default: `0.5`.
- `n_inter_edges`: number of repeated assignment layers per batch pair. Default: `1`.
- `metric`: distance metric used by nearest-neighbor and assignment steps. Default: `euclidean`.
- `assignment_quantile`: upper distance quantile retained for cross-batch matches. Default: `0.95`.
- `symmetrize`: whether to make the assembled graph symmetric. Default: `true`.

## Expected Behavior

When batches share overlapping cell types, cross-batch assignment edges should connect corresponding populations and improve batch mixing. When batches contain non-overlapping states, the assignment quantile should prune the weakest matches and reduce forced alignment.

The most important tradeoff is controlled by the cross-batch filtering threshold:

- Lower `assignment_quantile`: stricter integration, fewer cross-batch edges, better protection against false matches.
- Higher `assignment_quantile`: stronger integration, more batch mixing, greater risk of over-alignment.

## Evaluation

SCALP-lite supports graph and embedding evaluation through metrics such as:

- batch mixing in embedding neighborhoods
- batch silhouette
- kNN label agreement
- label silhouette
- graph density
- runtime metadata

For a quick smoke test, the notebooks use PBMC3k with artificial batches. For a real integration benchmark, a multi-batch dataset such as scIB human pancreas is a better target.

## Limitations

SCALP-lite is designed to be readable and dependency-light, not exhaustive.

Current limitations:

- Cross-batch assignment requires pairwise distance matrices for each batch pair.
- PBMC3k is only a smoke test because artificial batches do not represent real batch effects.
- The implementation does not yet include hubness correction.
- The graph currently uses simple inverse-distance weights.
- Large atlas-scale datasets may need approximate nearest-neighbor search, chunked assignment, or subsampling.

## Practical Workflow

```text
load .h5ad
validate AnnData schema
normalize expression and select highly variable genes
ensure PCA representation exists
build SCALP-lite graph
embed graph with UMAP or spectral embedding
plot by batch and label
score embedding quality
save embedded .h5ad
```

The object-oriented entry point for this workflow is `ScalpEstimator`. Its `preprocess()` method uses Scanpy's `normalize_total`, `log1p`, `filter_genes`, and `highly_variable_genes` when Scanpy is installed, matching the standard preprocessing pattern used in the original `cellsaw` codebase. A variance-based selector is kept as a lightweight fallback.

This workflow is implemented in:

- `notebooks/01_visualize_embedding.ipynb`
- `notebooks/02_evaluate_embedding.ipynb`

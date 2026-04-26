# CLASP Whitepaper

## Summary

**CLASP** stands for **Cell integration via Linear Assignment and Sparse Pairing**.

CLASP is a compact implementation of a graph-based strategy for integrating single-cell datasets across batches, donors, technologies, time points, or related biological conditions. The central idea is to build a cell-cell graph that preserves local neighborhoods inside each batch while adding sparse, globally optimized links across batches.

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

In those harder cases, fully mixing all batches can be wrong: some cells should align, while distinct biological states should remain separate. CLASP addresses this by adding cross-batch links sparsely and globally. The Hungarian assignment step prevents one popular cell from becoming the nearest neighbor of many cells in another batch, while the distance cutoff removes weak assignments that likely represent non-overlapping biology.

## Data Model

Input is an `AnnData` object or `.h5ad` file.

Required fields:

- `adata.X`: expression matrix, needed if PCA must be computed.
- `adata.obs[batch_key]`: batch, dataset, donor, time point, or condition label.

Optional fields:

- `adata.obs[label_key]`: biological label used for evaluation and plotting.
- `adata.obsm[rep_key]`: precomputed representation, usually `X_pca`.

CLASP works on `adata.obsm[rep_key]`. If that representation is missing, the project provides `ensure_pca()` to compute it from `adata.X`.

## Algorithm

Let there be `m` batches:

$$
X = \{X_1, X_2, \ldots, X_m\}
$$

where `X_i` is the matrix of cells in batch `i`, represented in PCA or another shared embedding.

The output is a sparse adjacency matrix:

$$
G \in \mathbb{R}^{n \times n}
$$

where `n` is the total number of cells across all batches.

### Within-Batch Graph

For each batch, CLASP builds a symmetric k-nearest-neighbor graph. These blocks preserve local biological structure inside a batch. By default, nearest-neighbor search uses hubness-corrected distances rather than raw Euclidean distances, and within-batch edges are retained only when the neighbor relation is mutual.

### Hubness Correction

High-dimensional nearest-neighbor graphs often contain hubs: cells that appear as neighbors of many other cells because of geometry rather than biology. Following the CLASP paper, CLASP uses Cross-domain Similarity Local Scaling (CSLS) to correct pairwise distances before neighbor selection and cross-batch assignment.

For a distance matrix `D` between domains `A` and `B`, define the local scale of a cell as its mean distance to the `k_csls` nearest cells in the opposite domain:

$$
r_A(i) =
\frac{1}{k_{\mathrm{csls}}}
\sum_{y \in N_{k_{\mathrm{csls}}}^{B}(x_i)}
D_{iy}
$$

$$
r_B(j) =
\frac{1}{k_{\mathrm{csls}}}
\sum_{x \in N_{k_{\mathrm{csls}}}^{A}(y_j)}
D_{xj}
$$

The corrected distance is:

$$
D_{ij}^{\mathrm{CSLS}} =
2D_{ij} - r_A(i) - r_B(j)
$$

For within-batch graphs, the same formula is applied with `A = B`, excluding each cell from its own local neighborhood. For cross-batch graphs, `A` and `B` are the two batches being matched.

Edge weights are converted from distances as:

$$
w_{uv} = \frac{1}{1 + \tilde{D}_{uv}}
$$

where `tilde D` is the non-negative shifted corrected distance used only for edge weighting. Neighbor selection and assignment use the unshifted corrected distance.

CLASP can also binarize retained edges:

$$
w_{uv} = 1
$$

This mirrors the paper's implementation note that boolean graph connectivity can substantially reduce cost with limited impact on quality. The notebook defaults use binary weighting; distance weighting remains available when edge confidence should be preserved for downstream embedding.

The number of within-batch neighbors is controlled by:

$$
k_{\mathrm{intra}} =
\left\lceil
k_{\mathrm{total}} \cdot \rho_{\mathrm{intra}}
\right\rceil
$$

Rather than selecting neighbors directly by corrected distance, CLASP defaults to rank-based neighbor selection. For each cell, corrected distances are converted into row-wise ranks:

$$
R_{uv} =
\operatorname{rank}_{z}
\left(D_{uz}^{\mathrm{CSLS}}\right)(v)
$$

The neighbor score is the reciprocal rank sum:

$$
S_{uv} = R_{uv} + R_{vu}
$$

The directed neighbor proposal is then:


$$
u \rightarrow v
\quad \Longleftrightarrow \quad
v \in N_{k_{\mathrm{intra}}}^{S}(u)
$$

With mutual-neighbor filtering enabled, the retained within-batch edge set is:

$$
(u, v) \in E_{\mathrm{intra}}
\quad \Longleftrightarrow \quad
u \in N_{k_{\mathrm{intra}}}(v)
\ \mathrm{and}\
v \in N_{k_{\mathrm{intra}}}(u)
$$

Mutual neighbors reduce one-sided high-dimensional nearest-neighbor artifacts and are therefore the default.

### Cross-Batch Graph

For each pair of batches, CLASP computes all pairwise distances, applies the configured hubness correction, and solves a linear assignment problem:

$$
\min_{\pi}
\sum_{i}
D_{i,\pi(i)}^{\mathrm{CSLS}}
$$

This creates one-to-one matches between cells from the two batches. If `n_inter_edges > 1`, assignment is repeated after removing previous matches, creating multiple sparse matching layers.

Assignments can be filtered with `assignment_quantile`. For example, `0.95` drops the worst 5% of assignment distances for each batch pair.

Surviving assignment edges are weighted as:

$$
w_{i,\pi(i)} =
\frac{1}{1 + \tilde{D}_{i,\pi(i)}^{\mathrm{CSLS}}}
$$

### Block Assembly

The final graph is assembled as a block matrix:

$$
G =
\begin{bmatrix}
G_{11} & G_{12} & \cdots & G_{1m} \\
G_{21} & G_{22} & \cdots & G_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
G_{m1} & G_{m2} & \cdots & G_{mm}
\end{bmatrix}
$$

where:

- `G_ii` is the within-batch kNN graph for batch `i`.
- `G_ij` is the cross-batch assignment graph from batch `i` to batch `j`.
- `G_ji` is the transpose of `G_ij`.

The graph is optionally symmetrized and diagonal entries are removed.

## Pseudocode

```text
function BUILD_CLASP_GRAPH(
    adata,
    rep_key = "X_pca",
    batch_key = "batch",
    n_neighbors = 15,
    intra_fraction = 0.5,
    n_inter_edges = 1,
    metric = "euclidean",
    assignment_quantile = 0.95,
    hubness_correction = "csls",
    hubness_k = 10,
    edge_weighting = "binary",
    mutual_neighbors = true,
    neighbor_mode = "rank",
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
            metric = metric,
            hubness_correction = hubness_correction,
            hubness_k = hubness_k,
            edge_weighting = edge_weighting,
            mutual_neighbors = mutual_neighbors,
            neighbor_mode = neighbor_mode
        )

    for each pair of batches i < j:
        cross = BUILD_INTER_BATCH_GRAPH(
            X_batches[i],
            X_batches[j],
            n_inter_edges = n_inter_edges,
            metric = metric,
            assignment_quantile = assignment_quantile,
            hubness_correction = hubness_correction,
            hubness_k = hubness_k,
            edge_weighting = edge_weighting
        )
        blocks[i][j] = cross
        blocks[j][i] = transpose(cross)

    G = sparse_block_matrix(blocks)
    remove diagonal entries from G

    if symmetrize:
        G = elementwise_max(G, transpose(G))
        remove diagonal entries from G

    store graph metadata in adata.uns["clasp"]["graph"]
    return G
```

```text
function CSLS_DISTANCES(D, hubness_k, exclude_self = false):
    if exclude_self:
        set diagonal of D to infinity

    row_scale = mean k smallest values in each row
    column_scale = mean k smallest values in each column
    return 2 * D - row_scale[:, None] - column_scale[None, :]
```

```text
function BUILD_INTRA_BATCH_GRAPH(
    X,
    n_neighbors,
    metric,
    hubness_correction,
    hubness_k,
    edge_weighting,
    mutual_neighbors,
    neighbor_mode
):
    if X has zero or one cell:
        return empty sparse graph

    k = min(n_neighbors + 1, number_of_cells)
    distances = pairwise_distances(X, X, metric)

    if hubness_correction == "csls":
        distances = CSLS_DISTANCES(distances, hubness_k, exclude_self = true)

    if neighbor_mode == "rank":
        scores = reciprocal_rank_scores(distances)
    else:
        scores = distances

    neighbors = rowwise_k_smallest_non_self(scores, k - 1)

    for each cell:
        add edges to its corrected nearest non-self neighbors
        if edge_weighting == "binary":
            weight each edge as 1
        else:
            weight each edge as 1 / (1 + shifted_corrected_distance)

    if mutual_neighbors:
        keep only reciprocal directed edges
    else:
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
    assignment_quantile,
    hubness_correction,
    hubness_k,
    edge_weighting
):
    if either batch is empty or n_inter_edges <= 0:
        return empty sparse graph

    D = pairwise_distances(X_left, X_right, metric)
    if hubness_correction == "csls":
        D = CSLS_DISTANCES(D, hubness_k)
    assignments = []

    repeat n_inter_edges times:
        rows, cols = linear_sum_assignment(D)
        keep only finite assigned distances
        append assigned pairs and distances to assignments
        set assigned D[rows, cols] to infinity

    if assignment_quantile is not None:
        cutoff = quantile(assignment_distances, assignment_quantile)
        keep only assignments with distance <= cutoff

    if edge_weighting == "binary":
        weight each assignment as 1
    else:
        weight each assignment as 1 / (1 + shifted_corrected_distance)
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
- `hubness_correction`: distance correction applied before neighbor selection and assignment. Options: `csls`, `none`. Default: `csls`.
- `hubness_k`: local neighborhood size used by CSLS. Default: `10`.
- `edge_weighting`: how retained graph edges are weighted. Options: `binary`, `distance`. Notebook default: `binary`; library default: `distance`.
- `mutual_neighbors`: whether within-batch kNN edges must be reciprocal. Default: `true`.
- `neighbor_mode`: score used for within-batch kNN selection. Options: `rank`, `distance`. Default: `rank`.
- `symmetrize`: whether to make the assembled graph symmetric. Default: `true`.

## Expected Behavior

When batches share overlapping cell types, cross-batch assignment edges should connect corresponding populations and improve batch mixing. When batches contain non-overlapping states, the assignment quantile should prune the weakest matches and reduce forced alignment.

The most important tradeoff is controlled by the cross-batch filtering threshold:

- Lower `assignment_quantile`: stricter integration, fewer cross-batch edges, better protection against false matches.
- Higher `assignment_quantile`: stronger integration, more batch mixing, greater risk of over-alignment.

## Evaluation

CLASP supports graph and embedding evaluation through metrics such as:

- batch mixing in embedding neighborhoods
- batch silhouette
- kNN label agreement
- label silhouette
- graph density
- runtime metadata

For a quick smoke test, the notebooks use PBMC3k with artificial batches. For a real integration benchmark, a multi-batch dataset such as scIB human pancreas is a better target.

## Limitations

CLASP is designed to be readable and dependency-light, not exhaustive.

Current limitations:

- Cross-batch assignment requires pairwise distance matrices for each batch pair.
- PBMC3k is only a smoke test because artificial batches do not represent real batch effects.
- CSLS correction currently requires dense pairwise distance matrices.
- Binary graph weighting is faster and paper-compatible, but it discards edge confidence.
- Large atlas-scale datasets may need approximate nearest-neighbor search, chunked assignment, or subsampling.

## Practical Workflow

```text
load .h5ad
validate AnnData schema
normalize expression and select highly variable genes
ensure PCA representation exists
build CLASP graph
embed graph with UMAP or spectral embedding
plot by batch and label
score embedding quality
save embedded .h5ad
```

For paired batch/label plots, cells are drawn in a reproducible random order while keeping coordinates and labels synchronized. This avoids a misleading overlay when the AnnData rows are ordered by batch or cell type and many points occupy a crowded region.

The object-oriented entry point for this workflow is `ClaspEstimator`. Its `preprocess()` method uses Scanpy's `normalize_total`, `log1p`, `filter_genes`, and `highly_variable_genes` when Scanpy is installed, matching the standard preprocessing pattern used in the original `cellsaw` codebase. A variance-based selector is kept as a lightweight fallback.

This workflow is implemented in:

- `notebooks/01_latent_bayesopt.ipynb`
- `notebooks/02_visualize_embedding.ipynb`
- `notebooks/03_evaluate_embedding.ipynb`

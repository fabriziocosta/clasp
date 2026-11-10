"""Small reusable helpers for the CLASP plus IMSTE single-cell notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_top_de_genes(X, obs, label_key: str, n_top_genes: int | None):
    """Return X restricted to genes ranked by Scanpy Wilcoxon DE across labels."""
    if n_top_genes is None:
        return X, None
    if isinstance(n_top_genes, bool) or not isinstance(n_top_genes, (int, np.integer)) or n_top_genes < 1:
        raise ValueError("n_top_genes must be None or a positive integer")
    if label_key not in obs:
        raise KeyError(f"Label column {label_key!r} is not present in obs")
    if n_top_genes >= X.shape[1]:
        return X, np.arange(X.shape[1], dtype=int)

    import anndata as ad
    import scanpy as sc

    label_values = obs[label_key].astype("string")
    label_counts = label_values.value_counts(dropna=True)
    rankable_labels = label_counts[label_counts >= 2].index
    rank_rows = (label_values.notna() & label_values.isin(rankable_labels)).to_numpy()
    de_obs = pd.DataFrame(
        {label_key: label_values.loc[rank_rows].astype("category")},
        index=obs.index[rank_rows],
    )
    if de_obs[label_key].nunique() < 2:
        raise ValueError(
            f"Need at least two non-missing {label_key!r} groups with two or more cells "
            "for DE ranking"
        )

    gene_names = np.asarray([f"gene_{index}" for index in range(X.shape[1])])
    de_adata = ad.AnnData(X=X[rank_rows].copy(), obs=de_obs)
    de_adata.var_names = gene_names
    sc.pp.normalize_total(de_adata, target_sum=10_000)
    sc.pp.log1p(de_adata)
    sc.tl.rank_genes_groups(
        de_adata,
        groupby=label_key,
        method="wilcoxon",
        reference="rest",
        use_raw=False,
    )
    results = sc.get.rank_genes_groups_df(de_adata, group=None)
    scores = (
        results.dropna(subset=["names", "pvals_adj", "logfoldchanges"])
        .assign(abs_logfoldchange=lambda frame: frame["logfoldchanges"].abs())
        .groupby("names", as_index=False)
        .agg(
            best_adjusted_p_value=("pvals_adj", "min"),
            strongest_abs_logfoldchange=("abs_logfoldchange", "max"),
        )
        .sort_values(
            ["best_adjusted_p_value", "strongest_abs_logfoldchange"],
            ascending=[True, False],
        )
    )
    selected = scores.head(int(n_top_genes))["names"].to_numpy(dtype=str)
    indices = np.fromiter(
        (int(name.removeprefix("gene_")) for name in selected), dtype=int
    )
    return X[:, indices], indices


def infer_batch_key(obs, candidates=None) -> str:
    """Choose a useful batch column or add a single-batch marker when absent."""
    candidates = candidates or (
        "batch", "batch_id", "study", "tech", "technology", "chemistry",
        "sample_id", "sample", "donor", "orig.ident",
    )
    for key in candidates:
        if key in obs and obs[key].nunique(dropna=True) > 1:
            obs[key] = obs[key].astype("string").fillna("Missing").astype("category")
            return key
    key = "__clasp_batch"
    obs[key] = pd.Categorical(np.repeat("all_cells", len(obs)))
    return key


def fit_clasp_umap_imste(
    X,
    obs,
    *,
    label_key: str,
    batch_key: str,
    clasp_params: dict,
    preprocess_params: dict,
    umap_params: dict,
    imste_params: dict,
    random_state: int = 42,
    imste_embedder_class=None,
):
    """Preprocess cells, build an integration or within-batch graph, and embed it."""
    import anndata as ad
    from clasp import ClaspEstimator, ClaspGraphEmbedder

    if imste_embedder_class is None:
        from imste import IteratedMinimumSpanningTreeEmbedder as imste_embedder_class

    observations = obs.copy()
    if batch_key not in observations:
        raise KeyError(f"Batch column {batch_key!r} is not present in obs")
    adata = ad.AnnData(X=X, obs=observations)
    adata.var_names = [f"feature_{index}" for index in range(adata.n_vars)]

    pca_components = clasp_params.get("pca_components", 50)
    use_pca = pca_components is not None
    rep_key = "X_pca" if use_pca else "X_clasp_raw"
    if not use_pca:
        # Keep the selected expression matrix as the graph representation.
        adata.obsm[rep_key] = np.asarray(X)

    model_keys = {
        "pca_components", "n_neighbors", "intra_fraction", "n_inter_edges",
        "metric", "assignment_quantile", "hubness_correction", "hubness_k",
        "rank_correction", "edge_weighting", "inter_edge_mode", "mutual_neighbors",
        "neighbor_mode", "symmetrize",
    }
    model_params = {
        key: clasp_params[key]
        for key in model_keys
        if key in clasp_params and key != "pca_components"
    }
    effective_pca_components = int(pca_components) if use_pca else 50
    clasp = ClaspEstimator(
        rep_key=rep_key,
        batch_key=batch_key,
        label_key=label_key,
        n_components=effective_pca_components,
        random_state=random_state,
        **model_params,
    )
    if use_pca:
        processed = clasp.preprocess(adata, copy=False, **preprocess_params)
    else:
        processed = adata
    graph_params = {key: value for key, value in clasp_params.items() if key in {
        "n_neighbors", "intra_fraction", "n_inter_edges", "metric",
        "assignment_quantile", "hubness_correction", "hubness_k", "rank_correction",
        "edge_weighting", "inter_edge_mode", "mutual_neighbors", "neighbor_mode", "symmetrize",
    }}
    has_multiple_batches = processed.obs[batch_key].nunique(dropna=False) > 1

    def build_single_batch_graph(*, edge_weighting):
        from clasp.graph.intra import build_intra_batch_graph

        return build_intra_batch_graph(
            np.asarray(processed.obsm[rep_key]),
            n_neighbors=graph_params.get("n_neighbors", clasp.n_neighbors),
            metric=graph_params.get("metric", clasp.metric),
            hubness_correction=graph_params.get("hubness_correction", clasp.hubness_correction),
            hubness_k=graph_params.get("hubness_k", clasp.hubness_k),
            rank_correction=graph_params.get("rank_correction", clasp.rank_correction),
            edge_weighting=edge_weighting,
            mutual_neighbors=graph_params.get("mutual_neighbors", clasp.mutual_neighbors),
            neighbor_mode=graph_params.get("neighbor_mode", clasp.neighbor_mode),
        )

    if has_multiple_batches:
        graph = clasp.data_to_graph(processed, **graph_params)
    else:
        # A single-batch dataset has no cross-batch assignment step; use its
        # within-batch kNN graph for both graph embedders.
        graph = build_single_batch_graph(
            edge_weighting=graph_params.get("edge_weighting", clasp.edge_weighting)
        )

    umap_options = dict(umap_params)
    # Graph UMAP always uses precomputed distances; embed_graph supplies this.
    umap_options.pop("metric", None)
    umap_embedder = ClaspGraphEmbedder(
        method="umap",
        n_components=2,
        random_state=random_state,
        kwargs=umap_options,
    ).fit_graph(graph)

    imste_graph = graph
    if graph_params.get("edge_weighting", "distance") != "distance":
        if has_multiple_batches:
            imste_graph = clasp.data_to_graph(
                processed, **{**graph_params, "edge_weighting": "distance"}
            )
        else:
            imste_graph = build_single_batch_graph(edge_weighting="distance")
    imste_embedder = imste_embedder_class(**imste_params)
    imste_model = ClaspEstimator(
        rep_key=rep_key,
        batch_key=batch_key,
        label_key=label_key,
        random_state=random_state,
        edge_weighting="distance",
        embedder=imste_embedder,
    )
    imste_embedding = imste_model.graph_to_embeddings(
        imste_graph,
        init_features=processed.obsm[rep_key],
    )

    return processed, graph, umap_embedder.embedding_, imste_embedding, clasp

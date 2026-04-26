from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from clasp.embedding import embed_graph
from clasp.graph import GraphParams, build_clasp_graph
from clasp.io import read_h5ad, save_h5ad, validate_adata
from clasp.plotting import plot_embedding_pair
from clasp.preprocessing import ensure_pca


def _optional_scanpy():
    try:
        import scanpy as sc
    except ImportError:
        return None
    return sc


def _looks_log_normalized(matrix) -> bool:
    sample = matrix.data if sparse.issparse(matrix) else np.asarray(matrix).ravel()
    if sample.size == 0:
        return False
    sample = sample[: min(sample.size, 100_000)]
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return False
    mostly_integer = np.mean(np.isclose(finite, np.round(finite))) > 0.98
    return bool(finite.max() < 50 and not mostly_integer)


def _matrix_variance(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        mean = np.asarray(matrix.mean(axis=0)).ravel()
        mean_sq = np.asarray(matrix.multiply(matrix).mean(axis=0)).ravel()
        return mean_sq - mean**2
    return np.asarray(matrix).var(axis=0)


def _stratified_indices(values, *, max_cells: int, random_state: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    values = np.asarray(values)
    unique = list(dict.fromkeys(values))
    groups = [np.flatnonzero(values == value) for value in unique]

    quotas = np.array([len(group) for group in groups], dtype=float)
    quotas = quotas / quotas.sum() * max_cells
    counts = np.floor(quotas).astype(int)
    counts = np.minimum(counts, [len(group) for group in groups])

    remainder = max_cells - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(quotas - counts))
        for idx in order:
            if remainder == 0:
                break
            if counts[idx] < len(groups[idx]):
                counts[idx] += 1
                remainder -= 1

    selected = [rng.choice(group, size=count, replace=False) for group, count in zip(groups, counts) if count > 0]
    if not selected:
        return np.array([], dtype=int)
    return np.sort(np.concatenate(selected))


@dataclass
class PreprocessTuneParams:
    """Preprocessing defaults, fixed overrides, and searchable parameters."""

    base: dict = field(default_factory=dict)
    fixed: dict = field(default_factory=dict)
    search_space: dict = field(default_factory=dict)


@dataclass
class EstimatorTuneParams:
    """Estimator defaults and searchable estimator constructor parameters."""

    base: dict = field(default_factory=dict)
    search_space: dict = field(default_factory=dict)


@dataclass
class GraphTuneParams:
    """Graph defaults and searchable graph-construction parameters."""

    base: dict = field(default_factory=dict)
    search_space: dict = field(default_factory=dict)


@dataclass
class LatentBOTuneParams:
    """Latent Bayesian optimization run settings."""

    n_initial: int = 6
    latent_dim: int = 3
    n_iterations: int = 4
    embedding_model: str = "pca"
    acquisition: str = "ei"
    batch_size: int = 1

    def to_kwargs(self) -> dict:
        return {
            "n_initial": self.n_initial,
            "latent_dim": self.latent_dim,
            "n_iterations": self.n_iterations,
            "embedding_model": self.embedding_model,
            "acquisition": self.acquisition,
            "batch_size": self.batch_size,
        }


@dataclass
class TuneParams:
    """Complete CLASP tuning configuration."""

    preprocess: PreprocessTuneParams
    estimator: EstimatorTuneParams
    graph: GraphTuneParams
    pca_bo: LatentBOTuneParams = field(default_factory=LatentBOTuneParams)
    gplvm_bo: LatentBOTuneParams = field(default_factory=lambda: LatentBOTuneParams(embedding_model="gplvm"))
    run_gplvm_refinement: bool = False
    compact_radii: dict = field(default_factory=dict)
    fix_categoricals: bool = True
    random_state: int = 0
    embedding_method: str = "umap"
    embedding_epochs: int | None = 60
    invalid_score: float = -1e9
    label_weight: float = 1.0
    batch_weight: float = 0.25
    density_weight: float = -0.05


@dataclass
class TuneResult:
    """Best tuned parameters and optimization traces."""

    preprocess_params: dict
    estimator_params: dict
    graph_params: dict
    best_score: float
    best_model: str
    optimization_results: dict
    metadata: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "preprocess_params": self.preprocess_params,
            "estimator_params": self.estimator_params,
            "graph_params": self.graph_params,
            "metadata": {
                **self.metadata,
                "best_score": self.best_score,
                "best_model": self.best_model,
            },
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
        return path


def _optimization_search_space(
    preprocess: PreprocessTuneParams,
    estimator: EstimatorTuneParams,
    graph: GraphTuneParams,
) -> dict:
    return {**preprocess.search_space, **estimator.search_space, **graph.search_space}


def _split_tuned_params(params: dict, config: TuneParams) -> tuple[dict, dict, dict]:
    preprocess_params = {
        **config.preprocess.base,
        **config.preprocess.fixed,
        **{key: params[key] for key in config.preprocess.search_space if key in params},
    }
    estimator_params = {
        **config.estimator.base,
        **{key: params[key] for key in config.estimator.search_space if key in params},
    }
    graph_params = {
        **config.graph.base,
        **{key: params[key] for key in config.graph.search_space if key in params},
    }
    return preprocess_params, estimator_params, graph_params


def _compact_bounds(best: dict, name: str, low, high, radius, *, integer: bool = False) -> list:
    center = best[name]
    new_low = max(low, center - radius)
    new_high = min(high, center + radius)
    if integer:
        new_low = int(round(new_low))
        new_high = int(round(new_high))
        if new_low == new_high:
            new_low = max(low, new_low - 1)
            new_high = min(high, new_high + 1)
    return [new_low, new_high]


def _compact_search_space(search_space: dict, best_params: dict, radii: dict, *, fix_categoricals: bool) -> dict:
    compact = {}
    for name, spec in search_space.items():
        if spec["type"] == "categorical":
            compact[name] = {"type": "categorical", "values": [best_params[name]]} if fix_categoricals else spec.copy()
            continue
        low, high = spec["bounds"]
        compact[name] = {
            **spec,
            "bounds": _compact_bounds(best_params, name, low, high, radii[name], integer=spec["type"] == "int"),
        }
    return compact


@dataclass
class ClaspEstimator:
    """Object-oriented facade for the CLASP workflow."""

    rep_key: str = "X_pca"
    batch_key: str = "batch"
    label_key: str = "label"
    n_neighbors: int = 15
    intra_fraction: float = 0.5
    n_inter_edges: int = 1
    metric: str = "euclidean"
    assignment_quantile: float | None = 0.95
    hubness_correction: str = "csls"
    hubness_k: int = 10
    rank_correction: bool = True
    edge_weighting: str = "distance"
    mutual_neighbors: bool = True
    neighbor_mode: str = "distance"
    symmetrize: bool = True
    n_components: int = 40
    random_state: int = 0
    embedding_method: str = "auto"
    embedding_components: int = 2

    def to_data(self, path: str | Path) -> ad.AnnData:
        """Read an `.h5ad` file and return an AnnData object."""
        return read_h5ad(path)

    def input(self, path: str | Path) -> ad.AnnData:
        """Alias for `to_data`."""
        return self.to_data(path)

    def save(self, adata: ad.AnnData, path: str | Path, *, compression: str | None = "gzip") -> None:
        """Write an AnnData object to an `.h5ad` file."""
        save_h5ad(adata, path, compression=compression)

    def preprocess(
        self,
        adata: ad.AnnData,
        *,
        n_top_genes: int | None = 2000,
        max_cells: int | None = None,
        min_cell_genes: int | None = None,
        min_gene_counts: int = 3,
        normalize: bool | str = "auto",
        target_sum: float = 1e4,
        log1p: bool = True,
        hvg_flavor: str = "cell_ranger",
        hvg_batch_key: str | None = None,
        hvg_fallback: str = "variance",
        create_artificial_batch: bool = False,
        artificial_batch_count: int = 3,
        label_candidates: tuple[str, ...] = (
            "clusters_coarse",
            "cell_type",
            "lineages",
            "clusters_fine",
            "clusters",
            "leiden",
            "louvain",
        ),
        copy: bool = True,
    ) -> ad.AnnData:
        """Prepare AnnData for graph construction.

        Parameters control, in order: optional artificial batch/label setup,
        cell and gene filtering, expression normalization, HVG selection,
        optional cell subsampling, copying behavior, and PCA creation.
        `label_candidates` is tried in order when `obs[self.label_key]` is
        absent. `hvg_fallback` controls Scanpy HVG failures and can be
        "variance" or "error".
        """
        if hvg_fallback not in {"variance", "error"}:
            raise ValueError("hvg_fallback must be one of: 'variance', 'error'.")
        result = adata.copy() if copy else adata
        sc = _optional_scanpy()
        changed_expression = False

        if self.batch_key not in result.obs and create_artificial_batch:
            if artificial_batch_count < 1:
                raise ValueError("artificial_batch_count must be >= 1.")
            result.obs[self.batch_key] = pd.Categorical(
                [f"split_{i % artificial_batch_count}" for i in range(result.n_obs)]
            )

        if self.label_key not in result.obs:
            for candidate in label_candidates:
                if candidate in result.obs:
                    result.obs[self.label_key] = result.obs[candidate].astype("category")
                    break

        if min_cell_genes is not None and min_cell_genes > 0:
            if sc is None:
                raise ImportError("Cell filtering requires optional dependency `scanpy`.")
            cell_mask, _ = sc.pp.filter_cells(result, min_genes=min_cell_genes, inplace=False)
            if not np.all(cell_mask):
                result = result[cell_mask].copy()

        if min_gene_counts > 0 and sc is not None and result.n_vars > 0:
            gene_mask, _ = sc.pp.filter_genes(result, min_counts=min_gene_counts, inplace=False)
            if not np.all(gene_mask):
                result = result[:, gene_mask].copy()
                changed_expression = True

        should_normalize = normalize
        if normalize == "auto":
            should_normalize = not bool(result.uns.get("normlog", False)) and not _looks_log_normalized(result.X)
        if should_normalize:
            if sc is None:
                raise ImportError("Expression normalization requires optional dependency `scanpy`.")
            sc.pp.normalize_total(result, target_sum=target_sum)
            if log1p:
                sc.pp.log1p(result)
            result.uns["normlog"] = True
            changed_expression = True

        if n_top_genes is not None and result.n_vars > n_top_genes:
            if n_top_genes < 1:
                raise ValueError("n_top_genes must be >= 1 or None.")
            hvg_failed = False
            if sc is not None and hvg_flavor != "variance":
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="`n_top_genes` > number of normalized dispersions.*")
                        warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)
                        hvg = sc.pp.highly_variable_genes(
                            result,
                            n_top_genes=n_top_genes,
                            flavor=hvg_flavor,
                            batch_key=hvg_batch_key,
                            inplace=False,
                        )
                    selected = np.asarray(hvg["highly_variable"], dtype=bool)
                    score_column = "dispersions_norm" if "dispersions_norm" in hvg else "variances_norm"
                    scores = hvg[score_column].fillna(0).to_numpy() if score_column in hvg else np.zeros(result.n_vars)
                    result.var["clasp_hvg_score"] = scores
                except Exception as exc:
                    if hvg_fallback == "error":
                        raise
                    warnings.warn(
                        "Scanpy highly_variable_genes failed; falling back to variance-based gene selection. "
                        f"Original error: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    selected = np.zeros(result.n_vars, dtype=bool)
                    hvg_failed = True
            if sc is None or hvg_flavor == "variance" or not np.any(selected):
                if sc is not None and hvg_flavor != "variance" and hvg_fallback == "error":
                    raise RuntimeError("Scanpy highly_variable_genes did not select any genes.")
                if sc is not None and hvg_flavor != "variance" and not np.any(selected) and not hvg_failed:
                    warnings.warn(
                        "Scanpy highly_variable_genes selected no genes; falling back to variance-based gene selection.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                variances = _matrix_variance(result.X)
                selected = np.zeros(result.n_vars, dtype=bool)
                selected[np.argsort(variances)[::-1][:n_top_genes]] = True
                result.var["clasp_hvg_score"] = variances
            result = result[:, selected].copy()
            result.var["clasp_selected"] = True
            changed_expression = True

        if max_cells is not None and result.n_obs > max_cells:
            if max_cells < 1:
                raise ValueError("max_cells must be >= 1 or None.")
            if self.batch_key in result.obs:
                indices = _stratified_indices(result.obs[self.batch_key], max_cells=max_cells, random_state=self.random_state)
            else:
                rng = np.random.default_rng(self.random_state)
                indices = np.sort(rng.choice(result.n_obs, size=max_cells, replace=False))
            result = result[indices].copy()

        if changed_expression and self.rep_key in result.obsm:
            del result.obsm[self.rep_key]
        ensure_pca(result, rep_key=self.rep_key, n_components=self.n_components, random_state=self.random_state)
        return result

    def data_to_graph(
        self,
        adata: ad.AnnData,
        *,
        rep_key: str | None = None,
        batch_key: str | None = None,
        n_neighbors: int | None = None,
        intra_fraction: float | None = None,
        n_inter_edges: int | None = None,
        metric: str | None = None,
        assignment_quantile: float | None = None,
        hubness_correction: str | None = None,
        hubness_k: int | None = None,
        rank_correction: bool | None = None,
        edge_weighting: str | None = None,
        mutual_neighbors: bool | None = None,
        neighbor_mode: str | None = None,
        symmetrize: bool | None = None,
    ) -> sparse.csr_matrix:
        """Build and return the CLASP graph for an AnnData object."""
        rep_key = self.rep_key if rep_key is None else rep_key
        batch_key = self.batch_key if batch_key is None else batch_key
        n_neighbors = self.n_neighbors if n_neighbors is None else n_neighbors
        intra_fraction = self.intra_fraction if intra_fraction is None else intra_fraction
        n_inter_edges = self.n_inter_edges if n_inter_edges is None else n_inter_edges
        metric = self.metric if metric is None else metric
        assignment_quantile = self.assignment_quantile if assignment_quantile is None else assignment_quantile
        hubness_correction = self.hubness_correction if hubness_correction is None else hubness_correction
        hubness_k = self.hubness_k if hubness_k is None else hubness_k
        rank_correction = self.rank_correction if rank_correction is None else rank_correction
        edge_weighting = self.edge_weighting if edge_weighting is None else edge_weighting
        mutual_neighbors = self.mutual_neighbors if mutual_neighbors is None else mutual_neighbors
        neighbor_mode = self.neighbor_mode if neighbor_mode is None else neighbor_mode
        symmetrize = self.symmetrize if symmetrize is None else symmetrize

        params = GraphParams(
            n_neighbors=n_neighbors,
            intra_fraction=intra_fraction,
            n_inter_edges=n_inter_edges,
            metric=metric,
            assignment_quantile=assignment_quantile,
            hubness_correction=hubness_correction,
            hubness_k=hubness_k,
            rank_correction=rank_correction,
            edge_weighting=edge_weighting,
            mutual_neighbors=mutual_neighbors,
            neighbor_mode=neighbor_mode,
            symmetrize=symmetrize,
        )

        validate_adata(adata, batch_key=batch_key, rep_key=rep_key, require_rep=True)
        return build_clasp_graph(
            adata,
            rep_key=rep_key,
            batch_key=batch_key,
            n_neighbors=params.n_neighbors,
            intra_fraction=params.intra_fraction,
            n_inter_edges=params.n_inter_edges,
            metric=params.metric,
            assignment_quantile=params.assignment_quantile,
            hubness_correction=params.hubness_correction,
            hubness_k=params.hubness_k,
            rank_correction=params.rank_correction,
            edge_weighting=params.edge_weighting,
            mutual_neighbors=params.mutual_neighbors,
            neighbor_mode=params.neighbor_mode,
            symmetrize=params.symmetrize,
        )

    def graph_to_vector(
        self,
        graph: sparse.spmatrix,
        *,
        method: str | None = None,
        n_components: int | None = None,
        random_state: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Embed a graph and return low-dimensional vectors."""
        method = self.embedding_method if method is None else method
        n_components = self.embedding_components if n_components is None else n_components
        random_state = self.random_state if random_state is None else random_state
        return embed_graph(
            graph,
            method=method,
            n_components=n_components,
            random_state=random_state,
            **kwargs,
        )

    def embed(
        self,
        adata: ad.AnnData,
        *,
        rep_key: str | None = None,
        batch_key: str | None = None,
        n_neighbors: int | None = None,
        intra_fraction: float | None = None,
        n_inter_edges: int | None = None,
        metric: str | None = None,
        assignment_quantile: float | None = None,
        hubness_correction: str | None = None,
        hubness_k: int | None = None,
        rank_correction: bool | None = None,
        edge_weighting: str | None = None,
        mutual_neighbors: bool | None = None,
        neighbor_mode: str | None = None,
        symmetrize: bool | None = None,
        embedding_method: str | None = None,
        embedding_components: int | None = None,
        embedding_random_state: int | None = None,
        **embedding_kwargs,
    ) -> np.ndarray:
        """Build the graph from AnnData and return embedding vectors."""
        graph = self.data_to_graph(
            adata,
            rep_key=rep_key,
            batch_key=batch_key,
            n_neighbors=n_neighbors,
            intra_fraction=intra_fraction,
            n_inter_edges=n_inter_edges,
            metric=metric,
            assignment_quantile=assignment_quantile,
            hubness_correction=hubness_correction,
            hubness_k=hubness_k,
            rank_correction=rank_correction,
            edge_weighting=edge_weighting,
            mutual_neighbors=mutual_neighbors,
            neighbor_mode=neighbor_mode,
            symmetrize=symmetrize,
        )
        return self.graph_to_vector(
            graph,
            method=embedding_method,
            n_components=embedding_components,
            random_state=embedding_random_state,
            **embedding_kwargs,
        )

    def tune(self, adata: ad.AnnData, params: TuneParams) -> TuneResult:
        """Tune preprocessing, estimator, and graph parameters with latent BO."""
        from clasp.metrics import score_embedding
        from clasp.optimization import latent_bayesopt

        search_space = _optimization_search_space(params.preprocess, params.estimator, params.graph)

        def objective(candidate: dict) -> float:
            trial_preprocess, trial_estimator_params, trial_graph = _split_tuned_params(candidate, params)
            trial_estimator = type(self)(
                rep_key=self.rep_key,
                batch_key=self.batch_key,
                label_key=self.label_key,
                random_state=params.random_state,
                **trial_estimator_params,
            )
            try:
                trial = trial_estimator.preprocess(adata, **trial_preprocess)
                embedding_kwargs = {
                    "embedding_method": params.embedding_method,
                    "embedding_random_state": params.random_state,
                }
                if params.embedding_epochs is not None and params.embedding_method in {"auto", "umap"}:
                    embedding_kwargs["n_epochs"] = params.embedding_epochs
                trial.obsm["X_clasp"] = trial_estimator.embed(trial, **trial_graph, **embedding_kwargs)
                scores = score_embedding(
                    trial,
                    embedding_key="X_clasp",
                    batch_key=self.batch_key,
                    label_key=self.label_key if self.label_key in trial.obs else None,
                ).iloc[0]
            except Exception as exc:
                print(
                    "failed params="
                    f"preprocess={trial_preprocess}, estimator={trial_estimator_params}, graph={trial_graph}: {exc}"
                )
                return float(params.invalid_score)

            label_agreement = float(scores.get("knn_label_agreement", 0.0))
            batch_mixing = float(scores.get("batch_mixing", 0.0))
            graph_density = float(scores.get("graph_density", 0.0))
            label_agreement = 0.0 if not np.isfinite(label_agreement) else label_agreement
            batch_mixing = 0.0 if not np.isfinite(batch_mixing) else batch_mixing
            graph_density = 0.0 if not np.isfinite(graph_density) else graph_density
            return float(
                params.label_weight * label_agreement
                + params.batch_weight * batch_mixing
                + params.density_weight * graph_density
            )

        pca_result = latent_bayesopt(
            objective,
            search_space,
            **params.pca_bo.to_kwargs(),
            random_state=params.random_state,
            invalid_score=params.invalid_score,
        )
        optimization_results = {"pca": pca_result}

        if params.run_gplvm_refinement:
            compact = _compact_search_space(
                search_space,
                pca_result["best_params"],
                params.compact_radii,
                fix_categoricals=params.fix_categoricals,
            )
            optimization_results["gplvm"] = latent_bayesopt(
                objective,
                compact,
                **params.gplvm_bo.to_kwargs(),
                random_state=params.random_state + 1,
                invalid_score=params.invalid_score,
            )

        best_model, best_result = max(optimization_results.items(), key=lambda item: item[1]["best_score"])
        preprocess_params, estimator_params, graph_params = _split_tuned_params(best_result["best_params"], params)
        return TuneResult(
            preprocess_params=preprocess_params,
            estimator_params=estimator_params,
            graph_params=graph_params,
            best_score=float(best_result["best_score"]),
            best_model=best_model,
            optimization_results=optimization_results,
            metadata={"random_state": params.random_state},
        )

    def plot(
        self,
        adata: ad.AnnData,
        *,
        embedding_key: str = "X_clasp",
        batch_key: str | None = None,
        label_key: str | None = None,
        **kwargs,
    ):
        """Plot the paired batch/label embedding view."""
        return plot_embedding_pair(
            adata,
            embedding_key=embedding_key,
            batch_key=self.batch_key if batch_key is None else batch_key,
            label_key=self.label_key if label_key is None else label_key,
            **kwargs,
        )

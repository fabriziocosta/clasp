from __future__ import annotations

from pathlib import Path
import hashlib
import json
import shutil
import urllib.request

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scalp_lite.estimator import ScalpEstimator
from scalp_lite.metrics import score_embedding


DATASET_REGISTRY = {
    "zebrafish": {
        "input": "data/cellrank-zebrafish.h5ad",
        "embedded": "data/cellrank-zebrafish-scalp.h5ad",
        "batch_key": "Stage",
        "label_key": "lineages",
        "preprocess": {"normalize": False, "hvg_flavor": "variance", "create_artificial_batch": False},
        "graph": {
            "n_neighbors": 30,
            "intra_fraction": 0.5,
            "n_inter_edges": 1,
            "assignment_quantile": 0.75,
            "hubness_correction": "csls",
            "hubness_k": 10,
            "rank_correction": True,
            "edge_weighting": "distance",
            "mutual_neighbors": False,
            "neighbor_mode": "distance",
            "symmetrize": True,
        },
    },
    "pancreas": {
        "input": "data/pancreas_normalized.h5ad",
        "embedded": "data/pancreas_normalized-scalp.h5ad",
        "batch_key": "study",
        "label_key": "cell_type",
        "preprocess": {
            "normalize": False,
            "hvg_flavor": "variance",
            "min_gene_counts": 0,
            "max_cells": 6000,
            "create_artificial_batch": False,
        },
        "graph": {
            "n_neighbors": 20,
            "intra_fraction": 0.5,
            "n_inter_edges": 5,
            "assignment_quantile": 0.35,
            "hubness_correction": "csls",
            "hubness_k": 10,
            "rank_correction": True,
            "edge_weighting": "distance",
            "mutual_neighbors": False,
            "neighbor_mode": "distance",
            "symmetrize": True,
        },
    },
    "pbmc3k": {
        "input": "data/scanpy-pbmc3k.h5ad",
        "embedded": "data/scanpy-pbmc3k-scalp.h5ad",
        "batch_key": "batch",
        "label_key": "leiden",
        "preprocess": {
            "normalize": "auto",
            "hvg_flavor": "cell_ranger",
            "create_artificial_batch": True,
            "artificial_batch_count": 3,
        },
        "graph": {
            "n_neighbors": 15,
            "intra_fraction": 0.5,
            "n_inter_edges": 2,
            "assignment_quantile": 0.8,
            "hubness_correction": "csls",
            "hubness_k": 10,
            "rank_correction": True,
            "edge_weighting": "binary",
            "mutual_neighbors": True,
            "neighbor_mode": "distance",
            "symmetrize": True,
        },
    },
}


DOWNLOAD_REGISTRY = {
    "pancreas": {
        "kind": "url",
        "filename": "pancreas_normalized.h5ad",
        "url": "https://zenodo.org/records/3930949/files/pancreas_normalized.h5ad?download=1",
        "md5": "9b8bdaff49978c661f460f41afcfe0e0",
        "batch_key": "study",
        "label_key": "cell_type",
        "source": "https://zenodo.org/records/3930949",
    },
    "pbmc3k": {
        "kind": "figshare",
        "filename": "scanpy-pbmc3k.h5ad",
        "article_id": 16447278,
        "preferred_file": "scanpy-pbmc3k.h5ad",
        "batch_key": "batch",
        "label_key": "leiden",
        "source": "https://figshare.com/articles/dataset/scanpy-pbmc3k_h5ad/16447278",
    },
    "zebrafish": {
        "kind": "cellrank",
        "filename": "cellrank-zebrafish.h5ad",
        "function": "zebrafish",
        "batch_key": "Stage",
        "label_key": "lineages",
        "source": "https://cellrank.readthedocs.io/en/latest/api/_autosummary/datasets/cellrank.datasets.zebrafish.html",
    },
    "cellrank_pancreas": {
        "kind": "cellrank",
        "filename": "cellrank-pancreas.h5ad",
        "function": "pancreas",
        "batch_key": "clusters",
        "label_key": "clusters",
        "source": "https://cellrank.readthedocs.io/en/latest/api/_autosummary/datasets/cellrank.datasets.pancreas.html",
    },
}


def resolve_project_root(start: Path | None = None) -> Path:
    """Resolve the repository root whether Jupyter starts in root or notebooks/."""
    start = Path.cwd() if start is None else Path(start)
    return next(path for path in [start, *start.parents] if (path / "pyproject.toml").exists())


def resolve_project_path(path: str | Path, *, project_root: Path | None = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    project_root = resolve_project_root() if project_root is None else project_root
    return project_root / path


def dataset_config(name: str, *, embedded: bool = False, project_root: Path | None = None) -> dict:
    dataset = DATASET_REGISTRY[name].copy()
    key = "embedded" if embedded else "input"
    dataset["input_path"] = resolve_project_path(dataset[key], project_root=project_root)
    dataset["output_path"] = dataset["input_path"].with_name(f"{dataset['input_path'].stem}-scalp.h5ad")
    dataset["preprocess"] = dataset.get("preprocess", {}).copy()
    dataset["graph"] = dataset.get("graph", {}).copy()
    return dataset


def optimized_params_path(
    dataset_name: str,
    *,
    project_root: Path | None = None,
    params_dir: str | Path = "data/optimized_params",
) -> Path:
    params_dir = resolve_project_path(params_dir, project_root=project_root)
    return params_dir / f"{dataset_name}_graph_params.json"


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_optimized_graph_params(
    dataset_name: str,
    graph_params: dict,
    *,
    metadata: dict | None = None,
    project_root: Path | None = None,
    params_dir: str | Path = "data/optimized_params",
) -> Path:
    path = optimized_params_path(dataset_name, project_root=project_root, params_dir=params_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset_name,
        "graph_params": _json_ready(graph_params),
        "metadata": _json_ready(metadata or {}),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def load_optimized_graph_params(
    dataset_name: str,
    *,
    project_root: Path | None = None,
    params_dir: str | Path = "data/optimized_params",
) -> dict:
    path = optimized_params_path(dataset_name, project_root=project_root, params_dir=params_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Optimized graph parameter file not found: {path}. "
            "Run the latent Bayesian optimization notebook first."
        )
    return json.loads(path.read_text())


def make_estimator(dataset: dict, *, n_components: int = 100, random_state: int = 0) -> ScalpEstimator:
    return ScalpEstimator(
        batch_key=dataset["batch_key"],
        label_key=dataset["label_key"],
        n_components=n_components,
        random_state=random_state,
    )


def preprocess_params(dataset: dict, **overrides) -> dict:
    params = {
        "n_top_genes": 2000,
        "max_cells": None,
        "min_cell_genes": None,
        "min_gene_counts": 3,
        **dataset.get("preprocess", {}),
    }
    params.update({key: value for key, value in overrides.items() if value is not None})
    return params


def load_preprocessed_data(estimator: ScalpEstimator, dataset: dict, **preprocess_overrides) -> ad.AnnData:
    adata = estimator.to_data(dataset["input_path"])
    return estimator.preprocess(adata, **preprocess_params(dataset, **preprocess_overrides))


def embed_dataset(adata: ad.AnnData, estimator: ScalpEstimator, graph_params: dict) -> ad.AnnData:
    graph = estimator.data_to_graph(adata, **graph_params)
    adata.obsm["X_scalp"] = estimator.graph_to_vector(graph)
    return adata


def evaluate_dataset(estimator: ScalpEstimator, dataset: dict, *, embedding_key: str = "X_scalp") -> pd.DataFrame:
    adata = estimator.to_data(dataset["input_path"])
    label_key = dataset["label_key"] if dataset["label_key"] in adata.obs else None
    return score_embedding(
        adata,
        embedding_key=embedding_key,
        batch_key=dataset["batch_key"],
        label_key=label_key,
    )


def _coerce_sweep_value(value, template):
    if isinstance(template, bool) or template is None:
        return value
    if isinstance(template, int):
        return int(value)
    if isinstance(template, float):
        return float(value)
    return value


def parameterized_graph_params(base_graph_params: dict, sweep_parameter: str, sweep_value) -> dict:
    graph_params = base_graph_params.copy()
    graph_params[sweep_parameter] = _coerce_sweep_value(sweep_value, base_graph_params[sweep_parameter])
    return graph_params


def run_parameter_sweep(
    adata: ad.AnnData,
    *,
    estimator: ScalpEstimator,
    base_graph_params: dict,
    sweep_parameter: str,
    sweep_values,
) -> list[tuple[object, ad.AnnData]]:
    results = []
    for sweep_value in sweep_values:
        adata_sweep = adata.copy()
        graph_params = parameterized_graph_params(base_graph_params, sweep_parameter, sweep_value)
        adata_sweep.obsm["X_scalp"] = estimator.embed(
            adata_sweep,
            **graph_params,
            embedding_random_state=estimator.random_state,
        )
        results.append((graph_params[sweep_parameter], adata_sweep))
    return results


def summarize_sweep(results: list[tuple[object, ad.AnnData]]) -> list[tuple[object, tuple[int, int]]]:
    return [(round(value, 3) if isinstance(value, float) else value, adata_sweep.obsm["X_scalp"].shape) for value, adata_sweep in results]


def _format_sweep_value(value) -> str:
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def plot_parameter_sweep(
    results: list[tuple[object, ad.AnnData]],
    *,
    estimator: ScalpEstimator,
    sweep_parameter: str,
    batch_key: str,
    label_key: str,
):
    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 4 * n_rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(n_rows, 2)
    for row, (sweep_value, adata_sweep) in enumerate(results):
        estimator.plot(
            adata_sweep,
            embedding_key="X_scalp",
            axes=axes[row],
            s=8,
            alpha=0.85,
            random_state=estimator.random_state,
            legend_markerscale=3.5,
        )
        formatted_value = _format_sweep_value(sweep_value)
        axes[row, 0].set_title(f"{sweep_parameter}={formatted_value} by {batch_key}")
        axes[row, 1].set_title(f"{sweep_parameter}={formatted_value} by {label_key}")
    return fig


def run_and_plot_sweep(
    *,
    adata: ad.AnnData,
    estimator: ScalpEstimator,
    base_graph_params: dict,
    sweep_parameter: str,
    sweep_values,
    batch_key: str,
    label_key: str,
):
    results = run_parameter_sweep(
        adata,
        estimator=estimator,
        base_graph_params=base_graph_params,
        sweep_parameter=sweep_parameter,
        sweep_values=sweep_values,
    )
    fig = plot_parameter_sweep(results, estimator=estimator, sweep_parameter=sweep_parameter, batch_key=batch_key, label_key=label_key)
    return results, summarize_sweep(results), fig


def md5sum(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, output_path: str | Path, *, expected_md5: str | None = None, overwrite: bool = False) -> Path:
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        if expected_md5 is None or md5sum(output_path) == expected_md5:
            print(f"exists: {output_path}")
            return output_path
        raise ValueError(f"Existing file has wrong md5: {output_path}")

    tmp_path = output_path.with_suffix(output_path.suffix + ".download")
    tmp_path.unlink(missing_ok=True)
    print(f"downloading: {url}")
    with urllib.request.urlopen(url) as response, open(tmp_path, "wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)

    if expected_md5 is not None:
        observed_md5 = md5sum(tmp_path)
        if observed_md5 != expected_md5:
            tmp_path.unlink(missing_ok=True)
            raise ValueError(f"md5 mismatch for {output_path.name}: {observed_md5} != {expected_md5}")

    tmp_path.replace(output_path)
    print(f"saved: {output_path}")
    return output_path


def figshare_download_url(article_id: int, *, preferred_file: str | None = None) -> str:
    with urllib.request.urlopen(f"https://api.figshare.com/v2/articles/{article_id}") as response:
        article = json.load(response)
    files = article.get("files", [])
    if preferred_file is not None:
        for file_info in files:
            if file_info.get("name") == preferred_file:
                return file_info["download_url"]
    if len(files) == 1:
        return files[0]["download_url"]
    raise ValueError(f"Could not choose a Figshare file from: {[file_info.get('name') for file_info in files]}")


def download_cellrank_dataset(function_name: str, output_path: str | Path, *, overwrite: bool = False) -> Path:
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        print(f"exists: {output_path}")
        return output_path
    try:
        import cellrank as cr
    except ImportError as exc:
        raise ImportError("CellRank datasets require `pip install cellrank`.") from exc
    getattr(cr.datasets, function_name)(path=str(output_path))
    print(f"saved: {output_path}")
    return output_path


def download_datasets(selected_datasets, *, data_dir: str | Path, overwrite: bool = False) -> dict[str, Path]:
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    downloaded = {}
    for name in selected_datasets:
        dataset = DOWNLOAD_REGISTRY[name]
        output_path = data_dir / dataset["filename"]
        if dataset["kind"] == "url":
            path = download_file(dataset["url"], output_path, expected_md5=dataset.get("md5"), overwrite=overwrite)
        elif dataset["kind"] == "figshare":
            url = figshare_download_url(dataset["article_id"], preferred_file=dataset.get("preferred_file"))
            path = download_file(url, output_path, overwrite=overwrite)
        elif dataset["kind"] == "cellrank":
            path = download_cellrank_dataset(dataset["function"], output_path, overwrite=overwrite)
        else:
            raise ValueError(f"Unknown dataset kind: {dataset['kind']}")
        downloaded[name] = path
    return downloaded


def summarize_downloads(downloaded: dict[str, Path], *, project_root: Path | None = None) -> pd.DataFrame:
    project_root = resolve_project_root() if project_root is None else project_root
    rows = []
    for name, path in downloaded.items():
        dataset = DOWNLOAD_REGISTRY[name]
        adata = ad.read_h5ad(path, backed="r")
        rows.append(
            {
                "dataset": name,
                "path": str(Path(path).relative_to(project_root)),
                "cells": adata.n_obs,
                "genes": adata.n_vars,
                "batch_key": dataset["batch_key"],
                "has_batch_key": dataset["batch_key"] in adata.obs,
                "label_key": dataset["label_key"],
                "has_label_key": dataset["label_key"] in adata.obs,
                "source": dataset["source"],
            }
        )
        adata.file.close()
    return pd.DataFrame(rows)

from __future__ import annotations

from pathlib import Path
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import warnings

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from clasp.estimator import ClaspEstimator
from clasp.metrics import score_embedding


DEFAULT_PREPROCESS_PARAMS = {
    "normalize": False,
    "hvg_flavor": "variance",
    "min_gene_counts": 0,
    "create_artificial_batch": False,
}


DEFAULT_GRAPH_PARAMS = {
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
}


PAPER_DATASET_DOWNLOADS = [
    "scib_pancreas",
    "scib_lung_atlas",
    "scib_immune_human",
    "scib_immune_human_mouse",
    "cellrank_bone_marrow",
    "cellrank_lung",
    "cellrank_pancreas",
    "cellrank_reprogramming_schiebinger",
    "cellrank_reprogramming_morris",
    "zebrafish",
]


PAPER_DATASETS_REQUIRING_MANUAL_CURATION = [
    {
        "dataset": "Cancer benchmark collection",
        "paper_description": "11 cancer scRNA-seq studies: breast cancer, PDAC, melanoma, lung cancer, colorectal cancer.",
        "references": "Azizi, Bassez, Slyper, Elyada, Peng, Bi, Karlsson, Lee, Qian, Nath, Zhang",
        "reason": "The CLASP PDF names the studies via citations but does not provide direct h5ad files or a machine-readable manifest.",
    },
    {
        "dataset": "Additional time-series collection",
        "paper_description": "Human hematopoietic differentiation, fetal lung development, planaria regeneration, mouse cortex/cerebellum development.",
        "references": "Tusi, Travaglini, Plass, Tempora",
        "reason": "The CLASP PDF names the biological datasets but does not provide direct h5ad files or a machine-readable manifest.",
    },
]


def dataset_yaml_dir() -> Path:
    return Path(__file__).with_name("datasets")


def list_dataset_config_files(config_dir: str | Path | None = None) -> list[Path]:
    config_dir = dataset_yaml_dir() if config_dir is None else Path(config_dir)
    return sorted(config_dir.glob("*.yaml"))


def read_dataset_spec(name: str, *, config_dir: str | Path | None = None) -> dict:
    path = (dataset_yaml_dir() if config_dir is None else Path(config_dir)) / f"{name}.yaml"
    if not path.exists():
        available = [item.stem for item in list_dataset_config_files(config_dir)]
        raise KeyError(f"Unknown dataset {name!r}. Available datasets: {available}")
    with open(path) as handle:
        spec = yaml.safe_load(handle)
    if not isinstance(spec, dict):
        raise ValueError(f"Dataset config must be a mapping: {path}")
    if spec.get("name") != name:
        raise ValueError(f"Dataset config {path} has name={spec.get('name')!r}; expected {name!r}.")
    return spec


def load_dataset_specs(config_dir: str | Path | None = None) -> dict[str, dict]:
    specs = {}
    for path in list_dataset_config_files(config_dir):
        with open(path) as handle:
            spec = yaml.safe_load(handle)
        if not isinstance(spec, dict) or "name" not in spec:
            raise ValueError(f"Dataset config must contain a name: {path}")
        if spec["name"] in specs:
            raise ValueError(f"Duplicate dataset config name: {spec['name']!r}")
        specs[spec["name"]] = spec
    return specs


def _dataset_entry_from_spec(spec: dict) -> dict:
    required = ("input", "embedded", "batch_key", "label_key")
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(f"Dataset config {spec.get('name')!r} is missing keys: {missing}")
    return {
        "input": spec["input"],
        "embedded": spec["embedded"],
        "batch_key": spec["batch_key"],
        "label_key": spec["label_key"],
        "source": spec.get("source"),
        "paper_group": spec.get("paper_group"),
        "preprocess": spec.get("preprocess", {}).copy(),
        "graph": spec.get("graph", {}).copy(),
    }


def _download_entry_from_spec(spec: dict) -> dict:
    download = spec.get("download", {}).copy()
    if not download:
        return {}
    if "filename" not in download:
        download["filename"] = Path(spec["input"]).name
    download["batch_key"] = spec["batch_key"]
    download["label_key"] = spec["label_key"]
    download["source"] = spec.get("source")
    download["paper_group"] = spec.get("paper_group")
    return download


def _registries_from_dataset_specs() -> tuple[dict[str, dict], dict[str, dict]]:
    specs = load_dataset_specs()
    dataset_registry = {name: _dataset_entry_from_spec(spec) for name, spec in specs.items()}
    download_registry = {
        name: download
        for name, spec in specs.items()
        if (download := _download_entry_from_spec(spec))
    }
    return dataset_registry, download_registry


DATASET_REGISTRY, DOWNLOAD_REGISTRY = _registries_from_dataset_specs()


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
    if name in DATASET_REGISTRY:
        dataset = DATASET_REGISTRY[name].copy()
    elif name in DOWNLOAD_REGISTRY:
        download = DOWNLOAD_REGISTRY[name]
        input_path = Path("data") / download["filename"]
        dataset = {
            "input": str(input_path),
            "embedded": str(input_path.with_name(f"{input_path.stem}-clasp.h5ad")),
            "batch_key": download["batch_key"],
            "label_key": download["label_key"],
            "preprocess": DEFAULT_PREPROCESS_PARAMS.copy(),
            "graph": DEFAULT_GRAPH_PARAMS.copy(),
            "source": download["source"],
            "paper_group": download.get("paper_group"),
        }
    else:
        available = sorted(set(DATASET_REGISTRY) | set(DOWNLOAD_REGISTRY))
        raise KeyError(f"Unknown dataset {name!r}. Available datasets: {available}")
    key = "embedded" if embedded else "input"
    dataset["input_path"] = resolve_project_path(dataset[key], project_root=project_root)
    dataset["output_path"] = dataset["input_path"].with_name(f"{dataset['input_path'].stem}-clasp.h5ad")
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
    preprocess_params: dict | None = None,
    estimator_params: dict | None = None,
    metadata: dict | None = None,
    project_root: Path | None = None,
    params_dir: str | Path = "data/optimized_params",
) -> Path:
    path = optimized_params_path(dataset_name, project_root=project_root, params_dir=params_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset_name,
        "graph_params": _json_ready(graph_params),
        "preprocess_params": _json_ready(preprocess_params or {}),
        "estimator_params": _json_ready(estimator_params or {}),
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


def load_or_default_params(
    dataset_name: str,
    dataset: dict | None = None,
    *,
    project_root: Path | None = None,
    params_dir: str | Path = "data/optimized_params",
) -> dict:
    dataset = dataset_config(dataset_name, project_root=project_root) if dataset is None else dataset
    path = optimized_params_path(dataset_name, project_root=project_root, params_dir=params_dir)
    try:
        payload = load_optimized_graph_params(dataset_name, project_root=project_root, params_dir=params_dir)
        payload.setdefault("source", "optimized")
        payload.setdefault("path", str(path))
        return payload
    except FileNotFoundError:
        return {
            "dataset": dataset_name,
            "graph_params": dataset.get("graph", {}).copy(),
            "preprocess_params": dataset.get("preprocess", {}).copy(),
            "estimator_params": {},
            "metadata": {
                "source": "dataset_defaults",
                "message": f"No optimized parameter file found at {path}; using dataset registry defaults.",
            },
            "source": "dataset_defaults",
            "path": str(path),
        }


def install_bo_warning_filters() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"\s*Found Intel OpenMP .* LLVM OpenMP .*",
        category=RuntimeWarning,
        module=r"threadpoolctl",
    )


def limit_native_threads(n_threads: int = 1) -> None:
    value = str(n_threads)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(name, value)


def ensure_bo_dependencies(n_threads: int = 1) -> None:
    limit_native_threads(n_threads)
    install_bo_warning_filters()
    try:
        import botorch  # noqa: F401
        import gpytorch  # noqa: F401
        import torch

        torch.set_num_threads(n_threads)
        try:
            torch.set_num_interop_threads(n_threads)
        except RuntimeError:
            pass
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "..[bo]"])


def make_estimator(dataset: dict, *, n_components: int = 100, random_state: int = 0) -> ClaspEstimator:
    return ClaspEstimator(
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


def load_preprocessed_data(estimator: ClaspEstimator, dataset: dict, **preprocess_overrides) -> ad.AnnData:
    adata = estimator.to_data(dataset["input_path"])
    return estimator.preprocess(adata, **preprocess_params(dataset, **preprocess_overrides))


def embed_dataset(adata: ad.AnnData, estimator: ClaspEstimator, graph_params: dict) -> ad.AnnData:
    graph = estimator.data_to_graph(adata, **graph_params)
    adata.obsm["X_clasp"] = estimator.graph_to_vector(graph)
    return adata


def evaluate_dataset(estimator: ClaspEstimator, dataset: dict, *, embedding_key: str = "X_clasp") -> pd.DataFrame:
    adata = estimator.to_data(dataset["input_path"])
    label_key = dataset["label_key"] if dataset["label_key"] in adata.obs else None
    return score_embedding(
        adata,
        embedding_key=embedding_key,
        batch_key=dataset["batch_key"],
        label_key=label_key,
    )


def split_optimization_params(
    params: dict,
    *,
    base_preprocess_params: dict,
    fixed_preprocess_params: dict,
    base_estimator_params: dict,
    base_graph_params: dict,
    preprocess_search_space: dict,
    estimator_search_space: dict,
    graph_search_space: dict,
) -> tuple[dict, dict, dict]:
    preprocess_params = {
        **base_preprocess_params,
        **fixed_preprocess_params,
        **{key: params[key] for key in preprocess_search_space if key in params},
    }
    estimator_params = {
        **base_estimator_params,
        **{key: params[key] for key in estimator_search_space if key in params},
    }
    graph_params = {
        **base_graph_params,
        **{key: params[key] for key in graph_search_space if key in params},
    }
    return preprocess_params, estimator_params, graph_params


def metric_value(scores, key: str, default: float = 0.0) -> float:
    value = float(scores.get(key, default))
    return default if not np.isfinite(value) else value


def make_clasp_optimization_objective(
    *,
    dataset: dict,
    raw_adata: ad.AnnData,
    base_preprocess_params: dict,
    fixed_preprocess_params: dict,
    base_estimator_params: dict,
    base_graph_params: dict,
    preprocess_search_space: dict,
    estimator_search_space: dict,
    graph_search_space: dict,
    random_state: int = 0,
    embedding_method: str = "umap",
    embedding_epochs: int | None = 60,
    invalid_score: float = -1e9,
    label_weight: float = 1.0,
    batch_weight: float = 0.25,
    density_weight: float = -0.05,
):
    def objective(params: dict) -> float:
        trial_preprocess, trial_estimator_params, trial_graph = split_optimization_params(
            params,
            base_preprocess_params=base_preprocess_params,
            fixed_preprocess_params=fixed_preprocess_params,
            base_estimator_params=base_estimator_params,
            base_graph_params=base_graph_params,
            preprocess_search_space=preprocess_search_space,
            estimator_search_space=estimator_search_space,
            graph_search_space=graph_search_space,
        )
        trial_estimator = make_estimator(dataset, random_state=random_state, **trial_estimator_params)
        try:
            trial = trial_estimator.preprocess(raw_adata, **trial_preprocess)
            embedding_kwargs = {
                "embedding_method": embedding_method,
                "embedding_random_state": random_state,
            }
            if embedding_epochs is not None and embedding_method in {"auto", "umap"}:
                embedding_kwargs["n_epochs"] = embedding_epochs
            trial.obsm["X_clasp"] = trial_estimator.embed(
                trial,
                **trial_graph,
                **embedding_kwargs,
            )
            scores = score_embedding(
                trial,
                embedding_key="X_clasp",
                batch_key=dataset["batch_key"],
                label_key=dataset["label_key"] if dataset["label_key"] in trial.obs else None,
            ).iloc[0]
        except Exception as exc:
            print(
                "failed params="
                f"preprocess={trial_preprocess}, estimator={trial_estimator_params}, graph={trial_graph}: {exc}"
            )
            return float(invalid_score)

        try:
            label_agreement = metric_value(scores, "knn_label_agreement")
            batch_mixing = metric_value(scores, "batch_mixing")
            graph_density = metric_value(scores, "graph_density")
            return float(label_weight * label_agreement + batch_weight * batch_mixing + density_weight * graph_density)
        finally:
            del trial
            gc.collect()

    return objective


def optimization_search_space(
    *,
    preprocess_search_space: dict,
    estimator_search_space: dict,
    graph_search_space: dict,
) -> dict:
    return {
        **preprocess_search_space,
        **estimator_search_space,
        **graph_search_space,
    }


def compact_bounds(best: dict, name: str, low, high, radius, *, integer: bool = False) -> list:
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


def make_compact_search_space(
    search_space: dict,
    best_params: dict,
    radii: dict,
    *,
    fix_categoricals: bool = True,
) -> dict:
    compact = {}
    for name, spec in search_space.items():
        if spec["type"] == "categorical":
            compact[name] = {"type": "categorical", "values": [best_params[name]]} if fix_categoricals else spec.copy()
            continue
        low, high = spec["bounds"]
        radius = radii[name]
        compact[name] = {
            **spec,
            "bounds": compact_bounds(best_params, name, low, high, radius, integer=spec["type"] == "int"),
        }
    return compact


def run_latent_bayesopt(*args, **kwargs):
    ensure_bo_dependencies()
    from clasp.optimization import latent_bayesopt

    return latent_bayesopt(*args, **kwargs)


def save_best_optimization_result(
    *,
    dataset_name: str,
    optimization_results: dict,
    base_preprocess_params: dict,
    fixed_preprocess_params: dict,
    base_estimator_params: dict,
    base_graph_params: dict,
    preprocess_search_space: dict,
    estimator_search_space: dict,
    graph_search_space: dict,
    random_state: int,
    project_root: Path | None = None,
) -> tuple[Path, dict, dict, dict]:
    best_model_name, best_result = max(optimization_results.items(), key=lambda item: item[1]["best_score"])
    optimized_preprocess_params, optimized_estimator_params, optimized_graph_params = split_optimization_params(
        best_result["best_params"],
        base_preprocess_params=base_preprocess_params,
        fixed_preprocess_params=fixed_preprocess_params,
        base_estimator_params=base_estimator_params,
        base_graph_params=base_graph_params,
        preprocess_search_space=preprocess_search_space,
        estimator_search_space=estimator_search_space,
        graph_search_space=graph_search_space,
    )
    metadata = {
        "best_model": best_model_name,
        "best_score": best_result["best_score"],
        "fixed_preprocess_params": fixed_preprocess_params,
        "random_state": random_state,
    }
    if "pca" in optimization_results:
        metadata["pca_best_score"] = optimization_results["pca"]["best_score"]
    if "gplvm" in optimization_results:
        metadata["gplvm_best_score"] = optimization_results["gplvm"]["best_score"]

    path = save_optimized_graph_params(
        dataset_name,
        optimized_graph_params,
        preprocess_params=optimized_preprocess_params,
        estimator_params=optimized_estimator_params,
        metadata=metadata,
        project_root=project_root,
    )
    return path, optimized_preprocess_params, optimized_estimator_params, optimized_graph_params


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
    estimator: ClaspEstimator,
    base_graph_params: dict,
    sweep_parameter: str,
    sweep_values,
) -> list[tuple[object, ad.AnnData]]:
    results = []
    for sweep_value in sweep_values:
        adata_sweep = adata.copy()
        graph_params = parameterized_graph_params(base_graph_params, sweep_parameter, sweep_value)
        adata_sweep.obsm["X_clasp"] = estimator.embed(
            adata_sweep,
            **graph_params,
            embedding_random_state=estimator.random_state,
        )
        results.append((graph_params[sweep_parameter], adata_sweep))
    return results


def summarize_sweep(results: list[tuple[object, ad.AnnData]]) -> list[tuple[object, tuple[int, int]]]:
    return [(round(value, 3) if isinstance(value, float) else value, adata_sweep.obsm["X_clasp"].shape) for value, adata_sweep in results]


def _format_sweep_value(value) -> str:
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def plot_parameter_sweep(
    results: list[tuple[object, ad.AnnData]],
    *,
    estimator: ClaspEstimator,
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
            embedding_key="X_clasp",
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
    estimator: ClaspEstimator,
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
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cellrank"])
        import cellrank as cr
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


def summarize_downloads(
    downloaded: dict[str, Path],
    *,
    project_root: Path | None = None,
    inspect_h5ad: bool = False,
) -> pd.DataFrame:
    project_root = resolve_project_root() if project_root is None else project_root
    rows = []
    for name, path in downloaded.items():
        path = Path(path)
        dataset = DOWNLOAD_REGISTRY[name]
        row = {
            "dataset": name,
            "path": str(path.relative_to(project_root)),
            "file_size_gb": round(path.stat().st_size / 1024**3, 3),
            "batch_key": dataset["batch_key"],
            "label_key": dataset["label_key"],
            "source": dataset["source"],
            "paper_group": dataset.get("paper_group"),
        }
        if inspect_h5ad:
            adata = ad.read_h5ad(path, backed="r")
            try:
                row.update(
                    {
                        "cells": adata.n_obs,
                        "genes": adata.n_vars,
                        "has_batch_key": dataset["batch_key"] in adata.obs,
                        "has_label_key": dataset["label_key"] in adata.obs,
                    }
                )
            finally:
                adata.file.close()
        rows.append(row)
    return pd.DataFrame(rows)


def paper_dataset_manifest(selected_datasets=None) -> pd.DataFrame:
    selected_datasets = PAPER_DATASET_DOWNLOADS if selected_datasets is None else selected_datasets
    rows = []
    for name in selected_datasets:
        dataset = DOWNLOAD_REGISTRY[name]
        rows.append(
            {
                "dataset": name,
                "filename": dataset["filename"],
                "kind": dataset["kind"],
                "batch_key": dataset["batch_key"],
                "label_key": dataset["label_key"],
                "paper_group": dataset.get("paper_group"),
                "source": dataset["source"],
            }
        )
    return pd.DataFrame(rows)


def paper_datasets_requiring_manual_curation() -> pd.DataFrame:
    return pd.DataFrame(PAPER_DATASETS_REQUIRING_MANUAL_CURATION)

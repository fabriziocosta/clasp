from __future__ import annotations

from clasp.notebook_utils import (
    DOWNLOAD_REGISTRY,
    PAPER_DATASET_DOWNLOADS,
    PAPER_DATASETS_REQUIRING_MANUAL_CURATION,
    dataset_config,
    list_dataset_config_files,
    load_optimized_graph_params,
    make_compact_search_space,
    optimization_search_space,
    optimized_params_path,
    paper_dataset_manifest,
    read_dataset_spec,
    save_best_optimization_result,
    save_optimized_graph_params,
    split_optimization_params,
)


def test_optimized_graph_params_roundtrip(tmp_path):
    graph_params = {
        "n_neighbors": 33,
        "intra_fraction": 0.9,
        "rank_correction": True,
    }

    path = save_optimized_graph_params(
        "pancreas",
        graph_params,
        preprocess_params={"n_top_genes": 2000},
        estimator_params={"n_components": 80},
        metadata={"best_score": 0.94},
        project_root=tmp_path,
    )

    assert path == optimized_params_path("pancreas", project_root=tmp_path)
    payload = load_optimized_graph_params("pancreas", project_root=tmp_path)
    assert payload["dataset"] == "pancreas"
    assert payload["graph_params"] == graph_params
    assert payload["preprocess_params"]["n_top_genes"] == 2000
    assert payload["estimator_params"]["n_components"] == 80
    assert payload["metadata"]["best_score"] == 0.94


def test_optimization_helpers_split_and_compact():
    preprocess_space = {"n_top_genes": {"type": "int", "bounds": [500, 3000]}}
    estimator_space = {"n_components": {"type": "int", "bounds": [20, 150]}}
    graph_space = {
        "n_neighbors": {"type": "int", "bounds": [5, 40]},
        "edge_weighting": {"type": "categorical", "values": ["binary", "distance"]},
        "inter_edge_mode": {"type": "categorical", "values": ["propagate_neighbors", "assignment"]},
    }

    search_space = optimization_search_space(
        preprocess_search_space=preprocess_space,
        estimator_search_space=estimator_space,
        graph_search_space=graph_space,
    )
    compact = make_compact_search_space(
        search_space,
        {
            "n_top_genes": 2000,
            "n_components": 80,
            "n_neighbors": 33,
            "edge_weighting": "distance",
            "inter_edge_mode": "propagate_neighbors",
        },
        {"n_top_genes": 500, "n_components": 20, "n_neighbors": 5},
    )

    assert compact["n_top_genes"]["bounds"] == [1500, 2500]
    assert compact["n_components"]["bounds"] == [60, 100]
    assert compact["n_neighbors"]["bounds"] == [28, 38]
    assert compact["edge_weighting"]["values"] == ["distance"]
    assert compact["inter_edge_mode"]["values"] == ["propagate_neighbors"]

    preprocess, estimator, graph = split_optimization_params(
        {"n_top_genes": 1000, "n_components": 50, "n_neighbors": 10},
        base_preprocess_params={"n_top_genes": 2000},
        fixed_preprocess_params={"max_cells": 2000},
        base_estimator_params={"n_components": 100},
        base_graph_params={"n_neighbors": 15, "metric": "euclidean"},
        preprocess_search_space=preprocess_space,
        estimator_search_space=estimator_space,
        graph_search_space=graph_space,
    )

    assert preprocess == {"n_top_genes": 1000, "max_cells": 2000}
    assert estimator == {"n_components": 50}
    assert graph == {"n_neighbors": 10, "metric": "euclidean"}


def test_save_best_optimization_result_uses_best_model(tmp_path):
    result = save_best_optimization_result(
        dataset_name="pancreas",
        optimization_results={
            "pca": {"best_score": 0.5, "best_params": {"n_neighbors": 10}},
            "gplvm": {"best_score": 0.7, "best_params": {"n_neighbors": 22}},
        },
        base_preprocess_params={"n_top_genes": 2000},
        fixed_preprocess_params={"max_cells": 2000},
        base_estimator_params={"n_components": 100},
        base_graph_params={"n_neighbors": 15},
        preprocess_search_space={},
        estimator_search_space={},
        graph_search_space={"n_neighbors": {"type": "int", "bounds": [5, 40]}},
        random_state=0,
        project_root=tmp_path,
    )

    path, preprocess, estimator, graph = result
    payload = load_optimized_graph_params("pancreas", project_root=tmp_path)
    assert path == optimized_params_path("pancreas", project_root=tmp_path)
    assert preprocess == {"n_top_genes": 2000, "max_cells": 2000}
    assert estimator == {"n_components": 100}
    assert graph == {"n_neighbors": 22}
    assert payload["metadata"]["best_model"] == "gplvm"


def test_save_best_optimization_result_allows_pca_only(tmp_path):
    path, _, _, graph = save_best_optimization_result(
        dataset_name="pancreas",
        optimization_results={
            "pca": {"best_score": 0.5, "best_params": {"n_neighbors": 10}},
        },
        base_preprocess_params={},
        fixed_preprocess_params={},
        base_estimator_params={},
        base_graph_params={"n_neighbors": 15},
        preprocess_search_space={},
        estimator_search_space={},
        graph_search_space={"n_neighbors": {"type": "int", "bounds": [5, 40]}},
        random_state=0,
        project_root=tmp_path,
    )

    payload = load_optimized_graph_params("pancreas", project_root=tmp_path)
    assert path.exists()
    assert graph == {"n_neighbors": 10}
    assert payload["metadata"]["best_model"] == "pca"
    assert "gplvm_best_score" not in payload["metadata"]


def test_paper_dataset_download_manifest_is_complete():
    assert PAPER_DATASET_DOWNLOADS
    assert set(PAPER_DATASET_DOWNLOADS).issubset(DOWNLOAD_REGISTRY)

    manifest = paper_dataset_manifest()
    assert set(manifest["dataset"]) == set(PAPER_DATASET_DOWNLOADS)
    assert manifest["filename"].notna().all()
    assert manifest["source"].notna().all()
    assert manifest["paper_group"].notna().all()
    assert set(manifest["kind"]).issubset({"url", "figshare"})


def test_download_registry_uses_direct_sources():
    for name, entry in DOWNLOAD_REGISTRY.items():
        assert entry["kind"] in {"url", "figshare"}
        assert "function" not in entry, name
        if entry["kind"] == "url":
            assert entry["url"].startswith(("http://", "https://"))
        else:
            assert "article_id" in entry


def test_manual_paper_dataset_list_documents_unresolved_sources():
    assert PAPER_DATASETS_REQUIRING_MANUAL_CURATION
    for row in PAPER_DATASETS_REQUIRING_MANUAL_CURATION:
        assert row["dataset"]
        assert row["paper_description"]
        assert row["reason"]


def test_dataset_config_accepts_download_registry_entries(tmp_path):
    dataset = dataset_config("scib_pancreas", project_root=tmp_path)

    assert dataset["input_path"] == tmp_path / "data" / "human_pancreas_norm_complexBatch.h5ad"
    assert dataset["output_path"] == tmp_path / "data" / "human_pancreas_norm_complexBatch-clasp.h5ad"
    assert dataset["batch_key"] == "tech"
    assert dataset["label_key"] == "celltype"
    assert dataset["preprocess"]["normalize"] is False
    assert dataset["graph"]["hubness_correction"] == "csls"


def test_dataset_configs_are_loaded_from_yaml():
    files = list_dataset_config_files()
    names = {path.stem for path in files}

    assert "scib_pancreas" in names
    assert set(DOWNLOAD_REGISTRY).issubset(names)

    spec = read_dataset_spec("scib_pancreas")
    assert spec["batch_key"] == "tech"
    assert spec["label_key"] == "celltype"
    assert spec["download"]["filename"] == "human_pancreas_norm_complexBatch.h5ad"


def test_cellrank_bone_marrow_uses_artificial_batches():
    dataset = dataset_config("cellrank_bone_marrow")

    assert dataset["batch_key"] == "sample"
    assert dataset["label_key"] == "clusters"
    assert dataset["preprocess"]["create_artificial_batch"] is True

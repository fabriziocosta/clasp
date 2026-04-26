from __future__ import annotations

from scalp_lite.notebook_utils import load_optimized_graph_params, optimized_params_path, save_optimized_graph_params


def test_optimized_graph_params_roundtrip(tmp_path):
    graph_params = {
        "n_neighbors": 33,
        "intra_fraction": 0.9,
        "rank_correction": True,
    }

    path = save_optimized_graph_params(
        "pancreas",
        graph_params,
        metadata={"best_score": 0.94},
        project_root=tmp_path,
    )

    assert path == optimized_params_path("pancreas", project_root=tmp_path)
    payload = load_optimized_graph_params("pancreas", project_root=tmp_path)
    assert payload["dataset"] == "pancreas"
    assert payload["graph_params"] == graph_params
    assert payload["metadata"]["best_score"] == 0.94

from __future__ import annotations

from copy import deepcopy


CLASP_PRESETS = {
    "balanced": {
        "description": "Robust default selected from the median/majority of optimized benchmark runs.",
        "preprocess": {
            "n_top_genes": 1300,
            "normalize": False,
            "hvg_flavor": "variance",
            "min_gene_counts": 0,
            "create_artificial_batch": False,
        },
        "estimator": {
            "n_components": 80,
        },
        "graph": {
            "n_neighbors": 24,
            "intra_fraction": 0.5,
            "n_inter_edges": 3,
            "metric": "euclidean",
            "assignment_quantile": 0.32,
            "hubness_correction": "csls",
            "hubness_k": 12,
            "rank_correction": True,
            "edge_weighting": "distance",
            "inter_edge_mode": "assignment",
            "mutual_neighbors": True,
            "neighbor_mode": "distance",
            "symmetrize": True,
        },
    },
    "trajectory": {
        "description": "Higher-connectivity CellRank-style trajectory preset for smooth temporal datasets.",
        "preprocess": {
            "n_top_genes": 1300,
            "normalize": False,
            "hvg_flavor": "variance",
            "min_gene_counts": 0,
            "create_artificial_batch": False,
        },
        "estimator": {
            "n_components": 90,
        },
        "graph": {
            "n_neighbors": 28,
            "intra_fraction": 0.58,
            "n_inter_edges": 4,
            "metric": "euclidean",
            "assignment_quantile": 0.30,
            "hubness_correction": "csls",
            "hubness_k": 13,
            "rank_correction": True,
            "edge_weighting": "distance",
            "inter_edge_mode": "propagate_neighbors",
            "mutual_neighbors": True,
            "neighbor_mode": "distance",
            "symmetrize": True,
        },
    },
}


def available_presets() -> tuple[str, ...]:
    return tuple(CLASP_PRESETS)


def get_preset(name: str) -> dict:
    try:
        return deepcopy(CLASP_PRESETS[name])
    except KeyError as exc:
        available = ", ".join(available_presets())
        raise ValueError(f"Unknown CLASP preset {name!r}. Available presets: {available}.") from exc


def get_preprocess_preset(name: str) -> dict:
    return get_preset(name)["preprocess"]


def get_estimator_preset(name: str) -> dict:
    return get_preset(name)["estimator"]


def get_graph_preset(name: str) -> dict:
    return get_preset(name)["graph"]

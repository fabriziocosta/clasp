from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from clasp import ClaspEstimator, available_presets


def _none_or_int(value: str) -> int | None:
    if value.lower() == "none":
        return None
    return int(value)


def _none_or_float(value: str) -> float | None:
    if value.lower() == "none":
        return None
    return float(value)


def _bool_value(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected one of: true, false, yes, no, 1, 0")


def _normalize_value(value: str) -> bool | str:
    normalized = value.lower()
    if normalized == "auto":
        return "auto"
    return _bool_value(value)


def _set_if_present(target: dict[str, Any], name: str, value: Any) -> None:
    if value is not None:
        target[name] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clasp", description="CLASP command line tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    embed = subparsers.add_parser("embed", help="Embed an AnnData .h5ad file with CLASP.")
    embed.add_argument("input", type=Path, help="Input AnnData .h5ad file.")
    embed.add_argument("output", type=Path, help="Output embedded AnnData .h5ad file.")
    embed.add_argument(
        "--preset",
        choices=available_presets(),
        default="balanced",
        help="Named CLASP parameter preset.",
    )
    embed.add_argument("--batch-key", default="batch", help="Observation column containing batch/sample/time labels.")
    embed.add_argument("--label-key", default="label", help="Observation column containing biological labels.")
    embed.add_argument("--rep-key", default="X_pca", help="Representation key in adata.obsm used for graph construction.")
    embed.add_argument("--embedding-key", default="X_clasp", help="Output embedding key in adata.obsm.")
    embed.add_argument("--figure", type=Path, default=None, help="Optional output figure path.")
    embed.add_argument("--random-state", type=int, default=0, help="Random seed for preprocessing and embedding.")
    embed.add_argument("--embedding-method", default="auto", help="Graph embedding method: auto, umap, spectral, or a supported backend.")
    embed.add_argument("--embedding-components", type=int, default=2, help="Number of output embedding dimensions.")

    preprocess = embed.add_argument_group("preprocessing")
    preprocess.add_argument("--n-top-genes", type=_none_or_int, default=None, help="Highly variable genes to keep; use 'none' to disable.")
    preprocess.add_argument("--max-cells", type=_none_or_int, default=None, help="Optional maximum cells after stratified subsampling.")
    preprocess.add_argument("--min-cell-genes", type=_none_or_int, default=None, help="Optional minimum detected genes per cell.")
    preprocess.add_argument("--min-gene-counts", type=int, default=None, help="Minimum total counts per gene.")
    preprocess.add_argument("--normalize", type=_normalize_value, default=None, help="Normalize expression: auto, true, or false.")
    preprocess.add_argument("--target-sum", type=float, default=None, help="Library-size target for normalization.")
    preprocess.add_argument("--log1p", type=_bool_value, default=None, help="Apply log1p after normalization.")
    preprocess.add_argument("--hvg-flavor", default=None, help="HVG selection flavor, e.g. variance or cell_ranger.")
    preprocess.add_argument("--create-artificial-batch", type=_bool_value, default=None, help="Create deterministic split_* batches if batch_key is absent.")
    preprocess.add_argument("--artificial-batch-count", type=int, default=None, help="Number of artificial split_* batches.")

    estimator = embed.add_argument_group("estimator")
    estimator.add_argument("--n-components", type=int, default=None, help="PCA components used before graph construction.")

    graph = embed.add_argument_group("graph")
    graph.add_argument("--n-neighbors", type=int, default=None, help="Total neighborhood scale.")
    graph.add_argument("--intra-fraction", type=float, default=None, help="Fraction of neighbors assigned to within-batch edges.")
    graph.add_argument("--n-inter-edges", type=int, default=None, help="Number of assignment layers between batch pairs.")
    graph.add_argument("--metric", default=None, help="Distance metric.")
    graph.add_argument("--assignment-quantile", type=_none_or_float, default=None, help="Assignment distance quantile; use 'none' to disable.")
    graph.add_argument("--hubness-correction", choices=["none", "csls"], default=None, help="Hubness correction method.")
    graph.add_argument("--hubness-k", type=int, default=None, help="CSLS local neighborhood size.")
    graph.add_argument("--rank-correction", type=_bool_value, default=None, help="Use reciprocal rank scores for within-batch kNN.")
    graph.add_argument("--edge-weighting", choices=["binary", "distance"], default=None, help="Graph edge weighting mode.")
    graph.add_argument("--inter-edge-mode", choices=["assignment", "propagate_neighbors"], default=None, help="Cross-batch edge construction mode.")
    graph.add_argument("--mutual-neighbors", type=_bool_value, default=None, help="Keep reciprocal within-batch neighbors only.")
    graph.add_argument("--neighbor-mode", choices=["distance", "rank"], default=None, help="Within-batch neighbor ordering mode.")
    graph.add_argument("--symmetrize", type=_bool_value, default=None, help="Symmetrize final graph.")
    return parser


def _embed(args: argparse.Namespace) -> int:
    estimator_kwargs: dict[str, Any] = {
        "preset": args.preset,
        "rep_key": args.rep_key,
        "batch_key": args.batch_key,
        "label_key": args.label_key,
        "random_state": args.random_state,
        "embedding_method": args.embedding_method,
        "embedding_components": args.embedding_components,
    }
    _set_if_present(estimator_kwargs, "n_components", args.n_components)
    estimator = ClaspEstimator(**estimator_kwargs)

    preprocess_params = estimator.preprocess_defaults
    for name in (
        "n_top_genes",
        "max_cells",
        "min_cell_genes",
        "min_gene_counts",
        "normalize",
        "target_sum",
        "log1p",
        "hvg_flavor",
        "create_artificial_batch",
        "artificial_batch_count",
    ):
        _set_if_present(preprocess_params, name, getattr(args, name))

    graph_params: dict[str, Any] = {}
    for name in (
        "n_neighbors",
        "intra_fraction",
        "n_inter_edges",
        "metric",
        "assignment_quantile",
        "hubness_correction",
        "hubness_k",
        "rank_correction",
        "edge_weighting",
        "inter_edge_mode",
        "mutual_neighbors",
        "neighbor_mode",
        "symmetrize",
    ):
        _set_if_present(graph_params, name, getattr(args, name))

    adata = estimator.to_data(args.input)
    adata = estimator.preprocess(adata, **preprocess_params)
    adata.obsm[args.embedding_key] = estimator.embed(adata, **graph_params)
    estimator.save(adata, args.output)
    if args.figure is not None:
        estimator.plot(adata, embedding_key=args.embedding_key, filename=args.figure)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "embed":
        return _embed(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

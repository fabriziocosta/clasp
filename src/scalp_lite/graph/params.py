from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral, Real


def _coerce_int(value, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    if isinstance(value, Integral):
        result = int(value)
    elif isinstance(value, Real) and float(value).is_integer():
        result = int(value)
    else:
        raise ValueError(f"{name} must be an integer.")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return result


def _coerce_float(value, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a number.")
    return float(value)


@dataclass(frozen=True)
class GraphParams:
    """Validated SCALP-lite graph construction parameters."""

    n_neighbors: int = 15
    intra_fraction: float = 0.5
    n_inter_edges: int = 1
    metric: str = "euclidean"
    assignment_quantile: float | None = 0.95
    hubness_correction: str = "csls"
    hubness_k: int = 10
    edge_weighting: str = "distance"
    mutual_neighbors: bool = True
    neighbor_mode: str = "rank"
    symmetrize: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_neighbors", _coerce_int(self.n_neighbors, name="n_neighbors", minimum=1))
        object.__setattr__(self, "n_inter_edges", _coerce_int(self.n_inter_edges, name="n_inter_edges", minimum=0))
        object.__setattr__(self, "hubness_k", _coerce_int(self.hubness_k, name="hubness_k", minimum=1))

        intra_fraction = _coerce_float(self.intra_fraction, name="intra_fraction")
        if not 0 <= intra_fraction <= 1:
            raise ValueError("intra_fraction must be between 0 and 1.")
        object.__setattr__(self, "intra_fraction", intra_fraction)

        if self.assignment_quantile is not None:
            assignment_quantile = _coerce_float(self.assignment_quantile, name="assignment_quantile")
            if not 0 < assignment_quantile <= 1:
                raise ValueError("assignment_quantile must be in (0, 1] or None.")
            object.__setattr__(self, "assignment_quantile", assignment_quantile)

        if not isinstance(self.metric, str) or not self.metric:
            raise ValueError("metric must be a non-empty string.")
        if self.hubness_correction not in {"none", "csls"}:
            raise ValueError("hubness_correction must be one of: 'none', 'csls'.")
        if self.edge_weighting not in {"distance", "binary"}:
            raise ValueError("edge_weighting must be one of: 'distance', 'binary'.")
        if self.neighbor_mode not in {"rank", "distance"}:
            raise ValueError("neighbor_mode must be one of: 'rank', 'distance'.")
        if not isinstance(self.mutual_neighbors, bool):
            raise ValueError("mutual_neighbors must be True or False.")
        if not isinstance(self.symmetrize, bool):
            raise ValueError("symmetrize must be True or False.")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable parameter dictionary."""
        return asdict(self)

from __future__ import annotations

import pytest

from scalp_lite.optimization import latent_bayesopt


def test_latent_bayesopt_pca_runs_on_mixed_space():
    pytest.importorskip("botorch")
    search_space = {
        "x": {"type": "float", "bounds": [0.0, 1.0]},
        "k": {"type": "int", "bounds": [1, 4]},
        "mode": {"type": "categorical", "values": ["a", "b"]},
    }

    def objective(params):
        mode_bonus = 0.1 if params["mode"] == "b" else 0.0
        return -((params["x"] - 0.7) ** 2) - 0.05 * abs(params["k"] - 3) + mode_bonus

    result = latent_bayesopt(
        objective,
        search_space,
        n_initial=5,
        latent_dim=2,
        n_iterations=1,
        embedding_model="pca",
        random_state=3,
    )

    assert set(result["best_params"]) == set(search_space)
    assert len(result["history"]) == 6
    assert result["latent_points"].shape[1] == 2
    assert result["observed_points"].shape[0] == 6


def test_latent_bayesopt_penalizes_nan_scores():
    pytest.importorskip("botorch")
    search_space = {
        "x": {"type": "float", "bounds": [0.0, 1.0]},
        "k": {"type": "int", "bounds": [1, 3]},
    }

    def objective(params):
        if params["k"] == 1:
            return float("nan")
        return -abs(params["x"] - 0.5)

    result = latent_bayesopt(
        objective,
        search_space,
        n_initial=5,
        latent_dim=2,
        n_iterations=1,
        embedding_model="pca",
        invalid_score=-123.0,
        random_state=2,
    )

    assert result["history"]["score"].notna().all()

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import contextlib
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def _install_known_bo_warning_filters() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"\s*Found Intel OpenMP .* LLVM OpenMP .*",
        category=RuntimeWarning,
        module=r"threadpoolctl",
    )


_install_known_bo_warning_filters()


def _require_botorch():
    try:
        import torch
        from botorch.acquisition import LogExpectedImprovement, UpperConfidenceBound
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Standardize
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood
    except ImportError as exc:
        raise ImportError("latent_bayesopt requires optional dependencies: `pip install -e .[bo]`.") from exc
    return {
        "torch": torch,
        "LogExpectedImprovement": LogExpectedImprovement,
        "UpperConfidenceBound": UpperConfidenceBound,
        "fit_gpytorch_mll": fit_gpytorch_mll,
        "SingleTaskGP": SingleTaskGP,
        "Standardize": Standardize,
        "optimize_acqf": optimize_acqf,
        "ExactMarginalLogLikelihood": ExactMarginalLogLikelihood,
    }


def _require_gplvm():
    try:
        import gpytorch
        import torch
        from gpytorch.distributions import MultivariateNormal
        from gpytorch.likelihoods import GaussianLikelihood
        from gpytorch.mlls import VariationalELBO
        from gpytorch.models.gplvm import BayesianGPLVM, VariationalLatentVariable
        from gpytorch.priors import NormalPrior
        from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
    except ImportError as exc:
        raise ImportError("GPLVM latent BO requires optional dependencies: `pip install -e .[bo]`.") from exc
    return locals()


@dataclass
class EncodedSpace:
    search_space: dict
    names: list[str]
    slices: dict[str, slice]
    dim: int

    @classmethod
    def from_search_space(cls, search_space: dict) -> "EncodedSpace":
        names = list(search_space)
        slices = {}
        start = 0
        for name in names:
            spec = search_space[name]
            if spec["type"] == "categorical":
                width = len(spec["values"])
            else:
                width = 1
            slices[name] = slice(start, start + width)
            start += width
        return cls(search_space=search_space, names=names, slices=slices, dim=start)

    def sample_params(self, rng: np.random.Generator) -> dict:
        params = {}
        for name in self.names:
            spec = self.search_space[name]
            if spec["type"] == "float":
                low, high = spec["bounds"]
                if spec.get("scale") == "log":
                    params[name] = float(np.exp(rng.uniform(np.log(low), np.log(high))))
                else:
                    params[name] = float(rng.uniform(low, high))
            elif spec["type"] == "int":
                low, high = spec["bounds"]
                params[name] = int(rng.integers(low, high + 1))
            elif spec["type"] == "categorical":
                params[name] = spec["values"][int(rng.integers(0, len(spec["values"])))]
            else:
                raise ValueError(f"Unsupported parameter type for {name!r}: {spec['type']!r}")
        return params

    def encode(self, params: dict) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=float)
        for name in self.names:
            spec = self.search_space[name]
            sl = self.slices[name]
            value = params[name]
            if spec["type"] == "float":
                low, high = spec["bounds"]
                if spec.get("scale") == "log":
                    value = np.log(value)
                    low, high = np.log(low), np.log(high)
                vector[sl] = (float(value) - low) / (high - low)
            elif spec["type"] == "int":
                low, high = spec["bounds"]
                vector[sl] = (int(value) - low) / (high - low)
            elif spec["type"] == "categorical":
                index = spec["values"].index(value)
                vector[sl.start + index] = 1.0
        return np.clip(vector, 0.0, 1.0)

    def decode(self, vector: np.ndarray) -> dict:
        vector = np.asarray(vector, dtype=float)
        params = {}
        for name in self.names:
            spec = self.search_space[name]
            values = np.clip(vector[self.slices[name]], 0.0, 1.0)
            if spec["type"] == "float":
                low, high = spec["bounds"]
                value = float(values[0])
                if spec.get("scale") == "log":
                    params[name] = float(np.exp(np.log(low) + value * (np.log(high) - np.log(low))))
                else:
                    params[name] = float(low + value * (high - low))
            elif spec["type"] == "int":
                low, high = spec["bounds"]
                params[name] = int(np.clip(round(low + float(values[0]) * (high - low)), low, high))
            elif spec["type"] == "categorical":
                params[name] = spec["values"][int(np.argmax(values))]
        return params


@dataclass
class PCALatentModel:
    model: PCA
    latent_dim: int

    @classmethod
    def fit(cls, X: np.ndarray, latent_dim: int, random_state: int) -> "PCALatentModel":
        n_components = min(latent_dim, X.shape[0], X.shape[1])
        model = PCA(n_components=n_components, random_state=random_state)
        model.fit(X)
        return cls(model=model, latent_dim=latent_dim)

    def encode(self, X: np.ndarray) -> np.ndarray:
        Z = self.model.transform(X)
        if Z.shape[1] < self.latent_dim:
            Z = np.pad(Z, ((0, 0), (0, self.latent_dim - Z.shape[1])))
        return Z

    def decode(self, Z: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=float)
        return self.model.inverse_transform(Z[:, : self.model.n_components_])


class _GPLVM:
    def __init__(self, X: np.ndarray, latent_dim: int, random_state: int):
        deps = _require_gplvm()
        torch = deps["torch"]
        gpytorch = deps["gpytorch"]
        BayesianGPLVM = deps["BayesianGPLVM"]
        VariationalLatentVariable = deps["VariationalLatentVariable"]
        NormalPrior = deps["NormalPrior"]
        CholeskyVariationalDistribution = deps["CholeskyVariationalDistribution"]
        VariationalStrategy = deps["VariationalStrategy"]
        MultivariateNormal = deps["MultivariateNormal"]
        GaussianLikelihood = deps["GaussianLikelihood"]
        VariationalELBO = deps["VariationalELBO"]

        class GPLVMModel(BayesianGPLVM):
            def __init__(self, X_data, X_init):
                n, data_dim = X_data.shape
                batch_shape = torch.Size([data_dim])
                n_inducing = min(20, n)
                inducing_inputs = torch.randn(data_dim, n_inducing, latent_dim, dtype=torch.double)
                q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=batch_shape)
                q_f = VariationalStrategy(self, inducing_inputs, q_u, learn_inducing_locations=True)
                prior_x = NormalPrior(torch.zeros(n, latent_dim, dtype=torch.double), torch.ones(n, latent_dim, dtype=torch.double))
                latent_x = VariationalLatentVariable(n, data_dim, latent_dim, X_init, prior_x)
                super().__init__(latent_x, q_f)
                self.mean_module = gpytorch.means.ZeroMean(batch_shape=batch_shape)
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(batch_shape=batch_shape),
                    batch_shape=batch_shape,
                )

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return MultivariateNormal(mean_x, covar_x)

        rng = np.random.default_rng(random_state)
        X_init = PCALatentModel.fit(X, latent_dim, random_state).encode(X)
        X_init = X_init + rng.normal(scale=1e-3, size=X_init.shape)
        X_tensor = torch.as_tensor(X, dtype=torch.double)
        self.model = GPLVMModel(X_tensor, torch.as_tensor(X_init, dtype=torch.double)).double()
        self.likelihood = GaussianLikelihood(batch_shape=torch.Size([X.shape[1]])).double()
        self.mll = VariationalELBO(self.likelihood, self.model, num_data=X.shape[0])
        self.torch = torch
        self.target = X_tensor.T

    def fit(self, training_steps: int = 200, lr: float = 0.01) -> "_GPLVM":
        self.model.train()
        self.likelihood.train()
        optimizer = self.torch.optim.Adam(
            [{"params": self.model.parameters()}, {"params": self.likelihood.parameters()}],
            lr=lr,
        )
        for _ in range(training_steps):
            optimizer.zero_grad()
            output = self.model(self.model.X())
            loss = -self.mll(output, self.target).sum()
            loss.backward()
            optimizer.step()
        return self

    def encode(self, X: np.ndarray | None = None) -> np.ndarray:
        q_mu = self.model.X.q_mu.detach().cpu().numpy()
        return q_mu

    def decode(self, Z: np.ndarray) -> np.ndarray:
        self.model.eval()
        self.likelihood.eval()
        z_tensor = self.torch.as_tensor(Z, dtype=self.torch.double)
        with self.torch.no_grad():
            prediction = self.likelihood(self.model(z_tensor)).mean
        return prediction.detach().cpu().numpy().T


@dataclass
class GPLVMLatentModel:
    model: _GPLVM
    latent_dim: int

    @classmethod
    def fit(cls, X: np.ndarray, latent_dim: int, random_state: int) -> "GPLVMLatentModel":
        model = _GPLVM(X, latent_dim, random_state).fit()
        return cls(model=model, latent_dim=latent_dim)

    def encode(self, X: np.ndarray) -> np.ndarray:
        return self.model.encode(X)

    def decode(self, Z: np.ndarray) -> np.ndarray:
        return self.model.decode(Z)


def _fit_latent_model(X: np.ndarray, *, latent_dim: int, embedding_model: str, random_state: int):
    if embedding_model == "pca":
        return PCALatentModel.fit(X, latent_dim, random_state)
    if embedding_model == "gplvm":
        return GPLVMLatentModel.fit(X, latent_dim, random_state)
    raise ValueError("embedding_model must be one of: 'pca', 'gplvm'.")


@dataclass(frozen=True)
class LatentScaler:
    lo: np.ndarray
    width: np.ndarray

    @classmethod
    def fit(cls, Z: np.ndarray) -> "LatentScaler":
        lo = Z.min(axis=0)
        hi = Z.max(axis=0)
        width = np.maximum(hi - lo, 1e-12)
        return cls(lo=lo, width=width)

    def transform(self, Z: np.ndarray) -> np.ndarray:
        return np.clip((Z - self.lo) / self.width, 0.0, 1.0)

    def inverse_transform(self, Z_scaled: np.ndarray) -> np.ndarray:
        return Z_scaled * self.width + self.lo


@contextlib.contextmanager
def _suppress_known_bo_noise():
    with warnings.catch_warnings():
        _install_known_bo_warning_filters()
        warnings.filterwarnings("ignore", message="Data \\(input features\\) is not contained to the unit cube.*")
        yield


def _fit_surrogate(Z: np.ndarray, Y: np.ndarray):
    deps = _require_botorch()
    torch = deps["torch"]
    train_X = torch.as_tensor(Z, dtype=torch.double)
    train_Y = torch.as_tensor(Y.reshape(-1, 1), dtype=torch.double)
    with _suppress_known_bo_noise():
        model = deps["SingleTaskGP"](train_X, train_Y, outcome_transform=deps["Standardize"](m=1))
        mll = deps["ExactMarginalLogLikelihood"](model.likelihood, model)
        deps["fit_gpytorch_mll"](mll)
    return model


def _as_score(value, *, penalty: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        warnings.warn(f"Objective returned a non-numeric score {value!r}; using penalty {penalty}.", RuntimeWarning)
        return float(penalty)
    if not np.isfinite(score):
        warnings.warn(f"Objective returned a non-finite score {score!r}; using penalty {penalty}.", RuntimeWarning)
        return float(penalty)
    return score


def _latent_bounds(Z: np.ndarray, *, margin: float = 0.05):
    deps = _require_botorch()
    torch = deps["torch"]
    lo = np.zeros(Z.shape[1], dtype=float) - margin
    hi = np.ones(Z.shape[1], dtype=float) + margin
    bounds = np.vstack([lo, hi])
    return torch.as_tensor(bounds, dtype=torch.double)


def _propose_latent(Z: np.ndarray, Y: np.ndarray, *, surrogate_model, acquisition: str, batch_size: int, random_state: int) -> np.ndarray:
    deps = _require_botorch()
    torch = deps["torch"]
    bounds = _latent_bounds(Z)
    if acquisition == "ei":
        acq = deps["LogExpectedImprovement"](surrogate_model, best_f=torch.as_tensor(float(np.max(Y)), dtype=torch.double))
    elif acquisition == "ucb":
        acq = deps["UpperConfidenceBound"](surrogate_model, beta=0.2)
    else:
        raise ValueError("acquisition must be one of: 'ei', 'ucb'.")
    try:
        with _suppress_known_bo_noise():
            candidates, _ = deps["optimize_acqf"](
                acq,
                bounds=bounds,
                q=batch_size,
                num_restarts=10,
                raw_samples=128,
                options={"seed": random_state},
            )
        return candidates.detach().cpu().numpy()
    except Exception as exc:
        warnings.warn(f"Acquisition optimization failed; sampling latent candidates randomly. Original error: {exc}", RuntimeWarning)
        rng = np.random.default_rng(random_state)
        lo, hi = bounds.detach().cpu().numpy()
        return rng.uniform(lo, hi, size=(batch_size, Z.shape[1]))


def latent_bayesopt(
    objective_fn: Callable[[dict], float],
    search_space: dict,
    n_initial: int = 30,
    latent_dim: int = 3,
    n_iterations: int = 50,
    embedding_model: str = "gplvm",
    acquisition: str = "ei",
    batch_size: int = 1,
    random_state: int = 0,
    invalid_score: float = -1e9,
):
    """Expensive hyperparameter optimization through Bayesian optimization in latent space."""
    if n_initial < max(2, latent_dim + 1):
        raise ValueError("n_initial must be at least max(2, latent_dim + 1).")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")

    _require_botorch()
    rng = np.random.default_rng(random_state)
    encoded_space = EncodedSpace.from_search_space(search_space)

    params_history = [encoded_space.sample_params(rng) for _ in range(n_initial)]
    observed_points = [encoded_space.encode(params) for params in params_history]
    observed_scores = [_as_score(objective_fn(params), penalty=invalid_score) for params in params_history]
    history_rows = [
        {"iteration": 0, "phase": "initial", "score": score, **params}
        for params, score in zip(params_history, observed_scores)
    ]

    latent_model = None
    surrogate_model = None
    for iteration in range(1, n_iterations + 1):
        X = np.vstack(observed_points)
        Y = np.asarray(observed_scores, dtype=float)
        latent_model = _fit_latent_model(X, latent_dim=latent_dim, embedding_model=embedding_model, random_state=random_state + iteration)
        Z = latent_model.encode(X)
        latent_scaler = LatentScaler.fit(Z)
        Z_scaled = latent_scaler.transform(Z)
        surrogate_model = _fit_surrogate(Z_scaled, Y)
        z_scaled_candidates = _propose_latent(
            Z_scaled,
            Y,
            surrogate_model=surrogate_model,
            acquisition=acquisition,
            batch_size=batch_size,
            random_state=random_state + iteration,
        )
        z_candidates = latent_scaler.inverse_transform(z_scaled_candidates)
        decoded = np.clip(latent_model.decode(z_candidates), 0.0, 1.0)
        for decoded_vector in decoded:
            params = encoded_space.decode(decoded_vector)
            score = _as_score(objective_fn(params), penalty=invalid_score)
            params_history.append(params)
            observed_points.append(encoded_space.encode(params))
            observed_scores.append(score)
            history_rows.append({"iteration": iteration, "phase": "bo", "score": score, **params})

    X = np.vstack(observed_points)
    Y = np.asarray(observed_scores, dtype=float)
    if latent_model is None:
        latent_model = _fit_latent_model(X, latent_dim=latent_dim, embedding_model=embedding_model, random_state=random_state)
    Z = latent_model.encode(X)
    if surrogate_model is None:
        surrogate_model = _fit_surrogate(LatentScaler.fit(Z).transform(Z), Y)

    best_index = int(np.argmax(Y))
    return {
        "best_params": params_history[best_index],
        "best_score": float(Y[best_index]),
        "history": pd.DataFrame(history_rows),
        "latent_points": Z,
        "observed_points": X,
        "observed_scores": Y,
        "models": {
            "latent_model": latent_model,
            "surrogate_model": surrogate_model,
        },
    }

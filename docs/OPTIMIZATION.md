# Latent Bayesian Optimization

SCALP-lite includes an optional hyperparameter optimizer for expensive graph-embedding runs. A single SCALP evaluation can take minutes, so the optimizer avoids direct Bayesian optimization over the full mixed hyperparameter space. Instead, it learns a low-dimensional latent representation of sampled configurations and runs Bayesian optimization there.

The implementation lives in:

- `src/scalp_lite/optimization/latent_bo.py`
- `notebooks/01_latent_bayesopt.ipynb`

The optional dependencies are installed with:

```bash
pip install -e '.[bo]'
```

The notebook performs this installation lazily if BoTorch or GPyTorch are missing.

## Motivation

The preprocessing and graph builder expose several interacting parameters:

- `n_top_genes`
- `n_components`
- `n_neighbors`
- `intra_fraction`
- `n_inter_edges`
- `assignment_quantile`
- `hubness_k`
- `rank_correction`
- `edge_weighting`
- `mutual_neighbors`

Direct Bayesian optimization over this mixed space is expensive and brittle, especially when categorical and integer parameters are present. Latent Bayesian optimization uses an encoder/decoder between the original hyperparameter space and a lower-dimensional continuous space:

$$
x \in \mathbb{R}^{d}
\rightarrow
z \in \mathbb{R}^{q}
$$

with:

$$
q \ll d
$$

Bayesian optimization then works on `z`, and proposed latent points are decoded back to valid SCALP parameters.

## Search Space

The search space is explicit and typed:

```python
search_space = {
    "n_top_genes": {"type": "int", "bounds": [500, 3000]},
    "n_components": {"type": "int", "bounds": [20, 150]},
    "n_neighbors": {"type": "int", "bounds": [5, 40]},
    "intra_fraction": {"type": "float", "bounds": [0.2, 0.9]},
    "n_inter_edges": {"type": "int", "bounds": [1, 8]},
    "assignment_quantile": {"type": "float", "bounds": [0.05, 1.0]},
    "hubness_k": {"type": "int", "bounds": [3, 30]},
    "rank_correction": {"type": "categorical", "values": [False, True]},
    "edge_weighting": {"type": "categorical", "values": ["binary", "distance"]},
    "mutual_neighbors": {"type": "categorical", "values": [False, True]},
}
```

Encoding rules:

- Floats are normalized to `[0, 1]`.
- Log-scaled floats are transformed in log space before normalization.
- Integers are normalized as continuous values internally, then rounded on decode.
- Categoricals are one-hot encoded.

The optimizer always repairs decoded candidates back into the valid search domain before evaluation.

## Algorithm

`latent_bayesopt(...)` runs:

1. Sample `n_initial` random configurations from the typed search space.
2. Evaluate the expensive objective for each configuration.
3. Encode configurations into normalized numeric vectors.
4. Fit a latent model:
   - `embedding_model="pca"` for the baseline.
   - `embedding_model="gplvm"` for the nonlinear GPyTorch GPLVM.
5. Fit a BoTorch `SingleTaskGP` surrogate from latent points to scores.
6. Optimize an acquisition function in latent space.
7. Decode latent candidates to valid hyperparameters.
8. Evaluate the objective.
9. Append observations and repeat.

Mathematically:

$$
f(x) \rightarrow y
$$

where `x` is a SCALP parameter configuration and `y` is the validation score. The latent model learns:

$$
z \mapsto x
$$

and Bayesian optimization fits:

$$
g(z) \approx f(x(z))
$$

## PCA vs GPLVM

The implementation intentionally provides two latent models.

`embedding_model="pca"` is the recommended first run. It is fast, deterministic, and useful for checking that the objective, search space, and acquisition loop behave sensibly.

`embedding_model="gplvm"` uses a GPyTorch Bayesian GPLVM. It is nonlinear and can model curved structure in the parameter space, but it is substantially heavier because the latent model is refit during the optimization loop.

Practical workflow:

```python
result = latent_bayesopt(
    objective_fn,
    search_space,
    n_initial=6,
    latent_dim=3,
    n_iterations=4,
    embedding_model="pca",
)
```

Once the setup is stable:

```python
result = latent_bayesopt(
    objective_fn,
    search_space,
    n_initial=30,
    latent_dim=3,
    n_iterations=50,
    embedding_model="gplvm",
)
```

## Objective Used In The Notebook

`notebooks/01_latent_bayesopt.ipynb` defines a SCALP objective that:

1. Copies a preprocessed `AnnData`.
2. Embeds it with candidate graph parameters.
3. Scores the embedding.
4. Returns a scalar to maximize.

The default objective is:

$$
\mathrm{score}
=
\mathrm{knn\_label\_agreement}
+ 0.25 \cdot \mathrm{batch\_mixing}
- 0.05 \cdot \mathrm{graph\_density}
$$

This favors biological label coherence, modest batch mixing, and less dense graphs. These weights are intentionally exposed in the notebook because the right tradeoff depends on the dataset.

## Return Value

`latent_bayesopt(...)` returns:

```python
{
    "best_params": best_x,
    "best_score": best_y,
    "history": history_dataframe,
    "latent_points": Z,
    "observed_points": X,
    "observed_scores": Y,
    "models": {
        "latent_model": latent_model,
        "surrogate_model": surrogate_model,
    },
}
```

`history` is the main object to inspect during exploratory runs. It records every evaluated configuration and score.

## Current Limitations

- The objective still runs full SCALP embedding, so optimization is expensive.
- GPLVM refitting is heavier than PCA and should be used only after validating the search space.
- The optimizer assumes larger scores are better.
- Duplicate decoded configurations are currently allowed.
- Cross-batch assignment in SCALP still uses dense Hungarian costs, so very large datasets should use small `max_cells` during optimization.

## Recommended Usage

1. Start with `max_cells=1000` or `2000`.
2. Use `embedding_model="pca"`.
3. Run a small budget such as `n_initial=6`, `n_iterations=4`.
4. Inspect `history`.
5. Increase budget only after the objective behaves sensibly.
6. Switch to `embedding_model="gplvm"` for a final nonlinear search if PCA suggests useful structure.
